import numpy as np
import multiprocessing as mp
import concurrent.futures
import copy
import sys
from flare.env import AtomicEnvironment
from flare.struc import Structure
from flare.mgp.mgp import MappedGaussianProcess
from ase.calculators.calculator import Calculator

class FLARE_Calculator(Calculator):
    def __init__(self, gp_model, mgp_model, par=False, use_mapping=False):
        super().__init__() # all set to default values,TODO: change
        self.mgp_model = mgp_model
        self.gp_model = gp_model
        self.use_mapping = use_mapping
        self.par = par
        self.results = {}

    def get_potential_energy(self, atoms=None, force_consistent=False):
        # TODO: to be implemented
        return 1

        nat = len(atoms)
        struc_curr = Structure(np.array(atoms.cell), 
                               atoms.get_atomic_numbers(),
                               atoms.positions)
        local_energies = np.zeros(nat)

        for n in range(nat):
            chemenv = AtomicEnvironment(struc_curr, n,
                                        self.gp_model.cutoffs)
            local_energies[n] = self.gp_model.predict_local_energy(chemenv)

        return np.sum(local_energies)

    def get_forces(self, atoms):
        if self.use_mapping:
            return self.get_forces_mgp(atoms)
        else:
            return self.get_forces_gp(atoms)

    def get_forces_gp(self, atoms):
        nat = len(atoms)
        struc_curr = Structure(np.array(atoms.cell), 
                               atoms.get_atomic_numbers(),
                               atoms.positions)

        if self.par:
            res = predict_on_structure_par(struc_curr, self.gp_model)
        else:
            res = predict_on_structure(struc_curr, self.gp_model)

        forces = np.zeros((nat, 3))
        stds = np.zeros((nat, 3))
        for n in range(nat):
            forces[n] = res[n][0]
            stds[n] = res[n][1]

        self.results['stds'] = stds
        atoms.get_uncertainties = self.get_uncertainties

        return forces

    def get_forces_mgp(self, atoms):
        nat = len(atoms)
        struc_curr = Structure(np.array(atoms.cell), 
                               atoms.get_atomic_numbers(),
                               atoms.positions)

        forces = np.zeros((nat, 3))
        stds = np.zeros((nat, 3))
        for n in range(nat):
            chemenv = AtomicEnvironment(struc_curr, n,
                                        self.mgp_model.GP.cutoffs)
            f, v = self.mgp_model.predict(chemenv, mean_only=False)
            forces[n] = f
            stds[n] = np.sqrt(np.absolute(v))

        self.results['stds'] = stds
        atoms.get_uncertainties = self.get_uncertainties
        return forces

    def get_stress(self, atoms):
        return np.eye(3)

    def calculation_required(self, atoms, quantities):
        return True

    def get_uncertainties(self):
        return self.results['stds']

    def train_gp(self, monitor=True):
        self.gp_model.train(monitor)

    def build_mgp(self, skip=True):
        # l_bound not implemented

        if skip and (self.curr_step in self.non_mapping_steps):
            return 1

        # set svd rank based on the training set, grid number and threshold 1000
        grid_params = self.mgp_model.grid_params
        struc_params = self.mgp_model.struc_params

        train_size = len(self.gp_model.training_data)
        rank_2 = np.min([1000, grid_params['grid_num_2'], train_size*3])
        rank_3 = np.min([1000, grid_params['grid_num_3'][0]**3, train_size*3])
        grid_params['svd_rank_2'] = rank_2
        grid_params['svd_rank_3'] = rank_3
       
        self.mgp_model = MappedGaussianProcess(self.gp_model, grid_params, struc_params)

def predict_on_atom(params):
    structure, atom, gp_model = params
    chemenv = AtomicEnvironment(structure, atom, gp_model.cutoffs)
    comps = np.zeros(3)
    stds = np.zeros(3)
    # predict force components and standard deviations
    for i in range(3):
        force, var = gp_model.predict(chemenv, i+1)
        comps[i] = float(force)
        stds[i] = np.sqrt(np.abs(var))

    return comps, stds

def predict_on_atom_en(params):
    structure, atom, gp_model = params
    chemenv = AtomicEnvironment(structure, atom, gp_model.cutoffs)
    comps = np.zeros(3)
    stds = np.zeros(3)
    # predict force components and standard deviations
    for i in range(3):
        force, var = gp_model.predict(chemenv, i+1)
        comps[i] = float(force)
        stds[i] = np.sqrt(np.abs(var))

    # predict local energy
    local_energy = gp_model.predict_local_energy(chemenv)
    return comps, stds, local_energy

def predict_on_structure_par(structure, gp_model):
    atom_list = [(structure, atom, gp_model) for atom in range(structure.nat)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        res = list(executor.map(predict_on_atom, atom_list))
        return res

def predict_on_structure_par_en(self):
    atom_list = [(structure, atom, gp_model) for atom in range(structure.nat)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        res = list(executor.map(predict_on_atom_en, atom_list))
        return res

def predict_on_structure(structure, gp_model):
    res = []
    for atom in range(structure.nat):
        chemenv = AtomicEnvironment(structure, atom, gp_model.cutoffs)
        res.append(predict_on_atom((structure, atom, gp_model)))
    return res

def predict_on_structure_en(strucutre, gp_model):
    res = []
    for atom in range(structure.nat):
        chemenv = AtomicEnvironment(structure, atom, gp_model.cutoffs)
        res.append(predict_on_atom_en((structure, atom, gp_model)))        
    return res