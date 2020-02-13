'''
Based on :class:`OTF` - it is the user's responsibility to feed this with
calculations that train the model.
'''
import os
import sys
import inspect
from copy import deepcopy

from flare.struc import Structure
from flare.util import is_std_in_bound
from flare.mgp.utils import get_l_bound

import numpy as np
from ase.md.md import MolecularDynamics
from ase.calculators.calculator import all_changes, Calculator
from ase import units


class OTF_Calculator(Calculator):
    """
    OTF (on-the-fly) training with the ASE interface. 
    
    Note: Dft calculator is set outside of the otf module, and input as 
        dft_calc, so that different calculators can be used

    Args:
        flare_calc (ASE Calculator): the ASE FLARE calculator
        dft_calc (ASE Calculator): the ASE DFT calculator (see ASE documentaion)
        dft_count (int): initial number of DFT calls
        std_tolerance_factor (float): the threshold of calling DFT = noise * 
            std_tolerance_factor
        init_atoms (list): the list of atoms in the first DFT call to add to
            the training set, since there's no uncertainty prediction initially
        calculate_energy (bool): if True, the energy will be calculated;
            otherwise, only forces will be predicted
        max_atoms_added (int): the maximal number of atoms to add to the 
            training set after each DFT calculation
        freeze_hyps (int or None): the hyperparameters will only be trained for
            the first `freeze_hyps` DFT calls, and will be fixed after that
        restart_from (str or None): the path of the directory that stores the
            training data from last OTF run, and this OTF will restart from it

    Other Parameters:
        use_mapping (bool): if True, the MGP will be used
        non_mapping_steps (list): a list of steps that MGP will not be 
            constructed and used
        l_bound (float): the lower bound of the interatomic distance, used for 
            MGP construction
        two_d (bool): used in the calculation of l_bound. If 2-D material is 
            considered, set to True, then the atomic environment construction 
            will only search the x & y periodic boundaries to save time
    """


    implemented_properties = ["forces", "energy", "stress"]

    nsteps = 0
    _is_init = False

    def __init__(self, 
            # ASE calculator parameters
            restart=None, # TODO
            ignore_bad_restart_file=False, # TODO
            label=None, # TODO
            atoms=None,
            directory='.', # TODO
            # FLARE parameters
            flare_calc=None,
            # on-the-fly parameters
            dft_calc=None, dft_count=None, std_tolerance_factor: float=1, 
            skip: int=0, init_atoms: list=[], calculate_energy=False, 
            max_atoms_added=1, freeze_hyps=1, restart_from=None,
            # mgp parameters
            use_mapping: bool=False, non_mapping_steps: list=[],
            l_bound: float=None, two_d: bool=False):

        super().__init__(atoms=atoms)

        # get all arguments as attributes 
        arg_dict = inspect.getargvalues(inspect.currentframe())[3]
        del arg_dict['self']
        self.__dict__.update(arg_dict)

        if dft_count is None:
            self.dft_count = 0
        self.noa = len(self.atoms.positions)

        # initialize local energies
        if calculate_energy:
            self.local_energies = np.zeros(self.noa)
        else:
            self.local_energies = None

        # initialize otf
        if init_atoms is None:
            self.init_atoms = [int(n) for n in range(self.noa)]

        self.observers = []

    def init_calculate(self):
        # observers
        for i, obs in enumerate(self.observers):
            if obs[0].__class__.__name__ == "OTFLogger":
                self.logger_ind = i
                break

        # restart from previous OTF training
        if self.restart_from is not None:
            self.restart()
            self._f = self.flare_calc.results['forces']

        # initialize gp by a dft calculation
        if not self.flare_calc.gp_model.training_data:
            self.dft_count = 0
            self.stds = np.zeros((self.noa, 3))
            dft_forces = self.call_DFT()
            self._f = dft_forces
   
            # update gp model
            curr_struc = Structure.from_ase_atoms(self.atoms)
            self.l_bound = get_l_bound(100, curr_struc, self.two_d)
            print('l_bound:', self.l_bound)

            self.flare_calc.gp_model.update_db(curr_struc, dft_forces,
                           custom_range=self.init_atoms)

            # train calculator
            for atom in self.init_atoms:
                # the observers[0][0] is the logger
                self.observers[self.logger_ind][0].add_atom_info(atom, 
                    self.stds[atom])
            self.train()
            self.observers[self.logger_ind][0].write_wall_time()
  
    def calculate(self, atoms=None, properties=['forces'],
            system_changes=all_changes):
        print("calculation #{}".format(self.nsteps))
        super().calculate(atoms=atoms, properties=properties,
            system_changes=system_changes)
        self.atoms.set_calculator(self) # ??? this is stupid
        self.observers[0][0].atoms = self.atoms
        # self.flare_calc.atoms = self.atoms

        if not self._is_init:
          self.init_calculate()
          self._is_init = True

        self.results = {} # clear the calculation from last step
        self.stds = np.zeros((self.noa, 3))

        for prop in properties:
            self.flare_calc.get_property(prop, atoms=self.atoms)
        self.results = self.flare_calc.results.copy()
        print("actual GP (FLARE) forces:")
        print(self.results["forces"])

        self.nsteps += 1
        self.stds = self.flare_calc.get_uncertainties(self.atoms)

        # figure out if std above the threshold
        self.call_observers()
        curr_struc = Structure.from_ase_atoms(self.atoms)
        self.l_bound = get_l_bound(self.l_bound, curr_struc, self.two_d)
        print('l_bound:', self.l_bound)
        curr_struc.stds = np.copy(self.stds)
        noise = self.flare_calc.gp_model.hyps[-1]
        self.std_in_bound, self.target_atoms = is_std_in_bound(\
                noise, self.std_tolerance_factor, curr_struc, self.max_atoms_added)

        print('std in bound:', self.std_in_bound, self.target_atoms)
        #self.is_std_in_bound([])

        if not self.std_in_bound:
            # call dft/eam
            print('calling dft')
            dft_forces = self.call_DFT()

            # update gp
            print('updating gp')
            self.update_GP(dft_forces)

    def finialize_calculate(self):
        """never actually called yet... just for documentation's sake"""
        self.observers[self.logger_ind][0].run_complete()
    
    def call_DFT(self):
        prev_calc = self.atoms.calc
        calc = deepcopy(self.dft_calc)
        self.atoms.set_calculator(calc)
        forces = self.atoms.get_forces()
        self.results = calc.results.copy()
        print("dft done")
        self.call_observers()
        self.atoms.set_calculator(prev_calc)
        self.dft_count += 1
        return forces

    def update_GP(self, dft_forces):
        atom_count = 0
        atom_list = []
        gp_model = self.flare_calc.gp_model

        # build gp structure from atoms
        atom_struc = Structure.from_ase_atoms(self.atoms)

        while (not self.std_in_bound and atom_count <
               np.min([self.max_atoms_added, len(self.target_atoms)])):

            target_atom = self.target_atoms[atom_count]

            # update gp model
            gp_model.update_db(atom_struc, dft_forces,
                               custom_range=[target_atom])
    
            if gp_model.alpha is None:
                gp_model.set_L_alpha()
            else:
                gp_model.update_L_alpha()

            # atom_list.append(target_atom)
            ## force calculation needed before get_uncertainties
            # forces = self.atoms.calc.get_forces_gp(self.atoms) 
            # self.stds = self.atoms.get_uncertainties()

            # write added atom to the log file, 
            # refer to ase.optimize.optimize.Dynamics
            self.observers[self.logger_ind][0].add_atom_info(target_atom, 
                                               self.stds[target_atom])
           
            #self.is_std_in_bound(atom_list)
            atom_count += 1

        self.train()
        self.observers[self.logger_ind][0].added_atoms_dat.write('\n')
        self.observers[self.logger_ind][0].write_wall_time()

    def train(self, output=None, skip=False):
        calc = self.flare_calc
        if (self.dft_count-1) < self.freeze_hyps:
            #TODO: add other args to train()
            calc.gp_model.train(output=output)
            self.observers[self.logger_ind][0].write_hyps(calc.gp_model.hyp_labels, 
                            calc.gp_model.hyps, calc.gp_model.likelihood, 
                            calc.gp_model.likelihood_gradient)
        else:
            #TODO: change to update_L_alpha()
            calc.gp_model.set_L_alpha()

        # build mgp
        if self.use_mapping:
            if self.get_time() in self.non_mapping_steps:
                skip = True

            calc.build_mgp(skip)

        np.save('ky_mat_inv', calc.gp_model.ky_mat_inv)
        np.save('alpha', calc.gp_model.alpha)

    def restart(self):
        # Recover atomic configuration: positions, velocities, forces
        positions, self.nsteps = self.read_frame('positions.xyz', -1)
        self.atoms.set_positions(positions)
        self.atoms.set_velocities(self.read_frame('velocities.dat', -1)[0])
        self.flare_calc.results['forces'] = self.read_frame('forces.dat', -1)[0]
        print('Last frame recovered')

        # Recover training data set
        gp_model = self.flare_calc.gp_model
        atoms = deepcopy(self.atoms)
        nat = len(self.atoms.positions)
        dft_positions = self.read_all_frames('dft_positions.xyz', nat)
        dft_forces = self.read_all_frames('dft_forces.dat', nat)
        added_atoms = self.read_all_frames('added_atoms.dat', 1, 1, 'int')
        for i, frame in enumerate(dft_positions):
            atoms.set_positions(frame)
            curr_struc = Structure.from_ase_atoms(atoms)
            gp_model.update_db(curr_struc, dft_forces[i], added_atoms[i])
        gp_model.set_L_alpha()
        print('GP training set ready')

        # Recover FLARE calculator
        gp_model.ky_mat_inv = np.load(self.restart_from+'/ky_mat_inv.npy')
        gp_model.alpha = np.load(self.restart_from+'/alpha.npy')
        if self.flare_calc.use_mapping:
            for map_3 in self.flare_calc.mgp_model.maps_3:
                map_3.load_grid = self.restart_from + '/'
            self.flare_calc.build_mgp(skip=False)
        print('GP and MGP ready')

        self.l_bound = 10

    def read_all_frames(self, filename, nat, header=2, elem_type='xyz'):
        frames = []
        with open(self.restart_from+'/'+filename) as f:
            lines = f.readlines()
            frame_num = len(lines) // (nat+header)
            for i in range(frame_num):
                start = (nat+header) * i + header
                curr_frame = lines[start:start+nat]
                properties = []
                for line in curr_frame:
                    line = line.split()
                    if elem_type == 'xyz':
                        xyz = [float(l) for l in line[1:]]
                        properties.append(xyz)
                    elif elem_type == 'int':
                        properties = [int(l) for l in line]
                frames.append(properties)
        return np.array(frames)


    def read_frame(self, filename, frame_num):
        nat = len(self.atoms.positions)
        with open(self.restart_from+'/'+filename) as f:
            lines = f.readlines()
            if frame_num == -1: # read the last frame
                start_line = - (nat+2)
                frame = lines[start_line:]
            else:
                start_line = frame_num * (nat+2)
                end_line = (frame_num+1) * (nat+2)
                frame = f.lines[start_line:end_line]

            properties = []
            for line in frame[2:]:
                line = line.split()
                properties.append([float(d) for d in line[1:]])
        return np.array(properties), len(lines)//(nat+2)

    def call_observers(self):
        for obs in self.observers:
            obs[0].atoms = self.atoms
            obs[0]()

