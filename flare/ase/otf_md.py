'''
This module provides OTF training with ASE MD engines: VerlocityVerlet, NVTBerendsen, NPTBerendsen, NPT and Langevin. 
Please see the function `otf_md` below for usage
'''
import os
import sys
from flare.struc import Structure
from flare.ase.otf_calculator import OTF_Calculator

import numpy as np
from ase.calculators.espresso import Espresso
from ase.calculators.eam import EAM
from ase.md.npt import NPT
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.verlet import VelocityVerlet
from ase.md.md import MolecularDynamics
from ase.md.langevin import Langevin
from ase import units

def otf_md(md_engine: str, atoms, md_params: dict, otf_params: dict):
    '''
    Create an OTF MD engine 
    
    Args:
        md_engine (str): the name of md engine, including `VelocityVerlet`,
            `NVTBerendsen`, `NPTBerendsen`, `NPT`, `Langevin`
        atoms (Atoms): ASE Atoms to apply this md engine
        md_params (dict): parameters used in MD engines, 
            must include: `timestep`, `trajectory` (usually set to None).
            Also include those parameters required for ASE MD engine, 
            please look at ASE website to find out parameters for different engines
        otf_params (dict): parameters used in OTF module

    Return:
        An OTF MD class object

    Example:
        >>> from ase import units
        >>> from ase.spacegroup import crystal
        >>> super_cell = crystal(['Ag', 'I'],  
                                 basis=[(0, 0, 0), (0.5, 0.5, 0.5)],
                                 size=(2, 1, 1),
                                 cellpar=[3.85, 3.85, 3.85, 90, 90, 90])
        >>> md_engine = 'VelocityVerlet'
        >>> md_params = {'timestep': 1 * units.fs, 'trajectory': None, 
                         'dt': None} 
        >>> otf_params = {'dft_calc': dft_calc, 
                          'init_atoms': [0],
                          'std_tolerance_factor': 1, 
                          'max_atoms_added' : len(super_cell.positions),
                          'freeze_hyps': 10, 
                          'use_mapping': False}
        >>> test_otf = otf_md(md_engine, super_cell, md_params, otf_params)
    '''

    md = md_params
    timestep = md['timestep']
    trajectory = md['trajectory']

    calc = OTF_Calculator(**otf_params)
    atoms.set_calculator(calc)

    if md_engine == 'VelocityVerlet':
        md_obj = VelocityVerlet(atoms, timestep, trajectory, dt=md['dt'])
       
    elif md_engine == 'NVTBerendsen':
        md_obj = NVTBerendsen(atoms, timestep, md['temperature'], 
                md['taut'], md['fixcm'], trajectory)
    
    elif md_engine == 'NPTBerendsen':
        md_obj = NPTBerendsen(atoms, timestep, md['temperature'], 
                md['taut'], md['pressure'], md['taup'], 
                md['compressibility'], md['fixcm'], trajectory)

    elif md_engine == 'NPT':
        md_obj = NPT(atoms, timestep, md['temperature'],
                md['externalstress'], md['ttime'], md['pfactor'], 
                md['mask'], trajectory)

    elif md_engine == 'Langevin':
        md_obj = Langevin(atoms, timestep, md['temperature'],
                md['friction'], md['fixcm'], trajectory)

    else:
        raise NotImplementedError(md_engine+' is not implemented')

    atoms.calc.md = md_obj # actually only required if MGP is used
    return md_obj
