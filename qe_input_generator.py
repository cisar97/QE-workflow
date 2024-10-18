#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from ase.io import read
import re

def get_celldm(rho, nat, mass=1.6735575e-24, cm_to_au=188972598.85789) : 
    # Compute dimension of the cell from density 
    # rho = density in [g/cm^3] 
    # nat = number of atoms in the cell 
    # mass = mass in [g] of the element
    # cm_to_au = measure coonversion from cm to bohr 
    # returns length of the cubic cell in [Bohr] 

    celldm = (((nat*mass) / rho)**(1/3)) * cm_to_au
    return np.round(celldm, 6)  

def get_last_MDstep(file_name):
    # Get the total number of MD step from a Quantum Espresso MD output file
    step_pattern = re.compile(r'^\s*Entering Dynamics:\s+iteration\s*=\s*\d+', re.MULTILINE)
    step_count = 0
    with open(file_name, 'r') as file:
        for line in file:
            if step_pattern.search(line):
                step_count += 1

    return step_count

def get_atoms_from_file(file_name, config, convert_to_bohr=True):
    # Read atomic positions from a file and save them in a numpy array
    # Default is to convert the positions from Angstrom to Bohr
    # config : numpy array of config index
    # return list of array, each the cordinates of the atoms at given config
    Atomic_pos = []
    if not isinstance(config, list):
        atom = read(file_name, index=config)
        at_pos = atom.get_positions()
        if convert_to_bohr:
            at_pos *= 1.8897259886
        Atomic_pos.append(at_pos)
    else :
        atoms = read(file_name, index=f"{config[0]}:{config[-1]+1}")
        for i in config :
            atom = atoms[i-config[0]]
            at_pos = atom.get_positions()
            if convert_to_bohr:
                at_pos *= 1.8897259886
            Atomic_pos.append(at_pos)
    return Atomic_pos

def get_atoms_random(nat, celldm, seed): 
    # Generate random atomic positions in a cubic cell of length celldm
    # Output a list with one array to be consistent with the list outputted
    # by get_atoms_from_file
    np.random.seed(seed)
    atomic_pos = np.random.uniform(0., celldm, (nat,3))
    return [atomic_pos]

def electronic_temp(T, kB=8.617333e-5, ry_to_ev=13.605698) : 
    # Compute degauss paramemtr for smearing corresponding 
    # to the desired electronic temperature
    # T : Temperature in [K]
    return kB*T/ry_to_ev

def write_qe_input(qe_input, atom, file_name):
    # Write a Quantum Espresso input file from a dictionary
    # qe_input = dictionary with the input parameters
    # atom = atomic symbol
    f = open(file_name, 'w')
    for Key in qe_input : 
        if Key == 'ATOMIC_SPECIES':
            print(f"\n{Key}", file=f)
            print(qe_input[Key], file=f)
            continue
        if Key == 'ATOMIC_POSITIONS':
            print(f"\n{Key}  {qe_input[Key]['length_metric']}", file=f)
            for i in range(qe_input['&SYSTEM']['nat']):
                print(f"{atom}  {qe_input[Key]['atomic_pos'][i,0]:.8f}  {qe_input[Key]['atomic_pos'][i,1]:.8f}  {qe_input[Key]['atomic_pos'][i,2]:.8f}", file=f)
            continue
        if Key == 'K_POINTS':
            print(f"\n{Key}  {qe_input[Key]['kpoints_name']}", file=f)
            print(qe_input[Key]['kpoints'], file=f)
            continue
        else : 
            print(f"\n{Key}", file=f)
            for keys, value in qe_input[Key].items():
                print(f"{keys} = {value}", file=f)
            print(f"/", file=f)
    f.close()


def build_qe_input(calculation, prefix, celldm1, ecutwfc, ecutrho, pseudo, atomic_pos, file_name, nstep=1000 , dt=10,
                   Temp='1000', thermo='svr', restart_mode='from_scratch', pseudo_dir='/users/ccozza/pseudo', outdir='./',  
                   tpstress='.true.', tprnfor='.true.', ibrav=1,  nat=128, ntyp=1, occupations='smearing', 
                   smearing='gaussian', degauss=0.002, input_dft='pbe', conv_thr=1.0e-6, ndim=8, beta=0.7, diagonalization='david',
                   ion_dynamics='verlet', ion_temperature='svr', nraise=1, refold_pos='.false.', atom='H', 
                   mass=1.000, kpoints_name='automatic', kpoints='3 3 3 1 1 1', length_metric='bohr') :
    # Build a dictionary with the input parameters for Quantum Espresso  
    if thermo == 'langevin' or thermo == 'langevin-smc':
        ion_dynamics = thermo
    else:
        ion_dynamics = "verlet"
        ion_temperature = thermo
        nraise = nraise

    qe_input = {
        "&CONTROL" : {
            "calculation" : f"'{calculation}'",
            "restart_mode" : f"'{restart_mode}'", 
            "prefix" : f"'{prefix}'",
            "pseudo_dir" : f"'{pseudo_dir}'",
            "outdir" : f"'{outdir}'",
            "nstep" : nstep,
            "dt" : dt,
            "tstress" : tpstress,
            "tprnfor" : tprnfor,
        },
        "&SYSTEM" : {
            "ibrav" : ibrav,
            "celldm(1)" : celldm1,
            "nat" : nat,
            "ntyp" : ntyp,
            "ecutwfc" : ecutwfc,
            "ecutrho" : ecutrho,
            "occupations" : f"'{occupations}'",
            "smearing" : f"'{smearing}'",
            "degauss" : degauss,
            "input_dft" : f"'{input_dft}'"
        },
        "&ELECTRONS" : {
            "conv_thr" : conv_thr,
            "mixing_ndim" : ndim,
            "mixing_beta" : beta,
            "diagonalization" : f"'{diagonalization}'"
        },
        "&IONS" : {
            "ion_dynamics" : f"'{ion_dynamics}'",
            "ion_temperature" : f"'{ion_temperature}'",
            "nraise" : nraise,
            "tempw" : Temp,
            "refold_pos" : refold_pos
        },
        "&CELL" : {},
        "ATOMIC_SPECIES" : f"{atom}  {mass:.3f}  {pseudo}",
        "ATOMIC_POSITIONS" : {
            'length_metric' : length_metric,
            'atomic_pos' : atomic_pos[0]
        },
        "K_POINTS" : {
            'kpoints_name' : kpoints_name,
            'kpoints' : kpoints
        }
    }

    if thermo == 'langevin' or thermo == 'langevin-smc':
        del qe_input["&IONS"]["ion_temperature"]
        del qe_input["&IONS"]["nraise"] 
    if calculation != "md":
        del qe_input["&CONTROL"]["restart_mode"]
        del qe_input["&CONTROL"]["nstep"], qe_input["&CONTROL"]["dt"]
        del qe_input["&IONS"]

    write_qe_input(qe_input, atom, file_name)

def replace_string_in_file(file_path, old_string, new_string):
    try:
        # Read the file contents
        with open(file_path, 'r') as file:
            file_content = file.read()
        # Perform the replacement
        new_content = file_content.replace(old_string, new_string)
        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.write(new_content)

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"Failed to modify the file: {e}")


