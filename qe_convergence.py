#! /usr/bin/env python3 
# -*- coding: utf-8 -*-

from qe_input_generator import get_atoms_from_file, build_qe_input
from qe_postproc import gcm3_to_rs
import os, glob, re, shutil
import numpy as np 
from ase.io import read, write 

ang_to_bohr = 0.5291772109 # used in turbo.
amu_to_g = 1.660539066 * 10**(-27)
ang_to_cm = 10**(-8)

def write_dataset_atom(project_dir, MDfile, T, rho, config, qe_input_dir_name="dataset",) : 
    ## Inside the home directory provided as input create a /dataset directory 
    ## containing the atomic configurations extracted from the MD trajectory
    ## provided as input. The atomic configurations are stored as .xsf files
    # homedir : path to the home directory
    # MDfile : path to the MD trajectory file
    # config : LIST of configurations to extract from the MD trajectory
    os.makedirs(f"{project_dir}/{qe_input_dir_name}", exist_ok=True)
    rs = gcm3_to_rs(rho)
    atoms = get_atoms_from_file(MDfile, config)
    for i, atom in enumerate(atoms) :
        write(f"{project_dir}/dataset/input_T{T}_d{rs}_{config[i]}.xsf", atom)

def generate_distorted_configurations(project_dir, qe_input_dir_name="dataset", rs_pm_list=[-0.05, -0.02, -0.01, 0.00, +0.01, +0.02, +0.05], indshift=0) :
    ## Generate a set of distorted atomic configurations from the atomic configurations
    ## stored in the /dataset directory. The distorted configurations are stored in the 
    ## qe_input_dir directory.
    # all the other scripts assume the followings
    # 1) the number of rs points is odd
    # 2) the median rs is 0.00
    # check them
    if len(rs_pm_list)%2 != 1:
        raise ValueError
    if rs_pm_list[int(len(rs_pm_list)/2)] != 0.00:
        raise ValueError

    qe_input_dir=os.path.join(project_dir, qe_input_dir_name)
    qe_input_list=glob.glob(os.path.join(qe_input_dir, "*.xsf"))
    qe_input_list=sorted(qe_input_list, key=lambda s: tuple(map(float, re.search(r'.*input_T(\d+)_d(\d+(?:\.\d+)?)_(\d+).xsf', s).groups())))
    qe_prefix_list=[os.path.splitext(os.path.basename(qe_input))[0] for qe_input in qe_input_list]
    qe_atoms_list=[read(qe_input, format="xsf") for qe_input in qe_input_list]
    struct_dir=os.path.join(project_dir, "struct_input_dir")
    os.makedirs(struct_dir, exist_ok=True)

    for ind, (qe_prefix, qe_atoms) in enumerate(zip(qe_prefix_list, qe_atoms_list)):
        s_ind_dir=os.path.join(struct_dir, f"{ind+1+indshift:04d}_{qe_prefix.replace('input_','')}")
        os.makedirs(s_ind_dir, exist_ok=True)
        write(os.path.join(s_ind_dir, f"org_struct_{qe_prefix.replace('input_','')}.xsf"), qe_atoms)
        fname=f"org_struct_{qe_prefix.replace('input_','')}.xsf"
        temp_label_org, rs_label_org, steps_label_org=re.search(r'org_struct_T(\d+)_d(\d+(?:\.\d+)?)_(\d+).xsf', fname).groups()

        atoms = read(os.path.join(s_ind_dir, f"org_struct_{qe_prefix.replace('input_','')}.xsf"))
        num_ele_org = np.sum(atoms.get_atomic_numbers())
        vol_org = atoms.get_volume() / ang_to_bohr**3
        rs_org = (3.0 * vol_org / (num_ele_org * 4 * np.pi))**(1.0/3.0)

        if np.abs(rs_org - float(rs_label_org)) > 1.0e-2:
            print("Error! rs_org is inconsistent with rs_label_org.")
            print(f"rs_org = {rs_org}, while rs_label_org = {float(rs_label_org)}.")
            raise ValueError

        for rs_ind, rs_pm in enumerate(rs_pm_list):
            rs = rs_org + rs_pm
            rs_label = f"{rs:.3f}"
            vol_after = num_ele_org * 4 * np.pi * rs**3 / 3.0
            cell=atoms.get_cell()
            rescaled_cell=atoms.get_cell() * (vol_after/vol_org) ** (1.0/3.0)
            s_atoms=atoms.copy()
            s_atoms.set_cell(rescaled_cell, scale_atoms=True)
            #print(f"rs_label={rs_label}")
            rs_fname=f"{rs_ind+1:02}_struct_T{temp_label_org}_d{rs_label}_{steps_label_org}.xsf"
            write(os.path.join(s_ind_dir, rs_fname), s_atoms)


from ase.calculators.espresso import Espresso
def compute_qe_distorted_conf(project_dir, scratch_dir, xc_name, pp_prefix, pp, ecutwfc, pseudo_NC=True, kpts=(3,3,3),koffset=(1,1,1),
                              rs_tasks=1.310, struct_dir_name="struct_input_dir", qe_input_dir_name="dataset"):  
    pseudopotentials = {'H':pp}
    if pseudo_NC==True : ecutrho = 4*ecutwfc
    os.change_dir(project_dir)
    os.makedirs(scratch_dir, exist_ok=True)
    struct_dir=os.path.join(project_dir, struct_dir_name)
    qe_results_dir=os.path.join(project_dir, f"qe_results_xc_{xc_name}_pp_{pp_prefix}_ecutw_{ecutwfc}_dir")
    print(f"Working directory : {qe_results_dir} \n\n")
    struct_dirname_list=[os.path.basename(d) for d in glob.glob(os.path.join(struct_dir, '*')) if os.path.isdir(d)]
    os.makedirs(struct_dir, exist_ok=True)
    os.makedirs(qe_results_dir, exist_ok=True)
    
    for struct_dirname in struct_dirname_list:
        target_struct_dir=os.path.join(struct_dir, struct_dirname)
        xsf_filename_list=[os.path.basename(xsf) for xsf in glob.glob(os.path.join(target_struct_dir, '*.xsf')) if re.match(r'^\d+.*', os.path.basename(xsf))]
        for xsf_filename in xsf_filename_list:
            xsf_file = os.path.join(target_struct_dir, xsf_filename)
            atoms=read(xsf_file)
            rs_index, temp_label, rs_label, steps_labelt=re.search(r'(\d+)_struct_T(\d+)_d(\d+(?:\.\d+)?)_(\d+).xsf', xsf_filename).groups()
            qe_work_dir=os.path.join(qe_results_dir, struct_dirname, f"rs_index_{rs_index}")
            os.makedirs(qe_work_dir, exist_ok=True)
            shutil.copyfile(os.path.join(xsf_file), os.path.join(qe_work_dir, xsf_filename))
            os.chdir(qe_work_dir)

            if float(rs_label) >= rs_tasks :
                tasks = 64
            else :
                tasks = 48

            #ASE_ESPRESSO_COMMAND
            os.environ["ASE_ESPRESSO_COMMAND"]=f"srun -n {tasks} pw.x < PREFIX.pwi > PREFIX.pwo"
            # read atoms
            try:
                test_atoms=read(os.path.join(qe_work_dir, "espresso.pwo"), format="espresso-out")
                energy=test_atoms.get_total_energy()
                pressure=test_atoms.get_stress()
                os.chdir(project_dir)
                print(f"{struct_dirname}, rs_index={rs_index} is done. skip.", flush=True)
                continue
            except:
                print(f"{struct_dirname}, rs_index={rs_index} is not done. compute.", flush=True)

            # start qe calc.
            qe_input_data = {
                'control': {
                    'calculation' : 'scf',
                    'pseudo_dir' : '/users/ccozza/pseudo',
                    'verbosity' : 'high',
                    'disk_io' : 'none',
                    'outdir' : './' #qe_work_dir
                    },
                'system': {
                    'ecutwfc': ecutwfc,
                    'ecutrho': ecutrho,
                    'occupations' : 'smearing',
                    'degauss': 2.0e-3,
                    'input_dft': xc_name,
                    },
                'electrons':{
                    'conv_thr': 1.0e-6,
                    'electron_maxstep': 200,
                    'diagonalization': 'david',
                    }
                }

            qe_calc = Espresso(pseudopotentials=pseudopotentials,
                            tstress=True, tprnfor=True,
                            input_data=qe_input_data,
                            kpts=kpts, koffset=koffset)

            atoms.calc=qe_calc
            atoms.get_potential_energy()

            os.chdir(project_dir)

        #break # for test