#! /usr/bin/env python3 
# -*- coding: utf-8 -*-

from qe_input_generator import get_atoms_from_file, build_qe_input
from qe_postproc import gcm3_to_rs
import os, glob, re, shutil, sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from uncertainties import ufloat
from ase.io import read, write 
from ase import Atoms
from ase.units import GPa, Hartree

# unit conversion
amu_to_g = 1.660539066 * 10**(-27)
ang_to_cm = 10**(-8)
Ha_to_J=4.35974*10**-18
angstrom_to_m=1.0*10**-10
ang_to_bohr = 0.5291772109 # used in turbo.
atomic_mass_to_kg=1.66057*10**-27
hz_to_cm_1 = 33.35641*10**-12
Ha_bohr3_to_GPa=29421.02648438959
bohr_to_angstrom=0.529177210903
angstrom_to_bohr=1.0/bohr_to_angstrom


def write_dataset_atom(project_dir, MDfile, T, rho, config, cell, formula='H128', pbc=True, qe_input_dir_name="dataset") : 
    ## Inside the home directory provided as input create a /dataset directory 
    ## containing the atomic configurations extracted from the MD trajectory
    ## provided as input. The atomic configurations are stored as .xsf files
    # project_dir : path to the home directory
    # MDfile : path to the MD trajectory file
    # config : LIST of configurations to extract from the MD trajectory
    # cell : the cell you specify in ase. A list of floats.
    os.makedirs(f"{project_dir}/{qe_input_dir_name}", exist_ok=True)
    rs = gcm3_to_rs(rho)
    rs_label = f"{rs:.3f}"
    positions = get_atoms_from_file(MDfile, config)
    for i, pos in enumerate(positions) :
        pos *= ang_to_bohr
        atom = Atoms(formula, positions=pos, cell=cell, pbc=pbc)
        write(f"{project_dir}/dataset/input_T{T}_d{rs_label}_{config[i]}.xsf", atom)

def generate_distorted_configurations(project_dir, qe_input_dir_name="dataset", struct_dir_name="struct_input_dir" ,rs_pm_list=[-0.05, -0.02, -0.01, 0.00, +0.01, +0.02, +0.05], indshift=0) :
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
    struct_dir=os.path.join(project_dir, struct_dir_name)
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


from ase.calculators.espresso import Espresso, EspressoProfile

def compute_qe_distorted_conf(project_dir, ase_espresso_calculator, xc_name, pp_prefix, ecutwfc, struct_dir_name="struct_input_dir", qe_input_dir_name="dataset", scratch_dir=None ):  
    ## From the distorted structures in struct_dir_name create the espresso input file and compute.
    ## project_dir: Working directory 
    ## ase_espresso_calculator: QuantumESPRESSO calculator object in ASE
    ## xc_name: Exc functional name string to be used in the folder name  
    ## pp_prefix: Pseudopotential name string to be used in the folder name 
    ## ecutwfc: Wavefunction cutoff to be used in the filder name. The cutoff for the calcutor needs to be specified outside
    ## scratch_dir: SCRATCH directory for QE in case the QE flag 'dik_io' is NOT set to 'none'.  

    os.chdir(project_dir)
    if scratch_dir != None : os.makedirs(scratch_dir, exist_ok=True)
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
                atoms.calc=ase_espresso_calculator
            
            atoms.get_potential_energy()
            os.chdir(project_dir)


# define functions for fitting.
def fit_function(x,y,xs,x_deriv_target,order=3):
    order_fit=order
    val_pd = pd.DataFrame(index=[], columns=x)
    val_pd_deriv = pd.DataFrame(index=[], columns=x)
    val_pd_plot = pd.DataFrame(index=[], columns=xs)
    val_pd_deriv_plot = pd.DataFrame(index=[], columns=xs)
    x_min_list=[]; y_min_list=[]; sigma_list = [];
    y_deriv_target_list=[]
    w = np.polyfit(x,y,order_fit)
    ys = np.polyval(w,x)
    ys_plot = np.polyval(w,xs)
    c = np.poly1d(w).deriv(1)
    ys_deriv = c(x)
    ys_deriv_plot = c(xs)
    ys_deriv_target = c(x_deriv_target)

    return ys_plot, ys_deriv_plot, ys_deriv_target


def analyze_distorted_configs(project_dir, xc_name, pp_prefix, ecutwfc) : 
    ## Analyze the results from the strained configurations 
    ## project_dir : Working directory
    ## xc_name: Exc functional name string to be used in the folder name  
    ## pp_prefix: Pseudopotential name string to be used in the folder name 
    ## ecutwfc: Wavefunction cutoff to be used in the filder name. The cutoff for the calcutor needs to be specified outside
    ## single_plot: Plot results
    
    pressure_dict = {} 
    #  CHECK QE COMPUTATION HAVE BEEN PERFORMED
    # loop for all config
    qe_results_dir=os.path.join(project_dir, f"qe_results_xc_{xc_name}_pp_{pp_prefix}_ecutw_{ecutwfc}_dir")
    qe_config_dirname_list=[os.path.basename(d) for d in glob.glob(os.path.join(qe_results_dir, '*')) if os.path.isdir(d)]
    qe_config_dirname_list=sorted(qe_config_dirname_list)
    os.chdir(qe_results_dir)

    # check if jobs are completed
    qe_config_dirname_list_=[]
    for qe_config_dirname in qe_config_dirname_list:
        qe_config_dir=os.path.join(qe_results_dir, qe_config_dirname)
        index_ref, T_ref, rs_label_ref, steps_ref = re.search(r'(\d+)_T(\d+)_d(\d+(?:\.\d+)?)_(\d+)', qe_config_dir).groups()
        rs_dirname_list=[os.path.basename(d) for d in glob.glob(os.path.join(qe_results_dir, qe_config_dirname, 'rs_index_*')) if os.path.isdir(d)]
        if len(rs_dirname_list) != 7:
            print('rs_dirname_list different from 7')
            continue

        try:
            for rs_dirname in rs_dirname_list:
                rs_dir=os.path.join(qe_results_dir, qe_config_dirname, rs_dirname)
                atoms=read(os.path.join(rs_dir, "espresso.pwo"), format="espresso-out")
                pressure=np.mean(atoms.get_stress()[0:3]) / GPa
            print(f"{qe_config_dirname}...ok")
        except:
            print(f"{qe_config_dirname}...ng")
            continue

        qe_config_dirname_list_.append(qe_config_dirname)

    qe_config_dirname_list = qe_config_dirname_list_
    # ANALYSIS
    for qe_config_dirname in qe_config_dirname_list:
        print(f"{qe_config_dirname} in progress...")
        qe_config_dir=os.path.join(qe_results_dir, qe_config_dirname)
        index_ref, T_ref, rs_label_ref, steps_ref = re.search(r'(\d+)_T(\d+)_d(\d+(?:\.\d+)?)_(\d+)', qe_config_dir).groups()
        rs_dirname_list=[os.path.basename(d) for d in glob.glob(os.path.join(qe_results_dir, qe_config_dirname, 'rs_index_*')) if os.path.isdir(d)]

        # read data
        all_atoms_d={}
        all_energy_qe_d={}
        all_pressure_qe_d={}
        rs_list=[]
        rs_label_list=[]
        os.chdir(qe_config_dir)

        for rs_dirname in rs_dirname_list:
            rs_dir=os.path.join(qe_results_dir, qe_config_dirname, rs_dirname)
            # read atoms
            atoms=read(os.path.join(rs_dir, "espresso.pwo"), format="espresso-out")
            num_ele = np.sum(atoms.get_atomic_numbers())
            vol = atoms.get_volume() / ang_to_bohr**3
            rs = (3.0 * vol / (num_ele * 4 * np.pi))**(1.0/3.0)
            rs_label = f"{rs:.3f}"
            rs_list.append(rs)
            rs_label_list.append(rs_label)
            all_atoms_d[rs_label]=atoms
            # energy
            energy=atoms.get_total_energy() / Hartree # Hartree
            all_energy_qe_d[rs_label]=energy
            # pressure
            pressure=np.mean(atoms.get_stress()[0:3]) / GPa # -> pressure (GPa)
            all_pressure_qe_d[rs_label]=pressure
        
        # for single plot
        fig = plt.figure(figsize=(6, 6), facecolor='white', dpi=300)
        ax1 = fig.add_subplot(1,1,1)
        interpolate_num=100
        ax1.set_title(f"H$_{{128}}$, EOS, Plane-Wave(QE), cutoff={ecutwfc} Ry, XC={xc_name}, PP={pp_prefix}", fontsize=10)
        ax1.set_ylabel("Pressure (GPa)")
        ax1.set_xlabel("Volume (bohr$^3$)")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Relative energy (Ha)")
        plt.rcParams['font.family'] ='sans-serif'
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.major.width'] = 1.0
        plt.rcParams['ytick.major.width'] = 1.0
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.linewidth'] = 1.5

        # plot DFT energy and forces
        ref_rs_label=rs_label_ref
        ref_rs=rs_list[rs_label_list.index(rs_label_ref)]
        energy_list=[all_energy_qe_d[rs_label] for rs_label in rs_label_list]
        #Energy plot (ax1)
        #ax1.set_ylim(ymin, ymax)
        target_volume=all_atoms_d[ref_rs_label].get_volume()/bohr_to_angstrom**3
        target_pressure=all_pressure_qe_d[ref_rs_label]
        volume_list = [all_atoms_d[rs_label].get_volume()/bohr_to_angstrom**3 for rs_label in rs_label_list]
        x=volume_list; y=energy_list;
        xs = np.linspace(np.min(x),np.max(x),interpolate_num)
        ys_plot, ys_deriv_plot, ys_deriv_target = fit_function(x,y,xs,x_deriv_target=target_volume)
        # calc energy at zer
        energy_at_zero=np.min(ys_plot)
        # plot PES
        ax2.plot(x, y-energy_at_zero, marker="o", linestyle='', color='g', label=f"Energy (DFT-{xc_name})")
        ax2.plot(xs, ys_plot-energy_at_zero, marker="", linestyle='--', color='g', label="")
        # plot Pressure
        ax1.plot(xs, -(ys_deriv_plot)*Ha_bohr3_to_GPa, color="r", linestyle='--', label=f"Deriv. of Energy (DFT-{xc_name})")
        ax1.plot(target_volume, -target_pressure, marker="o", linestyle='', color='r', label=f"Pressure (DFT-{xc_name})")
        #print(f"VMC Pressure (P) at rs={ref_rs_label} = {-target_pressure} (GPa)")
        num_electron=len(all_atoms_d[ref_rs_label])
        def forward(V):
            rs = (4*np.pi*num_electron/V/3.0)**(-1.0/3.0)
            return rs
        def inverse(rs):
            V = num_electron * 4*np.pi * rs**3.0 / 3.0
            return V
        secax = ax1.secondary_xaxis('top', functions=(forward, inverse))
        secax.set_xlabel('Wigner?~@~SSeitz radius, $r_s$')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper right')
        plt.savefig(f"dft_{xc_name}_pressure_pp_{pp_prefix}.png")
        plt.cla()
        plt.clf()
        plt.close()
    
        pressure_dict[index_ref]={'T':T_ref, 'rs_label': rs_label_ref, 'md_steps': steps_ref, 'rs_actual':ref_rs,
                               'numerical_pressure(GPa)':-(ys_deriv_target)*Ha_bohr3_to_GPa,
                               'analytical_pressure(GPa)':-target_pressure
                              }

    # save data as a pickled pandas DataFrame
    index_list=pressure_dict.keys()
    col=['T', 'rs_label', 'rs_actual', 'md_steps', 'numerical_pressure(GPa)', 'analytical_pressure(GPa)']
    pressure_pd=pd.DataFrame(np.array([[pressure_dict[index][key] for index in index_list] for key in col]).T, index=index_list, columns=col)
    pressure_pd.to_pickle(os.path.join(project_dir, f'qe_{xc_name}_pressures_pp_{pp_prefix}_ecutw_{ecutwfc}.pkl'))
    print(pressure_pd)

