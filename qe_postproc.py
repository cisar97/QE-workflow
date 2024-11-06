#! /usr/bin/env python3

import numpy as np

def ionic_pressure(T, rho, nat, mass, K_boltz) :
    # Compute ionic pressure given T-rho
    # T = temperature in [K]
    # rho = density in [g/cm**3]
    # nat = Number of atoms 
    # mass = mass of the element in [g]
    # K_boltz = Boltzmann const in [J/K]
    alat = (((nat*mass) / rho)**(1/3)) / 100      # In [m]
    volume = alat**3
    p_ion = (K_boltz * nat * T) / volume / 1e9    # In [GPa]
    return np.round(p_ion, 6)


def reweight(energy_ref, energy_target, press_target, T, kb) :
    # Boltzmann Reweighting technique, rescale the target energy before reweight
    # energy_ref = total energy of the theory (e.g.PBE) from which the configuraions are extracted
    # energy_target = total energy of the parachuted configuration with new theory (e.g.QMC)
    # press_target = Quantity you want to reweight, from new thory 
    # T = temperature in [K] 
    # kb = Boltzmann constant in [eV/K] 
    # return : reweighted pressure with corresponding error

    energy_target = energy_target-(energy_target[0] - energy_ref[0])
    w = np.exp(-(energy_target - energy_ref)/(kb*T))
    wei = w / np.sum(w)
    p_rw = np.sum(press_target*wei)
    err_rw = np.sqrt( np.sum( wei * (press_target - np.mean(press_target))**2 ) / (((np.shape(w)[0] - 1)/(np.shape(w)[0]))*np.sum(wei)) )
    return p_rw, err_rw 


def binning(file, energy_col, press_col, iter_col, bin_len, init_step) :
    # Binning technique for averages and error bars
    # This function is a little bit silly... better to use scipy.stats.binned_statistic

    iterat = np.loadtxt(file, comments='#',skiprows=1,  unpack = True , usecols = (iter_col))
    totstep = np.shape(iterat)[0]
    step = totstep - init_step 

    if (step <= 0) :
        print('ERROR : init_step > totstep')

    nbin = int(step / bin_len) 
    if (step % bin_len) != 0 :
        nbin = nbin + 1 

    wei = np.zeros(nbin)
    energy_wei = np.zeros(nbin)
    press_wei = np.zeros(nbin)
    for i in range(nbin) :
        wei[i] = bin_len 
        energy_wei[i] = 0.0
        press_wei[i] = 0.0
    resto = step - (nbin-1)*bin_len 
    wei[-1] = resto 

    energy, press = np.loadtxt(file, comments='#', unpack = True, skiprows=init_step,  usecols = (energy_col, press_col))

    energy_sum = 0.0
    press_sum = 0.0
    j = 0   #count till bin_len 
    k = 0   #count nbin
    for i in range(0, step) :
        energy_sum += energy[i]
        press_sum += press[i]
        j += 1
        if (k<nbin-1) and (j==bin_len) :
            energy_wei[k] = energy_sum / wei[k]
            press_wei[k] = press_sum / wei[k] 
            j = 0 
            energy_sum = 0.0 
            press_sum = 0.0 
            k += 1
        elif (k==nbin-1) and (j==resto) :
            energy_wei[k] = energy_sum / wei[k]
            press_wei[k] = press_sum / wei[k]

    mean_en = 0.0 
    mean_en2 = 0.0 
    mean_pr = 0.0
    mean_pr2 = 0.0
    wei_tot = 0.0
    for i in range(nbin) :
        wei_tot += wei[i]
        mean_en += (energy_wei[i] * wei[i])
        mean_pr += (press_wei[i] * wei[i])
        mean_en2 += (energy_wei[i] * energy_wei[i] * wei[i])
        mean_pr2 += (press_wei[i] * press_wei[i] * wei[i])

    mean_en = mean_en / wei_tot
    mean_en2 = mean_en2 / wei_tot
    mean_pr = mean_pr / wei_tot
    mean_pr2 = mean_pr2 / wei_tot
    err_en = np.sqrt((mean_en2 - mean_en**2) / (nbin-1))
    err_pr = np.sqrt((mean_pr2 - mean_pr**2) / (nbin-1))
    return(mean_en, err_en, mean_pr, err_pr, nbin)


def rs_to_gcm3(rs):
    return np.round(2.6960431 / rs**3, 1)

def gcm3_to_rs(rho):
    return np.round((2.6960431 / rho)**(1/3), 3)


from ase.io import read
from ase.geometry.analysis import Analysis

def RDF(config, startFrame, stopFrame, rmax, nbins) :
    ## Provide the g(r) of a given MD 
    ## config: list of configurations (from ase read)
    ## startFrame,stopframe: MD steps to start and stop the g(r) 
    ## rmax: max radius at which compute g(r)
    ## nbins: number of interval in which the distance is divided

    RDF_tot = [0.0 for _ in range(nbins)]
    # Compute RDF
    for i in range(startFrame, stopFrame):
        analized = Analysis(configurations[i])

        if i == startFrame:
            RDF, distances = list(zip(*analized.get_rdf(rmax, nbins, return_dists=True)))
            RDF = RDF[0]
            distances = distances[0]
            RDF_tot += RDF
            continue

        RDF = analized.get_rdf(rmax, nbins)[0]
        RDF_tot += RDF

    # Normalize
    RDF_tot /= (stopFrame-startFrame)
    return RDF_tot, distances

##-----------------------------------------------------------------------
##              QE output reader
##-----------------------------------------------------------------------

from ase.units import Ry, GPa, Bohr
from ase import Atoms

def read_QE_output(calculation, filein, forces=False, positions=False, fermi_level=False, entropy=False, pos_to_bohr=True):
    ## Read the output of QE (md or scf) and save the desired quantities in a dictionary
    ## Default reads only electronic energy and pressure
    ## Default position output = Bohr

    results = {}
    if calculation == "md" :
        f = open(filein, "r")
        energy, pressure_e = [], []
        if positions: 
            x = []
            conf, conf_lines, idx = [], False, 0
            for line in f:
                if "number of atoms/cell" in line:
                    n = int(line.split()[-1])
                if "ATOMIC_POSITIONS" in line:
                    unit_pos = line.split()[-1]
                    break
        f.seek(0)
        if forces : 
            forces_data = [] 
            confF, confF_lines, idF = [], False, 0
            if positions != True :
                for line in f:
                    if "number of atoms/cell" in line:
                        n = int(line.split()[-1])
                        break
        f.seek(0)
        if fermi_level : 
            fermi_level_data = []
        if entropy : 
            entropy_data = [] 

        for line in f:
            if "!    total energy" in line:
                token = line.split()
                assert token[-1] == "Ry"
                energy.append( float(token[-2]) )
            if "total   stress" in line:
                token = line.split()
                assert token[-2] == "P="
                assert token[-3] == "(kbar)"
                pressure_e.append(float(token[-1]))
            if positions : 
                if "ATOMIC_POSITIONS" in line:
                    assert line.split()[-1] == unit_pos
                    conf_lines = True
                    continue
                if conf_lines:
                    conf.append([float(i) for i in line.split()[1:]])
                    idx += 1
                    if idx == n:
                        #print("conf:", np.array(conf))
                        x.append(np.array(conf))
                        conf, conf_lines, idx = [], False, 0
            if forces : 
                if "Forces acting on atoms" in line:
                    assert line.split()[-1] == "Ry/au):"
                    confF_lines = True
                    continue 
                if confF_lines :
                    confF.append([float(i) for i in line.split()[6:]])
                    idF += 1
                    if idF == n+1 :  ##There is blank line between "Forces..." and the values
                        del confF[0]  ##Delete the empty list from the blank line
                        #print("confF:", np.array(confF))
                        forces_data.append(np.array(confF))
                        confF, confF_lines, idF = [], False, 0
            if fermi_level : 
                if "the Fermi energy is" in line :
                    token = line.split()
                    assert token[-1] == "ev"
                    fermi_level_data.append(float(token[-2]))
            if entropy : 
                if "smearing contrib. (-TS)" in line: 
                    token = line.split()
                    assert token[-1] == "Ry"
                    entropy_data.append(float(token[-2]))
        f.close()
        energy, pressure_e = np.array(energy), np.array(pressure_e)
        results['energy[Ry]'] = energy
        results['pressure[GPa]'] = pressure_e/10 #To GPa
        if forces :
            forces_data = np.stack(forces_data)
            results['forces[Ry/bohr]'] = forces_data 
        if positions : 
            x = np.stack(x) 
            if unit_pos == "(bohr)":
                results['atomic_pos[bohr]'] = x
            elif unit_pos == "(angstrom)":
                results["atomic_pos[ang]"] = x
        if fermi_level :
            fermi_level_data = np.array(fermi_level_data) 
            results['fermi_en[eV]'] = fermi_level_data
        if entropy : 
            entropy_data = np.array(entropy_data) 
            results['entropy[Ry]'] = entropy_data 
        
    if calculation == "scf" : 
        atom = read(filein) 
        energy = atom.get_total_energy()
        pressure_e = atom.get_stress()
        results['energy[Ry]'] = energy/Ry
        results['pressure[GPa]'] = -np.mean(pressure_e[0:3])/GPa #Don't understand why ase pressure is negative
        if positions :
            positions_data = atom.get_positions()
            if pos_to_bohr :
                results['atomic_pos[bohr]'] = positions_data/Bohr
            else :
                results['atomic_pos[ang]'] = positions_data
        if forces : 
            forces_data = atom.get_forces()
            results['forces[Ry/bohr]'] = forces_data/(Ry/Bohr)
        if fermi_level or entropy : 
            f = open(filein, 'r')
            for line in f: 
                if fermi_level : 
                    if "the Fermi energy is" in line: 
                        token = line.split()
                        assert token[-1] == "ev"
                        fermi_level_data = token[-2]
                        results['fermi_en[eV]'] = fermi_level_data 
                if entropy : 
                    if "smearing contrib. (-TS)" in line : 
                        token = line.split()
                        assert token[-1] == "Ry"
                        entropy_data = token[-2]
                        results['entropy[Ry]'] = entropy_data

    return results
