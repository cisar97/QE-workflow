#! /usr/bin/env python3

import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.integrate import quad

# free energy calculation: thermodynamic integration ###########################

def F_isotherm(T, rhos, Ps, method="log_interp_all"):
    """ Compute an isotherm of (relative) free energies by thermodynamic integration:
            ∂(F/T)/∂ρ = p/(Tρ^2),
        where F is the specific free energy (divided by mass).

    Units: T (K), rhos (g/cm^3), Ps (GPa), Fs (MJ/kg/K).
    """
    if method == "direct_interp_all":
        spl = CubicSpline(rhos, Ps/rhos**2/T)
        Fs = [0.]
        for rho0, rho1 in zip(rhos[:-1], rhos[1:]):
            F, _ = quad(spl, rho0, rho1)
            Fs.append(F)
        Fs = np.array(Fs).cumsum()
    elif method == "log_interp_all":
        spl = CubicSpline(np.log(rhos), Ps/rhos/T)
        Fs = [0.]
        for rho0, rho1 in zip(rhos[:-1], rhos[1:]):
            F, _ = quad(spl, np.log(rho0), np.log(rho1))
            Fs.append(F)
        Fs = np.array(Fs).cumsum()
    return Fs

def F_isochore(Ts, rho, Es, method="log"):
    """ Compute an isochore of (relative) free energies by thermodynamic integration:
            ∂(F/T)/∂T = -E/T^2,
        where F, E are the specific (free) energy (divided by mass), respectively.

    Units: Ts (K), rho (g/cm^3), Es (MJ/kg), Fs (MJ/kg/K).
    """
    if method == "direct_interp_all":
        spl = CubicSpline(Ts, -Es/Ts**2)
        Fs = [0.]
        for T0, T1 in zip(Ts[:-1], Ts[1:]):
            F, _ = quad(spl, T0, T1)
            Fs.append(F)
        Fs = np.array(Fs).cumsum()
    elif method == "log_interp_all":
        spl = CubicSpline(np.log(Ts), -Es/Ts)
        Fs = [0.]
        for T0, T1 in zip(Ts[:-1], Ts[1:]):
            F, _ = quad(spl, np.log(T0), np.log(T1))
            Fs.append(F)
        Fs = np.array(Fs).cumsum()
    elif method == "log":
        spl = CubicSpline(Ts, Es)
        Fs = [0.]
        for T0, T1 in zip(Ts[:-1], Ts[1:]):
            F, _ = quad(lambda logT: -spl(np.exp(logT)) / np.exp(logT), np.log(T0), np.log(T1))
            Fs.append(F)
        Fs = np.array(Fs).cumsum()
    return Fs

def F_table(Ts, rhos, Ps, Es,
    method_isotherm="log_interp_all", method_isochore="log",
    ref=None, anchor=None, S_anchor=0.,
):
    """ Given a 2D grid of pressure and energy data, compute the free energy
    (divided by temperature) and entropy by thermodynamic integration along either
    isotherms or isochores.
    Args:
        ref: 2-tuple (T, rho) specifying the reference point to perform thermodynamic
            integration, which must be one element of the grid spanned by Ts and rhos.
            Defaults to None, which means the "bottom-left corner" (Ts[0], rhos[0]).
        anchor: 2-tuple (T, rho) specifying the anchor point of free energy and entropy.
            The ABSOLUTE entropy at the anchor point is set to the value of the input
            argument `S_anchor`. Defaults to None, which means the anchor point is the same
            as the REFERENCE point for TI.
    Returns:
        F_table_isotherm: (num_Ts, num_rhos), TI first along isochore, then isotherm ("⌜").
        F_table_isochore: (num_Ts, num_rhos), TI first along isotherm, then isochore ("⌟").
        S_table_isotherm, S_table_isochore: (num_Ts, num_rhos), corresponding entropies.
    """
    F_isotherms, F_isochores = [], []
    for i, T in enumerate(Ts):
        Fs = F_isotherm(T, rhos, Ps[i], method=method_isotherm)
        F_isotherms.append(Fs)
    for j, rho in enumerate(rhos):
        Fs = F_isochore(Ts, rho, Es[:, j], method=method_isochore)
        F_isochores.append(Fs)
    F_isotherms, F_isochores = np.array(F_isotherms), np.array(F_isochores).T

    if ref is None: ref = Ts[0], rhos[0]
    T_idx, rho_idx = np.nonzero(Ts == ref[0])[0][0], np.nonzero(rhos == ref[1])[0][0]
    F_isotherms -= F_isotherms[:, rho_idx, None]
    F_isochores -= F_isochores[T_idx]
    F_table_isotherm = F_isotherms + F_isochores[:, rho_idx, None]
    F_table_isochore = F_isochores + F_isotherms[T_idx]
    assert F_table_isotherm[T_idx, rho_idx] == 0. and F_table_isochore[T_idx, rho_idx] == 0.
    S_table_isotherm = Es/Ts[:, None] - (Es/Ts[:, None])[T_idx, rho_idx] - F_table_isotherm
    S_table_isochore = Es/Ts[:, None] - (Es/Ts[:, None])[T_idx, rho_idx] - F_table_isochore
    assert S_table_isotherm[T_idx, rho_idx] == 0. and S_table_isochore[T_idx, rho_idx] == 0.

    if anchor is None: anchor = ref
    T_idx, rho_idx = np.nonzero(Ts == anchor[0])[0][0], np.nonzero(rhos == anchor[1])[0][0]
    S_table_isotherm += (S_anchor - S_table_isotherm[T_idx, rho_idx])
    S_table_isochore += (S_anchor - S_table_isochore[T_idx, rho_idx])
    F_table_isotherm = Es/Ts[:, None] - S_table_isotherm
    F_table_isochore = Es/Ts[:, None] - S_table_isochore

    return F_table_isotherm, F_table_isochore, S_table_isotherm, S_table_isochore

def F_flux(F_isotherms, F_isochores):
    """ Given free energy isotherms and isochores, compute the "flux" of each
    unit square as a line integral over the closed loop. Note a zero flux
    corresponds to perfect thermodynamic consistency of the data.
        Note that for convenience, the two input arguments should be anchored
    at the index (0, 0), corresponding to the first element of Ts and rhos,
    respectively.
    Shapes:
        F_isotherms, F_isochores: (num_Ts, num_rhos)
        Returns: (num_Ts - 1, num_rhos - 1)
    """
    isotherms_diff = F_isotherms[:, 1:] - F_isotherms[:, :-1]
    isochores_diff = F_isochores[1:] - F_isochores[:-1]
    flux = isotherms_diff[:-1] - isotherms_diff[1:] \
         + isochores_diff[:, 1:] - isochores_diff[:, :-1]
    return flux

def make_F_interpolation(Ts, rhos, Ps, Es, Fs, dPs_rhoT_drho=None, dEs_T_drho=None):
    """ Make an interpolation function of the free energy (divided by temperature)
    over a 2D grid. The resulting partial derivatives ∂(F/T)/∂ρ and ∂(F/T)/∂T can
    largely reproduce the input pressure and energy data Ps and Es, respectively.

    Note that we have ∂(F/T)/∂lnρ = p/ρT, ∂(F/T)/∂lnT = -E/T.
    """
    Ps_rhoT_0, Ps_rhoT_1 = (Ps/rhos/Ts[:, None])[:, 0], (Ps/rhos/Ts[:, None])[:, -1]
    if dPs_rhoT_drho is None:
        spl_isotherms = CubicSpline(np.log(rhos), Fs, axis=1,
            bc_type=((1, Ps_rhoT_0), (1, Ps_rhoT_1)),
        )
    else:
        spl_isotherms = make_interp_spline(np.log(rhos), Fs, axis=1, k=5,
            bc_type=([(1, Ps_rhoT_0), (2, dPs_rhoT_drho[0])],
                     [(1, Ps_rhoT_1), (2, dPs_rhoT_drho[1])]),
        )
    Es_T_0, Es_T_1 = (Es/Ts[:, None])[0], (Es/Ts[:, None])[-1]
    if dEs_T_drho is None:
        spl_Es_T_0 = CubicSpline(np.log(rhos), Es_T_0)
        spl_Es_T_1 = CubicSpline(np.log(rhos), Es_T_1)
    else:
        spl_Es_T_0 = CubicSpline(np.log(rhos), Es_T_0, bc_type=((1, dEs_T_drho[0][0]), (1, dEs_T_drho[1][0])))
        spl_Es_T_1 = CubicSpline(np.log(rhos), Es_T_1, bc_type=((1, dEs_T_drho[0][-1]), (1, dEs_T_drho[1][-1])))

    def F_interpolation(_Ts, _rhos):
        _Fs_isochore = spl_isotherms(np.log(_rhos))
        _dFs_drho_isochore = spl_isotherms(np.log(_rhos), 1)
        spl_Fs_isochore = CubicSpline(np.log(Ts), _Fs_isochore, axis=0,
            bc_type=((1, -spl_Es_T_0(np.log(_rhos))), (1, -spl_Es_T_1(np.log(_rhos)))),
        )
        spl_dFs_drho_isochore = CubicSpline(np.log(Ts), _dFs_drho_isochore, axis=0)
        _Fs = spl_Fs_isochore(np.log(_Ts))
        _dFs_dT = spl_Fs_isochore(np.log(_Ts), 1)
        _dFs_drho = spl_dFs_drho_isochore(np.log(_Ts))
        return _Fs, _dFs_drho, _dFs_dT

    return F_interpolation

from scipy.interpolate import RegularGridInterpolator

def read_Chabrier_Trho(Ts, rhos):

    filename = "/Users/cesarecozza/dataset_H/EOS/DFT_EOS/Literature/Chabrier_ApJ_2019/TABLE_H_Trho_v1"
    # T: K, rho: g/cm^3, P: GPa, E: MJ/kg, S: MJ/kg/K.
    logT, logP, logrho, logE, logS = np.loadtxt(filename, comments="#", usecols=(0, 1, 2, 3, 4), unpack=True)
    num_Ts, num_rhos = 121, 281
    logT = logT.reshape(num_Ts, num_rhos)[:, 0]
    logrho = logrho.reshape(num_Ts, num_rhos)[0]
    logP, logE, logS = logP.reshape(num_Ts, num_rhos), logE.reshape(num_Ts, num_rhos), logS.reshape(num_Ts, num_rhos)
    #print("logT:", logT.shape, "logrho:", logrho.shape, "logP:", logP.shape, "logE:", logE.shape, "logS:", logS.shape)

    rgi = RegularGridInterpolator(
        points=(logT, logrho), values=np.stack([logP, logE, logS], axis=-1),
        method="cubic", bounds_error=True, fill_value=np.nan,
    )

    logTs, logrhos = np.meshgrid(np.log10(Ts), np.log10(rhos), indexing="ij")
    results = 10**rgi((logTs, logrhos))
    Ps, Es, Ss = results[..., 0], results[..., 1], results[..., 2]
    return Ps, Es, Ss

def read_Chabrier_TP(Ts, Ps):

    filename = "/Users/cesarecozza/dataset_H/EOS/DFT_EOS/Literature/Chabrier_ApJ_2019/TABLE_H_TP_v1"
    # T: K, rho: g/cm^3, P: GPa, E: MJ/kg, S: MJ/kg/K.
    logT, logP, logrho, logE, logS = np.loadtxt(filename, comments="#", usecols=(0, 1, 2, 3, 4), unpack=True)
    num_Ts, num_Ps = 121, 441
    logT = logT.reshape(num_Ts, num_Ps)[:, 0]
    logP = logP.reshape(num_Ts, num_Ps)[0,:]
    logrho, logE, logS = logrho.reshape(num_Ts, num_Ps), logE.reshape(num_Ts, num_Ps), logS.reshape(num_Ts, num_Ps)
    #print("logT:", logT.shape, "logrho:", logrho.shape, "logP:", logP.shape, "logE:", logE.shape, "logS:", logS.shape)

    rgi = RegularGridInterpolator(
        points=(logT, logP), values=np.stack([logrho, logE, logS], axis=-1),
        method="cubic", bounds_error=True, fill_value=np.nan,
    )

    logTs, logPs = np.meshgrid(np.log10(Ts), np.log10(Ps), indexing="ij")
    results = 10**rgi((logTs, logPs))
    rhos, Es, Ss = results[..., 0], results[..., 1], results[..., 2]
    return rhos, Es, Ss

def plot_heatmap(fig, ax, data, title, cbar_label, xlabel, ylabel, xticks, yticks, colormap='bwr', value_min=None, value_max=None, text=False, ticks_digit=None): 
    if value_max is None:
        im = ax.imshow(data, origin='lower', aspect='auto', cmap=colormap)
        cbar = fig.colorbar(im, ax=ax)
    else : 
        im = ax.imshow(data, origin='lower', aspect='auto', cmap=colormap, vmax=value_max, vmin=value_min)
        cbar = fig.colorbar(im, ax=ax, ticks=np.arange(value_min, value_max+((value_max+np.abs(value_min))/6), (value_max+np.abs(value_min))/6))
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(np.arange(len(xticks)), labels=xticks)
    ax.set_yticks(np.arange(len(yticks)), labels=yticks)
    plt.setp(ax.get_xticklabels(), fontsize=3.5)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    if text:
        for i in range(len(yticks)):
            for j in range(len(xticks)):
                value = data[i,j]
                text = ax.text(j, i, f'{value:.2f}',
                                ha="center", va="center", color="k", fontsize=3)       
    if ticks_digit is not None: 
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{ticks_digit}f'))
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

def tot_energy_per_atom(qe_energy, T, deg_of_freedom, N_atom, k_B=1.38064852e-23, H_mass=1.6735575e-24, ry_to_kJ=2.179874099e-21):  
    ## Convert the internal energy from QE to energy per atom in kJ/g
    ## Add ionic contribution to the internal energy
    ion_kin_energy = deg_of_freedom * k_B * T / 1000 ## kJ
    electron_energy = qe_energy / N_atom * ry_to_kJ ## kJ
    tot_energy = ion_kin_energy + electron_energy  ## kJ
    tot_energy /= H_mass ## kJ/g
    return tot_energy

def get_celldm(rho, nat, mass=1.6735575e-24, cm_to_au=188972598.85789) : 
    # Compute dimension of the cell from density 
    # rho = density in [g/cm^3] 
    # nat = number of atoms in the cell 
    # mass = mass in [g] of the element
    # cm_to_au = measure coonversion from cm to bohr 
    # returns length of the cubic cell in [Bohr] 
    celldm = (((nat*mass) / rho)**(1/3)) * cm_to_au
    return np.round(celldm, 6)  

def total_pressure(qe_pressure, temps, rhos, N_atom, k_B=1.38064852e-23, bohr_to_m = 5.2917725e-11) : 
    ## Compute the total pressure from the QE pressure and the ideal gas pressure
    ## qe_pressure = pressure from QE in GPa, numpy array 
    ## temps = temperatures in K, numpy array
    ## rhos = densities in g/cm^3, numpy array
    ## N_atom = number of atoms in the cell
    ## returns total pressure in GPa, numpy array 
    if np.shape(qe_pressure)[0] != np.shape(temps)[0]*np.shape(rhos)[0] :
        raise ValueError("qe_pressure dimension do not correspond with temps and rhos")
    p_ion = np.empty(0)
    for T in temps : 
        for rho in rhos : 
            celldm = get_celldm(rho, N_atom)
            p = (N_atom * k_B * T) / (1e9*(celldm*bohr_to_m)**3) ##in GPa
            p_ion = np.append(p_ion, p) 
    return qe_pressure+p_ion

def total_energy(qe_energy, temps, rhos, N_atom, deg_of_freedom=3/2, k_B=1.38064852e-23, H_mass=1.6735575e-24, ry_to_kJ=2.179874099e-21) :
    ## Convert the internal energy from QE to total energy in kJ/g
    ## Add ionic contribution to the internal energy
    ## qe_energy = internal energy from QE in Ry, numpy array
    ## temps = temperatures in K, numpy array
    ## rhos = densities in g/cm^3, numpy array
    ## N_atom = number of atoms in the cell
    ## deg_of_freedom = degrees of freedom of the system
    ## k_B = Boltzmann constant in J/K
    ## H_mass = mass of the hydrogen atom in g
    ## ry_to_kJ = conversion factor from Ry to kJ
    ## returns total energy in kJ/g, numpy array
    if np.shape(qe_energy)[0] != np.shape(temps)[0]*np.shape(rhos)[0] :
        raise ValueError("qe_energy dimension do not correspond with temps and rhos")
    electron_energy = qe_energy / N_atom * ry_to_kJ ## kJ
    tot_en = np.empty(0)
    for i,T in enumerate(temps) : 
        ion_kin_en = deg_of_freedom * k_B * T / 1000 ## kJ 
        te = electron_energy[i*np.shape(rhos)[0]:(i+1)*np.shape(rhos)[0]] + ion_kin_en
        tot_en = np.append(tot_en, te/H_mass)
    return tot_en

def optimize_Eshift(rho0, T0, T1, REOS_data, dft_data, T_start_index_dft, T_end_index_dft, rho_start_index_dft, rho_end_index_dft) : 
    temp = dft_data['Temp'].unique().astype(float)
    dens = dft_data['rho'].unique() 
    Ts_REOS = REOS_data['Temperature'].unique().astype(float)
    rhos_REOS = REOS_data['Density'].unique().astype(float)
    shift_extremes = []
    for T_0 in [T0, T1] :
        REOS_zeroEn = REOS_data.loc[(REOS_data['Temperature'] == T_0) & (REOS_data['Density'] == rho0)]['Spec Internal Energy'].astype(float).values[0]
        dft_zeroEn = dft_data.loc[(dft_data['Temp'] == T_0) & (dft_data['rho'] == rho0)]['En[Ry]'].astype(float).values[0]
        #print(f"REOS_zeroEn = {REOS_zeroEn}, dft_zeroEn = {dft_zeroEn}")
        shift = REOS_zeroEn - dft_zeroEn
        #print(f"---------------- T_0 = {T_0} ----------------")
        #print(f"SHIFT = {shift}")
        shift_extremes.append(shift)
    ## Create the table of all EOS with different shifts
    all_shift = np.linspace(shift_extremes[0]+5, shift_extremes[1]-5, 40)
    #print("All shifts : ",all_shift)
    all_eos_table = []
    for shift in all_shift :
        dft_data_shifted= dft_data.copy() 
        dft_data_shifted['En[Ry]'] += shift 
        #print(eos_dft)
        dft_eos = REOS_data.copy()
        for T in temp : 
            for rho in dens :
                dft_eos = dft_eos.replace(dft_eos.loc[(dft_eos['Temperature'] == T) & (dft_eos['Density'] == rho)]['Pressure'].values, 
                                        dft_data_shifted.loc[(dft_data_shifted['Temp'] == T) & (dft_data_shifted['rho'] == rho)]['Pr[GPa]'].values)

                dft_eos = dft_eos.replace(dft_eos.loc[(dft_eos['Temperature'] == T) & (dft_eos['Density'] == rho)]['Spec Internal Energy'].values, 
                                        dft_data_shifted.loc[(dft_data_shifted['Temp'] == T) & (dft_data_shifted['rho'] == rho)]['En[Ry]'].values)
        all_eos_table.append(dft_eos)
    ## Compute the fluxes at the border with REOS, both left, right and sum
    Fluxes_sums = []
    for eos_tab in all_eos_table : 
        Ps, Es = eos_tab["Pressure"].to_numpy().reshape(len(Ts_REOS), len(rhos_REOS)) , eos_tab["Spec Internal Energy"].to_numpy().reshape(len(Ts_REOS), len(rhos_REOS))
        flux = F_flux(Ts_REOS, rhos_REOS, Ps, Es)
        flux_val = np.sum(np.abs(flux[T_start_index_dft:T_end_index_dft+1, rho_start_index_dft:rho_end_index_dft+1]))
        #print("Flux:", flux_val)
        Fluxes_sums.append(flux_val)

    return all_eos_table[np.argmin(Fluxes_sums)], all_shift[np.argmin(Fluxes_sums)]


###################################################################################
## Read SCvH EOS data

def read_SCvH_table():
    filename = "/Users/cesarecozza/dataset_H/database/scvh_1995/H_TAB_I.A"
    filename = os.path.join(os.path.dirname(__file__), filename)
    f = open(filename, "r")
    logT = []
    logP, logrho, logS, logE = [], [], [], []
    isotherm_size = 0
    for line in f:
        token = line.split()
        if len(token) == 2:
            logT.append(float(token[0]))
            assert isotherm_size == 0
            isotherm_size = int(token[1])
            data = []
        else:
            data.append([float(t) for t in token])
            isotherm_size -= 1
            if isotherm_size == 0:
                data = np.array(data)
                _logP, _, _, _logrho, _logS, _logE, _, _, _, _, _ = data.T
                _logP -= 10  # unit: GPa
                _logS -= 10  # unit: MJ/kg/K
                _logE -= 10  # unit: MJ/kg
                assert np.allclose(_logP, -6. + 0.2*np.arange(_logP.size))
                logP.append(_logP)
                logrho.append(_logrho)
                logS.append(_logS)
                logE.append(_logE)
    logT = np.array(logT)
    return logT, logrho, logP, logE, logS

def read_SCvH_lowdensity(Ts, rhos):
    logT, logrho, logP, logE, logS = read_SCvH_table()

    logPs, logEs, logSs = [], [], []
    for _logT, _logP, _logrho, _logE, _logS in zip(logT, logP, logrho, logE, logS):
        """
        print("logT:", _logT, "T:", 10**_logT, "logrho.shape:", _logrho.shape,
              "rho min:", 10**_logrho[0], "rho max:", 10**_logrho[-1])
        """
        assert np.all(_logrho[:-1] < _logrho[1:])
        spl = CubicSpline(_logrho, np.vstack([_logP, _logE, _logS]), axis=1)
        _logP, _logE, _logS = spl(np.log10(rhos))
        logPs.append(_logP)
        logEs.append(_logE)
        logSs.append(_logS)
    logPs, logEs, logSs = np.array(logPs), np.array(logEs), np.array(logSs)

    if isinstance(Ts, tuple):
        assert len(Ts) == 2
        T_min, T_max = Ts
        T_idx = np.logical_and(10**logT >= T_min, 10**logT <= T_max)
        Ts = 10**logT[T_idx]
        logPs, logEs, logSs = logPs[T_idx], logEs[T_idx], logSs[T_idx]
    else:
        rgi = RegularGridInterpolator(
            points=(logT, np.log10(rhos)), values=np.stack([logPs, logEs, logSs], axis=-1),
            method="cubic", bounds_error=True, fill_value=np.nan,
        )

        logTs, logrhos = np.meshgrid(np.log10(Ts), np.log10(rhos), indexing="ij")
        results = rgi((logTs, logrhos))
        logPs, logEs, logSs = results[..., 0], results[..., 1], results[..., 2]
    Ps, Es, Ss = 10**logPs, 10**logEs, 10**logSs

    return Ts, rhos, Ps, Es, Ss


def make_F_interpolation(Ts, rhos, Ps, Es, Fs, dPs_rhoT_drho=None, dEs_T_drho=None):
    """ Make an interpolation function of the free energy (divided by temperature)
    over a 2D grid. The resulting partial derivatives ∂(F/T)/∂ρ and ∂(F/T)/∂T can
    largely reproduce the input pressure and energy data Ps and Es, respectively.

    Note that we have ∂(F/T)/∂lnρ = p/ρT, ∂(F/T)/∂lnT = -E/T.
    """
    Ps_rhoT_0, Ps_rhoT_1 = (Ps/rhos/Ts[:, None])[:, 0], (Ps/rhos/Ts[:, None])[:, -1]
    if dPs_rhoT_drho is None:
        spl_isotherms = CubicSpline(np.log(rhos), Fs, axis=1,
            bc_type=((1, Ps_rhoT_0), (1, Ps_rhoT_1)),
        )
    else:
        spl_isotherms = make_interp_spline(np.log(rhos), Fs, axis=1, k=5,
            bc_type=([(1, Ps_rhoT_0), (2, dPs_rhoT_drho[0])],
                     [(1, Ps_rhoT_1), (2, dPs_rhoT_drho[1])]),
        )
    Es_T_0, Es_T_1 = (Es/Ts[:, None])[0], (Es/Ts[:, None])[-1]
    if dEs_T_drho is None:
        spl_Es_T_0 = CubicSpline(np.log(rhos), Es_T_0)
        spl_Es_T_1 = CubicSpline(np.log(rhos), Es_T_1)
    else:
        spl_Es_T_0 = CubicSpline(np.log(rhos), Es_T_0, bc_type=((1, dEs_T_drho[0][0]), (1, dEs_T_drho[1][0])))
        spl_Es_T_1 = CubicSpline(np.log(rhos), Es_T_1, bc_type=((1, dEs_T_drho[0][-1]), (1, dEs_T_drho[1][-1])))

    def F_interpolation(_Ts, _rhos):
        _Fs_isochore = spl_isotherms(np.log(_rhos))
        _dFs_drho_isochore = spl_isotherms(np.log(_rhos), 1)
        spl_Fs_isochore = CubicSpline(np.log(Ts), _Fs_isochore, axis=0,
            bc_type=((1, -spl_Es_T_0(np.log(_rhos))), (1, -spl_Es_T_1(np.log(_rhos)))),
        )
        spl_dFs_drho_isochore = CubicSpline(np.log(Ts), _dFs_drho_isochore, axis=0)
        _Fs = spl_Fs_isochore(np.log(_Ts))
        _dFs_dT = spl_Fs_isochore(np.log(_Ts), 1)
        _dFs_drho = spl_dFs_drho_isochore(np.log(_Ts))
        return _Fs, _dFs_drho, _dFs_dT

    return F_interpolation


    # Function to map the entropy values to the T-P grid
def map_values(df, target_df) :
    merged = pd.merge(target_df, df, on=['T(K)', 'logP(GPa)'], how='outer', suffixes=('', '_drop'))
    for col in merged.columns:
        if col.endswith('_drop'):
            original_col = col[:-5]
            merged[original_col] = merged[original_col].combine_first(merged[col])
            merged = merged.drop(columns=[col])
    return merged

def read_coreP_scanvv10(filename) :
    data = pd.read_csv(filename, sep=' ')
    Ts = data['Temp'].unique()
    rhos = data['rho'].unique()
    #Ss = data['Entropy[kJ/g/K]']
    #Es = data['En[kJ/g]']
    #Ps = data['Pr[GPa]']
    return data, Ts, rhos, #Ss, Es, Ps 

