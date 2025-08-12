#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: final_bbh_mass_spin_kick_prec_surrogate_emri.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED: 
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import matplotlib.pyplot as plt

try:
    import surfinBH
except ImportError:
    raise ImportError("surfinBH package required. Install with: pip install surfinBH")
fit_name = 'NRSur7dq4EmriRemnant'
try:
    fit = surfinBH.LoadFits(fit_name)
except Exception as e:
    raise RuntimeError(f"Failed to load {fit_name}: {e}")

def individual_nrsur_emri_remnant_properties(q, chi_primary, chi_secondary, 
                            return_errors=False, verbose=False):
    """
    Calculate remnant black hole properties using NRSur7dq4Remnant fit.
    
    This function provides a clean interface to the surfinBH NRSur7dq4Remnant
    surrogate model, which is trained on numerical relativity simulations
    and provides state-of-the-art accuracy for remnant predictions.
    
    Parameters:
    - q: Mass ratio m1/m2 with q ≥ 1 (primary is heavier)
    - chi_primary: 3D spin vector of primary BH [χ1x, χ1y, χ1z]
    - chi_secondary: 3D spin vector of secondary BH [χ2x, χ2y, χ2z]
    - return_errors: Return 1-sigma error estimates (default: False)
    - verbose: Print detailed results (default: False)
    
    Returns:
    - remnant_mass: Final mass Mf/M (fraction of initial total mass)
    - remnant_spin: Final spin vector [χfx, χfy, χfz]
    - remnant_kick: Kick velocity vector [vx, vy, vz] in units of c
    - errors: If return_errors=True, returns (mass_err, spin_err, kick_err)
    
    Notes:
    - Reference epoch is t=-100M from waveform amplitude peak
    - Results are in co-orbital frame at t=-100M
    - Valid for q ≤ 4, |χi| ≤ 0.8 (training domain)
    
    Example:
    >>> mf, chif, vf = nrsur_remnant_properties(3.2, [0.5, 0.05, 0.3], [-0.5, -0.05, 0.1])
    """
    
    # Convert inputs to numpy arrays
    q = np.asarray(q, dtype=float)
    chi_primary = np.asarray(chi_primary, dtype=float)
    chi_secondary = np.asarray(chi_secondary, dtype=float)
    
    # Validate inputs
    if np.any(q < 1.0):
        raise ValueError("Mass ratio q must be ≥ 1 (primary is heavier)")
    
    if np.any(np.linalg.norm(chi_primary, axis=-1) > 1.0):
        raise ValueError("Primary spin magnitude must be ≤ 1")
    
    if np.any(np.linalg.norm(chi_secondary, axis=-1) > 1.0):
        raise ValueError("Secondary spin magnitude must be ≤ 1")
    
    # Calculate all remnant properties
    if return_errors:
        mf, chif, vf, mf_err, chif_err, vf_err = fit.all(q, chi_primary, chi_secondary)
        
        if verbose:
            print(f"🕳️  NRSur7dq4 Remnant Properties  🕳️")
            print(f"Mass ratio q: {q}")
            print(f"Primary spin: [{chi_primary[0]:.3f}, {chi_primary[1]:.3f}, {chi_primary[2]:.3f}]")
            print(f"Secondary spin: [{chi_secondary[0]:.3f}, {chi_secondary[1]:.3f}, {chi_secondary[2]:.3f}]")
            print(f"")
            print(f"Remnant mass: {mf:.6f} ± {mf_err:.6f}")
            print(f"Remnant spin: [{chif[0]:.4f}, {chif[1]:.4f}, {chif[2]:.4f}] ± [{chif_err[0]:.4f}, {chif_err[1]:.4f}, {chif_err[2]:.4f}]")
        
        return mf, chif, vf, (mf_err, chif_err, vf_err)
    
    else:
        mf, chif, vf = fit.all(q, chi_primary, chi_secondary)[:3]
        
        if verbose:
            print(f"🕳️  NRSur7dq4 Remnant Properties  🕳️")
            print(f"Mass ratio q: {q}")
            print(f"Primary spin: [{chi_primary[0]:.3f}, {chi_primary[1]:.3f}, {chi_primary[2]:.3f}]")
            print(f"Secondary spin: [{chi_secondary[0]:.3f}, {chi_secondary[1]:.3f}, {chi_secondary[2]:.3f}]")
            print(f"")
            print(f"Remnant mass: {mf:.6f}")
            print(f"Remnant spin: [{chif[0]:.4f}, {chif[1]:.4f}, {chif[2]:.4f}]")
        
        return mf, chif, vf


def bbh_final_state_precessing_NRSur7dq4EmriRemnant(m1, m2, s1_vec_input, s2_vec_input, bbh='precessing'):
    """
    Calculate remnant properties for multiple binary systems using a loop.
    
    Parameters:
    - m1: Array of primary masses
    - m2: Array of secondary masses  
    - s1_vec_input: Array of primary spin vectors, shape (N, 3)
    - s2_vec_input: Array of secondary spin vectors, shape (N, 3)
    - bbh: str, 'precessing', 'nonspinning', 'aligned_spin_projected', 'aligned_spin', 'single_spin_precession'
    
    Returns:
    - mf_vals: Final masses in solar masses
    - chif_vals: Final spin vectors, shape (N, 3) 
    - vf_kms_vals: None
    """
    # Convert to arrays
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    # Different BBH configurations from the same model
    if bbh=='precessing':
        s1_vec = s1_vec_input
        s2_vec = s2_vec_input
    elif bbh=='nonspinning':
        s1_vec = []
        s2_vec = []
        for i in range(len(m1)):
            s1_vec.append([0,0,0])
            s2_vec.append([0,0,0])
    elif bbh=='aligned_spin_projected':
        s1_vec = []
        s2_vec = []
        for i in range(len(m1)):
            s1_vec.append([0,0,np.linalg.norm(s1_vec_input[i])])
            s2_vec.append([0,0,np.linalg.norm(s2_vec_input[i])])
    elif bbh=='aligned_spin':
        s1_vec = []
        s2_vec = []
        for i in range(len(m1)):
            s1_vec.append([0,0,s1_vec_input[i][-1]])
            s2_vec.append([0,0,s2_vec_input[i][-1]])
    elif bbh=='single_spin_precession':
        s1_vec = []
        s2_vec = []
        for i in range(len(m1)):
            s1_vec.append([s1_vec_input[i][0],s1_vec_input[i][1],s1_vec_input[i][2]])
            s2_vec.append([0,0,0])
    else:
        raise ValueError("BBH type not recognized!")
        
    # vectorize the spin arrays
    s1_vec = np.asarray(s1_vec)
    s2_vec = np.asarray(s2_vec)
    
    # Calculate mass ratios q ≥ 1 (element-wise)
    q = np.maximum(m1/m2, m2/m1)
    
    # Total masses for conversion
    total_masses = m1 + m2
    
    # Initialize output arrays
    mf_vals = []
    chif_vals = []
    
    # Loop through each binary system
    for i in range(len(m1)):
        # Call individual_nrsur_remnant_properties for single system
        mf, chif, vf = individual_nrsur_emri_remnant_properties(
            q=q[i], 
            chi_primary=s1_vec[i], 
            chi_secondary=s2_vec[i]
        )
        
        # Store results
        mf_vals.append(mf)
        chif_vals.append(chif)
        
    # Convert to numpy arrays
    mf_vals = np.array(mf_vals)
    chif_vals = np.array(chif_vals)
    
    # Convert mass fractions to solar masses
    mf_solar = mf_vals * total_masses
    
    return mf_solar, chif_vals