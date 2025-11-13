#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: conversions.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED: 
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np

def m1_m2_to_mchirp(m1, m2):
    """
    Convert the component masses of a binary to its chirp mass.
    Parameters
    ==========
    m1: float
        Mass of the heavier object
    m2: float
        Mass of the lighter object
    Returns
    =======
    chirp_mass: float
        Chirp mass of the binary
    """
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2

def m1_m2_to_large_q(m1, m2):
    """
    Compact version: always returns q ≥ 1
    """
    m1, m2 = np.asarray(m1), np.asarray(m2)
    return np.maximum(m1/m2, m2/m1)

def m1_m2_to_small_q(m1, m2):
    """
    Compact version: always returns q ≥ 1
    """
    m1, m2 = np.asarray(m1), np.asarray(m2)
    return np.minimum(m1/m2, m2/m1)

def spin_vectors_to_angles(m1, m2, s1_vec, s2_vec):
    """
    Convert spin vectors to angular parameters used in remnant fits.
    
    Parameters:
    - m1: Primary mass (or array of masses)
    - m2: Secondary mass (or array of masses)  
    - s1_vec: Primary spin vector [sx, sy, sz] or array of shape (N, 3)
    - s2_vec: Secondary spin vector [sx, sy, sz] or array of shape (N, 3)
    
    Returns:
    - theta1: Angle between orbital angular momentum L and primary spin S1 (radians)
    - theta2: Angle between orbital angular momentum L and secondary spin S2 (radians)
    - deltaphi: Angle between projections of S1 and S2 onto orbital plane (radians)
    - q: Mass ratio (m1/m2 if m1≥m2, else m2/m1) - always ≥ 1
    - chi1: Dimensionless spin magnitude of primary |S1|/m1²
    - chi2: Dimensionless spin magnitude of secondary |S2|/m2²
    
    Notes:
    - Assumes orbital angular momentum L is along +z direction
    - Spin vectors are in geometric units (G=c=1)
    - Returns chi1, chi2 as dimensionless spins (|S|/m²)
    """
    # Convert to arrays
    m1 = np.asarray(m1, dtype=float)
    m2 = np.asarray(m2, dtype=float)
    s1_vec = np.asarray(s1_vec, dtype=float)
    s2_vec = np.asarray(s2_vec, dtype=float)
    
    # Handle single vectors vs arrays
    if s1_vec.ndim == 1:
        s1_vec = s1_vec.reshape(1, 3)
        s2_vec = s2_vec.reshape(1, 3)
        single_input = True
    else:
        single_input = False
    
    if m1.ndim == 0:
        m1 = np.full(s1_vec.shape[0], m1)
        m2 = np.full(s1_vec.shape[0], m2)
    
    # Calculate spin magnitudes
    chi1 = np.linalg.norm(s1_vec, axis=1) / (m1**2)  # Dimensionless spin
    chi2 = np.linalg.norm(s2_vec, axis=1) / (m2**2)  # Dimensionless spin
    
    # Calculate mass ratio q ≥ 1 and determine which is primary
    q_12 = m1 / m2  # m1/m2
    q_21 = m2 / m1  # m2/m1
    
    # Primary is the heavier one (q ≥ 1 convention)
    primary_is_m1 = q_12 >= 1.0
    
    # Set up arrays for primary/secondary
    m_primary = np.where(primary_is_m1, m1, m2)
    m_secondary = np.where(primary_is_m1, m2, m1)
    
    s_primary = np.where(primary_is_m1[:, None], s1_vec, s2_vec)
    s_secondary = np.where(primary_is_m1[:, None], s2_vec, s1_vec)
    
    chi_primary = np.where(primary_is_m1, chi1, chi2)
    chi_secondary = np.where(primary_is_m1, chi2, chi1)
    
    # Orbital angular momentum direction (assume +z)
    L_hat = np.array([0, 0, 1])
    
    # Calculate theta1: angle between L and primary spin
    s_primary_unit = s_primary / (np.linalg.norm(s_primary, axis=1, keepdims=True) + 1e-10)
    cos_theta1 = np.dot(s_primary_unit, L_hat)
    cos_theta1 = np.clip(cos_theta1, -1.0, 1.0)  # Handle numerical errors
    theta1 = np.arccos(cos_theta1)
    
    # Calculate theta2: angle between L and secondary spin  
    s_secondary_unit = s_secondary / (np.linalg.norm(s_secondary, axis=1, keepdims=True) + 1e-10)
    cos_theta2 = np.dot(s_secondary_unit, L_hat)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2 = np.arccos(cos_theta2)
    
    # Calculate deltaphi: angle between spin projections on orbital plane
    # Project spins onto orbital plane (xy-plane if L is along z)
    s1_perp = s_primary[:, :2]  # x,y components
    s2_perp = s_secondary[:, :2]  # x,y components
    
    # Handle cases where projections are zero
    s1_perp_mag = np.linalg.norm(s1_perp, axis=1)
    s2_perp_mag = np.linalg.norm(s2_perp, axis=1)
    
    # Initialize deltaphi
    deltaphi = np.zeros(len(s1_perp))
    
    # Only calculate deltaphi where both perpendicular components are non-zero
    valid_mask = (s1_perp_mag > 1e-10) & (s2_perp_mag > 1e-10)
    
    if np.any(valid_mask):
        s1_perp_unit = s1_perp[valid_mask] / s1_perp_mag[valid_mask, None]
        s2_perp_unit = s2_perp[valid_mask] / s2_perp_mag[valid_mask, None]
        
        # Angle between perpendicular projections
        cos_deltaphi = np.sum(s1_perp_unit * s2_perp_unit, axis=1)
        cos_deltaphi = np.clip(cos_deltaphi, -1.0, 1.0)
        
        # Use atan2 to get sign correct
        sin_deltaphi = np.cross(s1_perp_unit, s2_perp_unit)
        deltaphi[valid_mask] = np.arctan2(sin_deltaphi, cos_deltaphi)
        deltaphi[valid_mask] = np.abs(deltaphi[valid_mask])  # Take absolute value
    
    # Return single values if input was single
    if single_input:
        return theta1[0], theta2[0], deltaphi[0], q[0], chi_primary[0], chi_secondary[0]
    else:
        return theta1, theta2, deltaphi, chi_primary, chi_secondary

def angles_to_spin_vectors(theta1, theta2, deltaphi, chi1, chi2):
    """
    Convert angular parameters back to spin vectors (inverse operation).
    
    Parameters:
    - theta1, theta2: Polar angles (radians)
    - deltaphi: Azimuthal angle difference (radians)
    - chi1, chi2: Dimensionless spin magnitudes
    
    Returns:
    - s1_vec: Primary spin vector [sx, sy, sz]
    - s2_vec: Secondary spin vector [sx, sy, sz]
    """
    # Convert to arrays
    theta1 = np.asarray(theta1)
    theta2 = np.asarray(theta2)
    deltaphi = np.asarray(deltaphi)
    chi1 = np.asarray(chi1)
    chi2 = np.asarray(chi2)
    
    # Primary spin vector (φ1 = 0 by convention)
    s1_x = chi1 * m1**2 * np.sin(theta1) * 0  # cos(0)
    s1_y = chi1 * m1**2 * np.sin(theta1) * 0  # sin(0)  
    s1_z = chi1 * m1**2 * np.cos(theta1)
    
    # Secondary spin vector (φ2 = deltaphi)
    s2_x = chi2 * m2**2 * np.sin(theta2) * np.cos(deltaphi)
    s2_y = chi2 * m2**2 * np.sin(theta2) * np.sin(deltaphi)
    s2_z = chi2 * m2**2 * np.cos(theta2)
    
    # Stack into vectors
    if np.isscalar(theta1):
        s1_vec = np.array([s1_x, s1_y, s1_z])
        s2_vec = np.array([s2_x, s2_y, s2_z])
    else:
        s1_vec = np.column_stack([s1_x, s1_y, s1_z])
        s2_vec = np.column_stack([s2_x, s2_y, s2_z])
    
    return s1_vec, s2_vec

def calculate_derived_spin_quantities(q, chi1, chi2, theta1, theta2, phi12):
    """
    Calculate derived quantities needed for kick formula
    
    Parameters:
    q: mass ratio (q <= 1)
    chi1, chi2: spin magnitudes 
    theta1, theta2: angles between L and spin vectors (degrees)
    phi12: azimuthal angle difference between spins (degrees)
    
    Returns:
    eta, Delta_parallel, Delta_perp, chi_tilde_parallel, chi_tilde_perp
    """
    # Convert angles to radians
    theta1_rad = np.radians(theta1)
    theta2_rad = np.radians(theta2)
    phi12_rad = np.radians(phi12)
    
    # Delta parallel component
    Delta_parallel = abs((chi1 * np.cos(theta1_rad) - q * chi2 * np.cos(theta2_rad)) / (1 + q))
    
    # Delta perpendicular component
    Delta_perp_sq = (chi1**2 * np.sin(theta1_rad)**2 + q**2 * chi2**2 * np.sin(theta2_rad)**2 
                    - 2 * q * chi1 * chi2 * np.sin(theta1_rad) * np.sin(theta2_rad) * np.cos(phi12_rad)) / (1 + q)**2
    Delta_perp = np.sqrt(max(0, Delta_perp_sq))
    
    # Chi tilde parallel component
    chi_tilde_parallel = (chi1 * np.cos(theta1_rad) + q**2 * chi2 * np.cos(theta2_rad)) / (1 + q)**2
    
    # Chi tilde perpendicular component
    chi_tilde_perp_sq = (chi1**2 * np.sin(theta1_rad)**2 + q**4 * chi2**2 * np.sin(theta2_rad)**2 
                        + 2 * q**2 * chi1 * chi2 * np.sin(theta1_rad) * np.sin(theta2_rad) * np.cos(phi12_rad)) / (1 + q)**4
    chi_tilde_perp = np.sqrt(max(0, chi_tilde_perp_sq))
    
    return Delta_parallel, Delta_perp, chi_tilde_parallel, chi_tilde_perp