#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: analytical_vesc.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED: 
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np

# Physical constants
G = 6.67430e-11  # Gravitational constant in SI units (m^3 kg^-1 s^-2)
M_sun = 1.98892e30  # Solar mass in kg
pc_to_m = 3.08567758149137e16  # Parsec to meters

def Mcl_rh_to_vescape(Mcl, r_h):
    """
    Calculate escape velocity from a star cluster using the virial theorem
    
    The root mean square velocity of stars in the cluster is given by:
    <v_*^2>^(1/2) = sqrt(0.4 G M_cl / r_h)
    
    The escape velocity is:
    v_esc = 2 * <v_*^2>^(1/2) = 2 * sqrt(0.4 G M_cl / r_h)
    
    Parameters:
    - Mcl: Total mass of the cluster in solar masses (M_☉)
    - r_h: Half-mass radius in parsecs (pc)
    
    Returns:
    - v_esc: Escape velocity in km/s
    
    Reference:
    - https://arxiv.org/pdf/2210.10055, Equation 1
    
    Example:
    >>> Mcl_rh_to_vescape(1e5, 1.0)
    13.0  # km/s for M_cl = 10^5 M_☉, r_h = 1 pc
    """
    # Convert inputs to SI units
    Mcl_kg = Mcl * M_sun  # Convert solar masses to kg
    r_h_m = r_h * pc_to_m  # Convert parsecs to meters
    
    # Calculate RMS velocity: <v_*^2>^(1/2) = sqrt(0.4 G M_cl / r_h)
    v_rms_squared = 0.4 * G * Mcl_kg / r_h_m
    v_rms = np.sqrt(v_rms_squared)
    
    # Escape velocity: v_esc = 2 * <v_*^2>^(1/2)
    v_esc = 2.0 * v_rms
    
    # Convert from m/s to km/s
    v_esc_kms = v_esc / 1000.0
    
    return v_esc_kms

def Mcl_rho_to_vescape(Mcl, rho):
    """
    Calculate escape velocity from cluster mass and density
    
    The escape velocity is given by:
    v_esc = 40 km/s * (M_tot / 10^5 M_☉)^(1/3) * (ρ / 10^5 M_☉ pc^-3)^(1/6)
    
    This relationship is derived from M_tot and ρ observations and is consistent
    with observational samples of GCs and NSCs.
    
    Info: log10(rho) is 5, 3.3 and 3.3 for NSCs, GCs and YSCs
    
    Parameters:
    - Mcl: Total mass of the cluster in solar masses (M_☉)
    - rho: Density at the half-mass radius in M_☉ pc^-3
    
    Returns:
    - v_esc: Escape velocity in km/s
    
    Reference:
    - https://arxiv.org/pdf/2103.05016, Equation 22
    - Georgiev et al. 2009a,b; Fragione & Silk 2020
    
    Example:
    >>> Mcl_rho_to_vescape(1e5, 1e5)
    40.0  # km/s for M_cl = 10^5 M_☉, ρ = 10^5 M_☉ pc^-3
    """
    # Reference values
    Mcl_ref = 1e5  # M_☉
    rho_ref = 1e5  # M_☉ pc^-3
    v_ref = 40.0   # km/s
    
    # Calculate escape velocity using the scaling relation
    v_esc = v_ref * (Mcl / Mcl_ref)**(1.0/3.0) * (rho / rho_ref)**(1.0/6.0)
    
    return v_esc