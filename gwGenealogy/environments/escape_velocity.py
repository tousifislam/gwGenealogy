#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: escape_velocity.py
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
    """Escape velocity from cluster mass and half-mass radius (virial theorem).

    v_esc = 2 * sqrt(0.4 G M_cl / r_h)

    Parameters
    ----------
    Mcl : float or array
        Total cluster mass [Msun]
    r_h : float or array
        Half-mass radius [pc]

    Returns
    -------
    v_esc : float or array
        Escape velocity [km/s]

    Reference: https://arxiv.org/pdf/2210.10055, Equation 1
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
    """Escape velocity from cluster mass and half-mass density (scaling relation).

    v_esc = 40 km/s * (M_tot / 1e5 Msun)^(1/3) * (rho / 1e5 Msun pc^-3)^(1/6)

    Parameters
    ----------
    Mcl : float or array
        Total cluster mass [Msun]
    rho : float or array
        Density at the half-mass radius [Msun pc^-3]

    Returns
    -------
    v_esc : float or array
        Escape velocity [km/s]

    References: Georgiev et al. (2009a,b), Fragione & Silk (2020),
    https://arxiv.org/pdf/2103.05016 Equation 22
    """
    # Reference values
    Mcl_ref = 1e5  # M_☉
    rho_ref = 1e5  # M_☉ pc^-3
    v_ref = 40.0   # km/s

    # Calculate escape velocity using the scaling relation
    v_esc = v_ref * (Mcl / Mcl_ref)**(1.0/3.0) * (rho / rho_ref)**(1.0/6.0)

    return v_esc
