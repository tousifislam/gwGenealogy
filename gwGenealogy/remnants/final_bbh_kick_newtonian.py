#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: final_bbh_kick_newtonian.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED: 
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from scipy.optimize import minimize_scalar

# Physical constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
c = 299792458.0  # m/s
M_sun = 1.98847e30  # kg

def kerr_isco_radius(a):
    """
    Calculate Boyer-Lindquist r_ISCO(a) for equatorial orbits.
    
    Parameters:
    - a: Dimensionless spin parameter (|a| ≤ 1)
         Positive for prograde, negative for retrograde orbits
    
    Returns:
    - ISCO radius in units of GM/c²
    """
    a = np.asarray(a, dtype=float)
    
    # Convenient intermediate variables
    Z1 = 1.0 + (1 - a**2)**(1/3.0) * ((1 + a)**(1/3.0) + (1 - a)**(1/3.0))
    Z2 = np.sqrt(3*a**2 + Z1**2)
    
    # Boyer-Lindquist ISCO radius
    r_isco = 3 + Z2 - np.sign(a) * np.sqrt((3 - Z1) * (3 + Z1 + 2*Z2))
    
    return r_isco
    
def f_q(q):
    """
    Mass ratio function f(q) = q^2 * (1-q) / (1+q)^5
    
    Parameters:
    q: mass ratio (q = m1/m2 <= 1, where m1 is the lighter mass)
    
    Returns:
    f: dimensionless function value
    """
    return q**2 * (1 - q) / (1 + q)**5

def find_f_max():
    """
    Find the maximum value of f(q) and the q value where it occurs
    
    Returns:
    q_max: mass ratio where f(q) is maximum
    f_max: maximum value of f(q)
    """
    # Minimize negative f(q) to find maximum
    result = minimize_scalar(lambda q: -f_q(q), bounds=(0, 1), method='bounded')
    q_max = result.x
    f_max = -result.fun
    return q_max, f_max

def bbh_final_kick_nonprecessing_Fitchett1983(q, a, units='km/s'):
    """
    Calculate Fitchett (1983) Newtonian kick velocity
    Eq(1) of https://arxiv.org/pdf/astro-ph/0402056
    
    V_F ≈ 1480 km/s * (f(q)/f_max) * (2GM/c^2 / r_term)^4
    
    Parameters:
    q: mass ratio (q = m1/m2 <= 1, where m1 is the lighter mass)
    a: spin value
    units: output units ('km/s' or 'm/s')
    
    Returns:
    V_F: Fitchett kick velocity
    """
    # Get ISCO radius - we will use that in place of r_term
    # orbital separation where GW emission terminates, in units of GM/c^2
    r_term = kerr_isco_radius(a)
    
    # Get maximum value of f(q)
    q_max, f_max = find_f_max()
    
    # Calculate f(q)/f_max ratio
    f_ratio = f_q(q) / f_max
    
    # Calculate the kick velocity
    # V_F = 1480 km/s * (f(q)/f_max) * (2GM/c^2 / r_term)^4
    V_F_km_s = 1480.0 * f_ratio * (2.0 / r_term)**4
    
    if units == 'km/s':
        return float(V_F_km_s)
    elif units == 'm/s':
        return float(V_F_km_s * 1000.0)
    else:
        raise ValueError("units must be 'km/s' or 'm/s'")

def f_spin_orbit(q, a_tilde_1, a_tilde_2):
    """
    Spin-orbit correction function f_SO(q, ã₁, ã₂) = q² * (ã₂ - q*ã₁) / (1+q)⁵
    
    Parameters:
    q: mass ratio (q = m1/m2 <= 1, where m1 is the lighter mass)
    a_tilde_1: dimensionless spin of lighter mass (-1 ≤ ã₁ ≤ 1)
    a_tilde_2: dimensionless spin of heavier mass (-1 ≤ ã₂ ≤ 1)
    
    Returns:
    f_SO: spin-orbit correction function value
    """
    return q**2 * (a_tilde_2 - q * a_tilde_1) / (1 + q)**5

def find_f_SO_max():
    """
    Find the maximum spin-orbit correction
    Maximum occurs when q=1, ã₁=-ã₂=±1, giving f_SO_max = 1/16
    
    Returns:
    f_SO_max: maximum value of spin-orbit correction (1/16)
    """
    return 1.0 / 16.0

def bbh_final_kick_nonprecessing_Kidder1995(q, a_tilde_1=0, a_tilde_2=0, units='km/s', all_terms=False):
    """
    Calculate Fitchett kick velocity with spin-orbit correction
    Eq(4) of https://arxiv.org/pdf/astro-ph/0402056
    
    V_total = V_F + V_SO
    where V_F is the original Fitchett kick and V_SO is the spin-orbit correction
    
    Parameters:
    q: mass ratio (q = m1/m2 <= 1, where m1 is the lighter mass)
    M_total: total mass (M = m1 + m2) in solar masses
    r_term: orbital separation where GW emission terminates, in units of GM/c^2
    a_tilde_1: dimensionless spin of lighter mass (-1 ≤ ã₁ ≤ 1)
    a_tilde_2: dimensionless spin of heavier mass (-1 ≤ ã₂ ≤ 1)
    units: output units ('km/s' or 'm/s')
    
    Returns:
    V_total: total kick velocity with spin-orbit correction
    V_F: original Fitchett kick (mass asymmetry only)
    V_SO: spin-orbit correction
    """
    # Get ISCO radius - we will use that in place of r_term
    # orbital separation where GW emission terminates, in units of GM/c^2
    r_term = kerr_isco_radius(a_tilde_2)
    
    # Original Fitchett kick (mass asymmetry)
    V_F = bbh_final_kick_nonprecessing_Fitchett1983(q, a_tilde_2, units)
    
    # Spin-orbit correction
    f_SO = f_spin_orbit(q, a_tilde_1, a_tilde_2)
    f_SO_max = find_f_SO_max()
    
    # Scale the spin-orbit correction by the same factor as the original kick
    # V_SO ∝ f_SO * (2GM/c²/r_term)⁴
    V_SO_km_s = 1480.0 * (f_SO / f_SO_max) * (2.0 / r_term)**4
    
    if units == 'km/s':
        V_SO = V_SO_km_s
    elif units == 'm/s':
        V_SO = V_SO_km_s * 1000.0
    else:
        raise ValueError("units must be 'km/s' or 'm/s'")
    
    # Total kick (vector addition - here we assume they add constructively)
    # In reality, the direction depends on orbital geometry
    V_total = np.sqrt(V_F**2 + V_SO**2)  # Conservative estimate
    
    if all_terms:
        return float(V_total), float(V_F), float(V_SO)
    else:
        return float(V_total)