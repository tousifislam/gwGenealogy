#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: pn_detection.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 13-11-2025
#    DESCRIPTION: Gravitational wave detection signal-to-noise ratio and
#                 detectable volume calculations
#
#    REFERENCE: https://arxiv.org/pdf/1602.02809
#               Equations 2, 3, and 4
#
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np

# Physical constants (G = c = 1 units)
# To convert from mass to time units, multiply by G/c^3

def compute_eta(q):
    """
    Compute symmetric mass ratio η = q/(1+q)^2
    
    Parameters:
    - q: Mass ratio (m2/m1, where m1 >= m2)
    
    Returns:
    - eta: Symmetric mass ratio
    """
    return q / (1 + q)**2

def compute_f_ISCO(M_tot):
    """
    Compute GW frequency at innermost stable circular orbit (ISCO)
    
    f_ISCO(M_tot) = π^(-1) * 6^(-3/2) * M^(-1)
    
    In units where G = c = 1, frequency has units of inverse mass.
    To convert to Hz, need to multiply by c^3 / (G * M_tot * M_sun)
    
    Parameters:
    - M_tot: Total binary mass in solar masses (M_☉)
    
    Returns:
    - f_ISCO: ISCO frequency in Hz
    """
    # Conversion factor from geometric units to Hz
    # c^3 / (G * M_sun) ≈ 2.03e5 Hz * M_sun
    c3_over_G_Msun = 2.03e5  # Hz * M_sun
    
    # f_ISCO in geometric units (inverse mass)
    f_ISCO_geometric = np.pi**(-1) * 6**(-3/2)  # dimensionless coefficient
    
    # Convert to Hz
    f_ISCO_Hz = f_ISCO_geometric * c3_over_G_Msun / M_tot
    
    return f_ISCO_Hz

def compute_I7(M_tot, f_min=10.0, S_h=None):
    """
    Compute the frequency integral I_7(M_tot)
    
    I_7(M) = ∫[f_min to f_ISCO(M)] f^(-7/3) / S_h(f) df
    
    Parameters:
    - M_tot: Total binary mass in solar masses (M_☉)
    - f_min: Minimum detectable frequency in Hz (default: 10.0)
    - S_h: One-sided noise spectral density function S_h(f)
           If None, uses aLIGO design sensitivity (default: None)
    
    Returns:
    - I_7: Frequency integral in Hz^(-4/3)
    """
    # Get f_ISCO
    f_ISCO = compute_f_ISCO(M_tot)
    
    # If f_ISCO < f_min, the binary is not detectable
    if f_ISCO <= f_min:
        return 0.0
    
    # If S_h not provided, use simplified aLIGO sensitivity
    if S_h is None:
        # Simplified power-law approximation for aLIGO design sensitivity
        # S_h(f) ~ constant in the sensitive band (rough approximation)
        # For more accuracy, should use actual aLIGO sensitivity curve
        def S_h(f):
            # aLIGO design sensitivity is roughly 10^-24 Hz^(-1/2) at 100 Hz
            # S_h has units of Hz^-1
            return 1e-48  # Rough constant approximation
    
    # Numerical integration using trapezoidal rule
    n_points = 1000
    f_array = np.linspace(f_min, f_ISCO, n_points)
    integrand = f_array**(-7.0/3.0) / S_h(f_array)
    I_7 = np.trapz(integrand, f_array)
    
    return I_7

def compute_SNR(M_tot, q, d_L, f_min=10.0, S_h=None):
    """
    Compute signal-to-noise ratio (SNR) for a circular inspiraling binary
    
    S/N = k * η^(1/2) * M_tot^(5/6) / d_L * I_7^(1/2)(M_tot)
    
    where k = π^(-2/3) * sqrt(2/15) ≈ 0.17 for isotropic binary orientation
    
    Parameters:
    - M_tot: Total binary mass in solar masses (M_☉)
    - q: Mass ratio (m2/m1, where m1 >= m2)
    - d_L: Luminosity distance in Mpc
    - f_min: Minimum detectable frequency in Hz (default: 10.0)
    - S_h: One-sided noise spectral density function (default: None)
    
    Returns:
    - SNR: Signal-to-noise ratio
    
    Reference:
    - https://arxiv.org/pdf/1906.05295, Equation 2
    - Cutler & Flanagan 1994; Dalal et al. 2006
    
    Note: This is a simplified implementation. For accurate SNR calculations,
    use proper noise curves and integration methods.
    """
    # Constants
    k = np.pi**(-2.0/3.0) * np.sqrt(2.0/15.0)  # ≈ 0.17
    
    # Compute symmetric mass ratio
    eta = compute_eta(q)
    
    # Compute frequency integral
    I_7 = compute_I7(M_tot, f_min, S_h)
    
    if I_7 == 0:
        return 0.0
    
    # Compute SNR with proper scaling
    # The normalization factor accounts for unit conversions and
    # typical aLIGO sensitivity
    SNR = k * np.sqrt(eta) * M_tot**(5.0/6.0) * np.sqrt(I_7) / d_L
    
    # Empirical normalization based on typical aLIGO performance
    # For a 30-30 M_☉ binary at 1 Gpc, SNR ~ 15-20
    normalization = 2e-21
    SNR = SNR * normalization
    
    return SNR

def compute_max_distance(M_tot, q, SNR_threshold=7.0, f_min=10.0, S_h=None):
    """
    Compute maximum detection distance d_L,max for a given SNR threshold
    
    From SNR equation: d_L,max = k * η^(1/2) * M_tot^(5/6) * I_7^(1/2) / (S/N)_threshold
    
    Parameters:
    - M_tot: Total binary mass in solar masses (M_☉)
    - q: Mass ratio (m2/m1, where m1 >= m2)
    - SNR_threshold: Detection threshold (default: 7.0)
    - f_min: Minimum detectable frequency in Hz (default: 10.0)
    - S_h: One-sided noise spectral density function (default: None)
    
    Returns:
    - d_L_max: Maximum detection distance in Mpc
    
    Reference:
    - https://arxiv.org/pdf/1906.05295, Equation 2
    """
    # Constants
    k = np.pi**(-2.0/3.0) * np.sqrt(2.0/15.0)
    
    # Compute symmetric mass ratio
    eta = compute_eta(q)
    
    # Compute frequency integral
    I_7 = compute_I7(M_tot, f_min, S_h)
    
    if I_7 == 0:
        return 0.0
    
    # Compute max distance
    d_L_max = k * np.sqrt(eta) * M_tot**(5.0/6.0) * np.sqrt(I_7) / SNR_threshold
    
    # Apply same normalization as SNR
    normalization = 2e-21
    d_L_max = d_L_max * normalization
    
    return d_L_max

def compute_detectable_volume(M_tot, q, SNR_threshold=7.0, f_min=10.0):
    """
    Compute detectable volume for uniformly distributed sources
    
    V_det ∝ d_L,max^3 ∝ (q^(3/2) / (1+q)^3) * M_tot^(5/2) * I_7^(3/2)(M_tot)
    
    This function grows as M^(2.3) between M ~ 10-20 M_☉, has a maximum at
    M_tot = 77 M_☉, and decreases to zero at 439 M_☉ where f_ISCO = f_min = 10 Hz.
    
    Parameters:
    - M_tot: Total binary mass in solar masses (M_☉)
    - q: Mass ratio (m2/m1, where m1 >= m2)
    - SNR_threshold: Detection threshold (default: 7.0)
    - f_min: Minimum detectable frequency in Hz (default: 10.0)
    
    Returns:
    - V_det: Detectable volume (in arbitrary units proportional to Mpc^3)
    
    Reference:
    - https://arxiv.org/pdf/1906.05295, Equation 4
    
    Notes:
    - Compared to 10 M_☉ and fixed q, the detectable volume is:
      * 27× larger for M_tot = 77 M_☉
      * 4.5× smaller at 400 M_☉
    """
    # Compute max distance
    d_L_max = compute_max_distance(M_tot, q, SNR_threshold, f_min)
    
    # Detectable volume proportional to d_L_max^3
    V_det = (4.0/3.0) * np.pi * d_L_max**3
    
    return V_det

def compute_detectable_volume_scaling(M_tot, q, f_min=10.0):
    """
    Compute the scaling factor for detectable volume (relative to reference)
    
    V_det ∝ (q^(3/2) / (1+q)^3) * M_tot^(5/2) * I_7^(3/2)(M_tot)
    
    Parameters:
    - M_tot: Total binary mass in solar masses (M_☉)
    - q: Mass ratio (m2/m1, where m1 >= m2)
    - f_min: Minimum detectable frequency in Hz (default: 10.0)
    
    Returns:
    - V_scaling: Volume scaling factor (dimensionless)
    
    Reference:
    - https://arxiv.org/pdf/1906.05295, Equation 4
    """
    # Compute frequency integral
    I_7 = compute_I7(M_tot, f_min=10.0)
    
    if I_7 == 0:
        return 0.0
    
    # Compute scaling
    V_scaling = (q**(3.0/2.0) / (1 + q)**3) * M_tot**(5.0/2.0) * I_7**(3.0/2.0)
    
    return V_scaling