#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: redshift_sampling.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 13-11-2025
#    DESCRIPTION: Redshift sampling for gravitational wave sources
#                 accounting for comoving volume and time dilation
#
#    REFERENCE: https://arxiv.org/pdf/1906.05295
#==============================================================================

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d

# Cosmological parameters (Planck 2018)
H0 = 67.4  # Hubble constant in km/s/Mpc
Om0 = 0.315  # Matter density parameter
OL0 = 0.685  # Dark energy density parameter
c = 299792.458  # Speed of light in km/s

def E(z, Om0=Om0, OL0=OL0):
    """
    Dimensionless Hubble parameter E(z) = H(z)/H0
    
    For flat ΛCDM: E(z) = sqrt(Ωm(1+z)^3 + ΩΛ)
    
    Parameters:
    - z: Redshift
    - Om0: Matter density parameter (default: 0.315)
    - OL0: Dark energy density parameter (default: 0.685)
    
    Returns:
    - E(z): Dimensionless Hubble parameter
    """
    return np.sqrt(Om0 * (1 + z)**3 + OL0)

def comoving_volume_element(z, H0=H0, Om0=Om0, OL0=OL0):
    """
    Differential comoving volume element dV_c/dz
    
    dV_c/dz = (4π c / H0) * (d_L(z) / (1+z))^2 / E(z)
    
    where d_L(z) is the luminosity distance
    
    Parameters:
    - z: Redshift
    - H0: Hubble constant in km/s/Mpc (default: 67.4)
    - Om0: Matter density parameter (default: 0.315)
    - OL0: Dark energy density parameter (default: 0.685)
    
    Returns:
    - dV_c/dz: Comoving volume element in Gpc^3
    """
    # Compute luminosity distance
    d_L = luminosity_distance(z, H0, Om0, OL0)
    
    # Compute comoving distance
    d_C = d_L / (1 + z)
    
    # Comoving volume element
    dV_dz = 4 * np.pi * c / H0 * d_C**2 / E(z, Om0, OL0)
    
    # Convert from Mpc^3 to Gpc^3
    dV_dz = dV_dz / 1e9
    
    return dV_dz

def luminosity_distance(z, H0=H0, Om0=Om0, OL0=OL0):
    """
    Luminosity distance as a function of redshift
    
    d_L(z) = (1+z) * (c/H0) * ∫[0 to z] dz'/E(z')
    
    Parameters:
    - z: Redshift (can be array)
    - H0: Hubble constant in km/s/Mpc (default: 67.4)
    - Om0: Matter density parameter (default: 0.315)
    - OL0: Dark energy density parameter (default: 0.685)
    
    Returns:
    - d_L: Luminosity distance in Mpc
    """
    # Handle scalar and array inputs
    z_array = np.atleast_1d(z)
    d_L_array = np.zeros_like(z_array, dtype=float)
    
    for i, z_val in enumerate(z_array):
        if z_val <= 0:
            d_L_array[i] = 0
        else:
            # Integrate E(z') from 0 to z
            integral, _ = integrate.quad(lambda zp: 1.0/E(zp, Om0, OL0), 0, z_val)
            d_L_array[i] = (1 + z_val) * (c / H0) * integral
    
    # Return scalar if input was scalar
    if np.isscalar(z):
        return float(d_L_array[0])
    else:
        return d_L_array

def redshift_probability_density(z, z_max=10.0):
    """
    Probability density for redshift distribution
    
    p(z) ∝ (dV_c/dz) / (1+z)
    
    The (1+z) factor accounts for cosmological time dilation, converting
    source-frame time to observer-frame time.
    
    Parameters:
    - z: Redshift (can be array)
    - z_max: Maximum redshift to consider (default: 10.0)
    
    Returns:
    - p(z): Unnormalized probability density
    """
    if np.any(z < 0) or np.any(z > z_max):
        return 0.0
    
    # Compute dV_c/dz
    dV_dz = comoving_volume_element(z)
    
    # Apply time dilation factor
    p_z = dV_dz / (1 + z)
    
    return p_z

def sample_redshift(n_samples, z_min=0.0, z_max=10.0, seed=None):
    """
    Sample redshifts from the distribution p(z) ∝ (dV_c/dz)/(1+z)
    
    This distribution accounts for:
    1. Comoving volume increasing with redshift
    2. Time dilation reducing the observed rate
    
    Parameters:
    - n_samples: Number of redshift values to sample
    - z_min: Minimum redshift (default: 0.0)
    - z_max: Maximum redshift (default: 10.0)
    - seed: Random seed for reproducibility (default: None)
    
    Returns:
    - z_samples: Array of sampled redshifts
    
    Method: Uses inverse transform sampling with interpolation
    """
    rng = np.random.default_rng(seed)
    
    # Create fine grid of redshifts
    n_grid = 1000
    z_grid = np.linspace(z_min, z_max, n_grid)
    
    # Compute probability density on grid
    p_grid = np.array([redshift_probability_density(z, z_max) for z in z_grid])
    
    # Compute CDF by integrating PDF
    cdf_grid = np.zeros_like(p_grid)
    for i in range(1, len(z_grid)):
        cdf_grid[i] = cdf_grid[i-1] + 0.5 * (p_grid[i] + p_grid[i-1]) * (z_grid[i] - z_grid[i-1])
    
    # Normalize CDF
    cdf_grid = cdf_grid / cdf_grid[-1]
    
    # Create inverse CDF interpolator
    inverse_cdf = interp1d(cdf_grid, z_grid, kind='linear', 
                          bounds_error=False, fill_value=(z_min, z_max))
    
    # Sample uniform random numbers and map through inverse CDF
    u_samples = rng.uniform(0, 1, n_samples)
    z_samples = inverse_cdf(u_samples)
    
    return z_samples

def detector_frame_mass(m_source, z):
    """
    Convert source-frame mass to detector-frame mass
    
    m_detector = (1 + z) * m_source
    
    Parameters:
    - m_source: Source-frame mass in M_☉
    - z: Redshift
    
    Returns:
    - m_detector: Detector-frame mass in M_☉
    """
    return (1 + z) * m_source

def source_frame_mass(m_detector, z):
    """
    Convert detector-frame mass to source-frame mass
    
    m_source = m_detector / (1 + z)
    
    Parameters:
    - m_detector: Detector-frame mass in M_☉
    - z: Redshift
    
    Returns:
    - m_source: Source-frame mass in M_☉
    """
    return m_detector / (1 + z)
