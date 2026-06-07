#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: redshift_sampling.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 13-11-2025
#    DESCRIPTION: Redshift sampling for gravitational wave sources
#                 accounting for comoving volume and time dilation.
#
#                 Merger-rate models:
#                   Uniform:          R(z) = const
#                   PowerLaw:         R(z) = (1+z)^lambda
#                   Madau-Dickinson:  R(z) = (1+z)^gamma / [1+((1+z)/(1+z_p))^kappa]
#
#                 Observed distribution: p(z) ∝ R(z) * dVc/dz / (1+z)
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
    z = np.asarray(z, dtype=float)
    scalar = (z.ndim == 0)
    z = np.atleast_1d(z)

    p_z = np.zeros_like(z)
    mask = (z >= 0) & (z <= z_max)
    if np.any(mask):
        dV_dz = comoving_volume_element(z[mask])
        p_z[mask] = dV_dz / (1 + z[mask])

    return float(p_z[0]) if scalar else p_z

def sample_redshift(n_samples, z_min=0.0, z_max=10.0, seed=None):
    """
    Sample redshifts from the uniform-rate distribution:

    p(z) ∝ dVc/dz / (1+z)

    i.e. R(z) = const (uniform in comoving volume and source-frame time).

    Parameters:
    - n_samples: Number of redshift values to sample
    - z_min: Minimum redshift (default: 0.0)
    - z_max: Maximum redshift (default: 10.0)
    - seed: Random seed for reproducibility (default: None)

    Returns:
    - z_samples: Array of sampled redshifts
    """
    return _inverse_cdf_sample_redshift(
        n_samples, lambda z: np.ones_like(z), z_min, z_max, seed)


# ============================================================================
# PowerLaw redshift model: R(z) = (1+z)^lambda
# ============================================================================

def powerlaw_redshift_pdf(z, lamb, z_max=10.0):
    """
    Unnormalized probability density for the PowerLaw merger-rate model.

    p(z) ∝ (1+z)^lambda * dVc/dz / (1+z)

    Parameters:
    - z: Redshift (scalar or array)
    - lamb: Power-law index lambda (lamb > 0 = rate increases with z)
    - z_max: Maximum redshift (default: 10.0)

    Returns:
    - p(z): Unnormalized probability density
    """
    z = np.asarray(z, dtype=float)
    scalar = (z.ndim == 0)
    z = np.atleast_1d(z)

    p_z = np.zeros_like(z)
    mask = (z >= 0) & (z <= z_max)
    if np.any(mask):
        # R(z) = (1+z)^lambda
        rate = (1.0 + z[mask]) ** lamb
        dV_dz = comoving_volume_element(z[mask])
        # p(z) ∝ R(z) * dVc/dz / (1+z)
        p_z[mask] = rate * dV_dz / (1.0 + z[mask])

    return float(p_z[0]) if scalar else p_z


def sample_redshift_powerlaw(n_samples, lamb, z_min=0.0, z_max=10.0, seed=None):
    """
    Sample redshifts from the PowerLaw merger-rate model:

    p(z) ∝ (1+z)^lambda * dVc/dz / (1+z)

    Parameters:
    - n_samples: Number of redshift values to sample
    - lamb: Power-law index lambda
    - z_min: Minimum redshift (default: 0.0)
    - z_max: Maximum redshift (default: 10.0)
    - seed: Random seed for reproducibility (default: None)

    Returns:
    - z_samples: Array of sampled redshifts
    """
    # R(z) = (1+z)^lambda
    rate_fn = lambda z: (1.0 + z) ** lamb
    return _inverse_cdf_sample_redshift(n_samples, rate_fn, z_min, z_max, seed)


# ============================================================================
# Madau-Dickinson redshift model:
#   R(z) = (1+z)^gamma / [1 + ((1+z)/(1+z_peak))^kappa]
# ============================================================================

def madau_dickinson_redshift_pdf(z, gamma, kappa, z_peak, z_max=10.0):
    """
    Unnormalized probability density for the Madau-Dickinson merger-rate model.

    R(z) = (1+z)^gamma / [1 + ((1+z)/(1+z_peak))^kappa]
    p(z) ∝ R(z) * dVc/dz / (1+z)

    Parameters:
    - z: Redshift (scalar or array)
    - gamma: Low-z power-law slope
    - kappa: High-z turnover steepness
    - z_peak: Peak redshift of the merger rate
    - z_max: Maximum redshift (default: 10.0)

    Returns:
    - p(z): Unnormalized probability density
    """
    z = np.asarray(z, dtype=float)
    scalar = (z.ndim == 0)
    z = np.atleast_1d(z)

    p_z = np.zeros_like(z)
    mask = (z >= 0) & (z <= z_max)
    if np.any(mask):
        zz = z[mask]
        # R(z) = (1+z)^gamma / [1 + ((1+z)/(1+z_peak))^kappa]
        rate = ((1.0 + zz) ** gamma
                / (1.0 + ((1.0 + zz) / (1.0 + z_peak)) ** kappa))
        dV_dz = comoving_volume_element(zz)
        # p(z) ∝ R(z) * dVc/dz / (1+z)
        p_z[mask] = rate * dV_dz / (1.0 + zz)

    return float(p_z[0]) if scalar else p_z


def sample_redshift_madau_dickinson(n_samples, gamma, kappa, z_peak,
                                     z_min=0.0, z_max=10.0, seed=None):
    """
    Sample redshifts from the Madau-Dickinson merger-rate model:

    R(z) = (1+z)^gamma / [1 + ((1+z)/(1+z_peak))^kappa]
    p(z) ∝ R(z) * dVc/dz / (1+z)

    Parameters:
    - n_samples: Number of redshift values to sample
    - gamma: Low-z power-law slope
    - kappa: High-z turnover steepness
    - z_peak: Peak redshift of the merger rate
    - z_min: Minimum redshift (default: 0.0)
    - z_max: Maximum redshift (default: 10.0)
    - seed: Random seed for reproducibility (default: None)

    Returns:
    - z_samples: Array of sampled redshifts
    """
    # R(z) = (1+z)^gamma / [1 + ((1+z)/(1+z_peak))^kappa]
    rate_fn = lambda z: ((1.0 + z) ** gamma
                         / (1.0 + ((1.0 + z) / (1.0 + z_peak)) ** kappa))
    return _inverse_cdf_sample_redshift(n_samples, rate_fn, z_min, z_max, seed)


# ============================================================================
# Shared inverse-CDF sampler
# ============================================================================

def _inverse_cdf_sample_redshift(n_samples, rate_fn, z_min, z_max, seed,
                                  n_grid=2000):
    """
    Inverse-CDF sampling from p(z) ∝ rate_fn(z) * dVc/dz / (1+z).

    rate_fn(z) is the merger-rate density R(z) evaluated on a grid.
    """
    from scipy.integrate import cumulative_trapezoid

    rng = np.random.default_rng(seed)
    z_grid = np.linspace(z_min, z_max, n_grid)
    z_grid[0] = max(z_grid[0], 1e-6)

    # p(z) ∝ R(z) * dVc/dz / (1+z)
    dV = np.array([comoving_volume_element(zi) for zi in z_grid])
    pdf = rate_fn(z_grid) * dV / (1.0 + z_grid)
    pdf = np.maximum(pdf, 0.0)

    cdf = cumulative_trapezoid(pdf, z_grid, initial=0.0)
    cdf /= cdf[-1]

    return np.interp(rng.random(n_samples), cdf, z_grid)

