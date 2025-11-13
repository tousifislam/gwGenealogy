#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: distributions.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED: 
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

from .rcparams import *

import numpy as np
import matplotlib.pyplot as plt
from .conversions import angles_to_cartesian

def sample_uniform_1d(n_samples, low=0.0, high=1.0, seed=None, plot=False, bins=50):
    """
    Sample using NumPy's recommended random generator
    
    Parameters:
    - n_samples: Number of samples to generate
    - low: Lower bound of the uniform distribution (default: 0.0)
    - high: Upper bound of the uniform distribution (default: 1.0)
    - seed: Random seed for reproducibility (default: None)
    - plot: Whether to plot the distribution (default: False)
    - bins: Number of histogram bins for plotting (default: 50)
    
    Returns:
    - Array of sampled points

    samples = sample_uniform_rng(1000, seed=42)
    
    """
    rng = np.random.default_rng(seed)
    samples = rng.uniform(low, high, n_samples)
    
    if plot:
        plt.figure(figsize=(4,4))
        plt.hist(samples, bins=bins, density=True, alpha=0.7, histtype='stepfilled', 
                 color='C0', edgecolor='C0')
        plt.xlabel('Value', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return samples

def sample_gaussian_1d(n_samples, mean=0.0, std=1.0, seed=None, plot=False, bins=50):
    """
    Sample from a Gaussian (normal) distribution
    
    Parameters:
    - n_samples: Number of samples to generate
    - mean: Mean of the Gaussian distribution (default: 0.0)
    - std: Standard deviation of the Gaussian distribution (default: 1.0)
    - seed: Random seed for reproducibility (default: None)
    - plot: Whether to plot the distribution (default: False)
    - bins: Number of histogram bins for plotting (default: 50)
    
    Returns:
    - Array of sampled points
    """
    rng = np.random.default_rng(seed)
    samples = rng.normal(mean, std, n_samples)
    
    if plot:
        plt.figure(figsize=(4,4))
        plt.hist(samples, bins=bins, density=True, alpha=0.7, histtype='stepfilled', 
                 color='C1', edgecolor='C1')
        plt.xlabel('Value', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return samples

def sample_lognormal_1d(n_samples, mean=0.0, sigma=1.0, seed=None, plot=False, bins=50):
    """
    Sample from a log-normal distribution
    
    Parameters:
    - n_samples: Number of samples to generate
    - mean: Mean of the underlying normal distribution (default: 0.0)
    - sigma: Standard deviation of the underlying normal distribution (default: 1.0)
    - seed: Random seed for reproducibility (default: None)
    - plot: Whether to plot the distribution (default: False)
    - bins: Number of histogram bins for plotting (default: 50)
    
    Returns:
    - Array of sampled points
    """
    rng = np.random.default_rng(seed)
    samples = rng.lognormal(mean, sigma, n_samples)
    
    if plot:
        plt.figure(figsize=(4,4))
        plt.hist(samples, bins=bins, density=True, alpha=0.7, histtype='stepfilled', 
                 color='C2', edgecolor='C2')
        plt.xlabel('Value', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return samples
    
def sample_powerlaw_1d(n_samples, beta, xmin=1.0, xmax=10.0, seed=None, plot=False, bins=50):
    """
    Sample from a power-law distribution with PDF ∝ x^β
    
    Parameters:
    - n_samples: Number of samples to generate
    - beta: Power-law exponent (β in x^β)
    - xmin: Minimum value of the distribution (default: 1.0)
    - xmax: Maximum value of the distribution (default: 10.0)
    - seed: Random seed for reproducibility (default: None)
    - plot: Whether to plot the distribution (default: False)
    - bins: Number of histogram bins for plotting (default: 50)
    
    Returns:
    - Array of sampled points
    """
    rng = np.random.default_rng(seed)
    
    # Use inverse transform sampling for power-law distribution
    # For PDF ∝ x^β, CDF = (x^(β+1) - xmin^(β+1)) / (xmax^(β+1) - xmin^(β+1))
    # Inverse CDF: x = [u*(xmax^(β+1) - xmin^(β+1)) + xmin^(β+1)]^(1/(β+1))
    
    if beta == -1:
        # Special case: β = -1 gives logarithmic distribution
        u = rng.uniform(0, 1, n_samples)
        samples = xmin * (xmax/xmin)**u
    else:
        u = rng.uniform(0, 1, n_samples)
        if beta + 1 != 0:
            samples = (u * (xmax**(beta + 1) - xmin**(beta + 1)) + xmin**(beta + 1))**(1/(beta + 1))
        else:
            # This shouldn't happen if beta != -1, but included for completeness
            samples = xmin * (xmax/xmin)**u
    
    if plot:
        plt.figure(figsize=(4,4))
        plt.hist(samples, bins=bins, density=True, alpha=0.7, histtype='stepfilled', 
                 color='C3', edgecolor='C3')
        plt.xlabel('Value', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return samples

def sample_beta_1d(n_samples, a=1.4, b=3.6, seed=None, plot=False, bins=50):
    """
    Sample from a Beta distribution
    
    Parameters:
    - n_samples: Number of samples to generate
    - a: Alpha parameter of the Beta distribution (default: 1.4)
    - b: Beta parameter of the Beta distribution (default: 3.6)
    - seed: Random seed for reproducibility (default: None)
    - plot: Whether to plot the distribution (default: False)
    - bins: Number of histogram bins for plotting (default: 50)
    
    Returns:
    - Array of sampled points (values between 0 and 1)
    """
    rng = np.random.default_rng(seed)
    samples = rng.beta(a, b, n_samples)
    
    if plot:
        plt.figure(figsize=(4,4))
        plt.hist(samples, bins=bins, density=True, alpha=0.7, histtype='stepfilled', 
                 color='C3', edgecolor='C3')
        plt.xlabel('Value', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title(f'Beta: a={a}, b={b}', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return samples

def sample_spins(n_samples, chi_min=0, chi_max=1, 
                 spin_magnitude='uniform', spin_angles='isotropic',
                 beta_a=1.4, beta_b=3.6):
    """
    Sample spin vectors for binary black hole systems with flexible distributions.
    
    Parameters:
    -----------
    n_samples : int
        Number of spin pairs to generate
    chi_min : float
        Minimum dimensionless spin magnitude (default: 0)
        Used for 'uniform' and 'random' magnitude distributions
    chi_max : float
        Maximum dimensionless spin magnitude (default: 1)
        Used for 'uniform' and 'random' magnitude distributions
    spin_magnitude : str
        Distribution for spin magnitudes. Options:
        - 'uniform': Uniform distribution between chi_min and chi_max
        - 'random': Scaled uniform distribution: chi_min + (chi_max - chi_min) * U(0,1)
        - 'beta': Beta distribution with parameters beta_a and beta_b
    spin_angles : str
        Distribution for spin orientations. Options:
        - 'isotropic': Uniform distribution on the sphere (physically motivated)
        - 'random': Uniform distribution in theta angle
    beta_a : float
        Alpha parameter for Beta distribution (default: 1.4)
        Only used when spin_magnitude='beta'
    beta_b : float
        Beta parameter for Beta distribution (default: 3.6)
        Only used when spin_magnitude='beta'
        
    Returns:
    --------
    chi1 : numpy array of shape (n_samples, 3)
        3D spin vectors for the primary black hole
    chi2 : numpy array of shape (n_samples, 3)
        3D spin vectors for the secondary black hole
    """
    # Sample spin magnitudes
    chi1_mag, chi2_mag = sample_spin_magnitudes(
        n_samples, chi_min, chi_max, spin_magnitude, beta_a, beta_b
    )
    
    # Sample angles
    isotropic = (spin_angles == 'isotropic')
    if spin_angles not in ['isotropic', 'random']:
        raise ValueError(f"Unknown spin_angles: '{spin_angles}'. "
                        f"Must be 'isotropic' or 'random'.")
    
    cos_theta1, cos_theta2, phi1, phi2 = sample_spin_angles(n_samples, isotropic)
    
    # Convert to 3D Cartesian vectors
    chi1, chi2 = angles_to_cartesian(chi1_mag, chi2_mag, cos_theta1, cos_theta2, phi1, phi2)
    
    return chi1, chi2

def sample_spin_magnitudes(n_samples, chi_min, chi_max, spin_magnitude, beta_a, beta_b):
    """
    Sample spin magnitudes based on specified distribution.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    chi_min : float
        Minimum spin magnitude
    chi_max : float
        Maximum spin magnitude
    spin_magnitude : str
        Distribution type: 'uniform', 'random', or 'beta'
    beta_a : float
        Alpha parameter for Beta distribution
    beta_b : float
        Beta parameter for Beta distribution
        
    Returns:
    --------
    chi1_mag : numpy array of shape (n_samples,)
        Spin magnitudes for primary black hole
    chi2_mag : numpy array of shape (n_samples,)
        Spin magnitudes for secondary black hole
    """
    if spin_magnitude == 'uniform':
        chi1_mag = np.random.uniform(chi_min, chi_max, n_samples)
        chi2_mag = np.random.uniform(chi_min, chi_max, n_samples)
    elif spin_magnitude == 'random':
        chi1_mag = chi_min + (chi_max - chi_min) * np.random.uniform(size=n_samples)
        chi2_mag = chi_min + (chi_max - chi_min) * np.random.uniform(size=n_samples)
    elif spin_magnitude == 'beta':
        chi1_mag = sample_beta_1d(n_samples, a=beta_a, b=beta_b)
        chi2_mag = sample_beta_1d(n_samples, a=beta_a, b=beta_b)
    else:
        raise ValueError(f"Unknown spin_magnitude: '{spin_magnitude}'. "
                        f"Must be 'uniform', 'random', or 'beta'.")
    
    return chi1_mag, chi2_mag

def sample_spin_angles(n_samples, isotropic):
    """
    Sample spin orientation angles.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    isotropic : bool
        If True, use isotropic distribution (uniform on sphere).
        If False, use uniform distribution in theta.
        
    Returns:
    --------
    cos_theta1 : numpy array of shape (n_samples,)
        Cosine of polar angle for primary black hole
    cos_theta2 : numpy array of shape (n_samples,)
        Cosine of polar angle for secondary black hole
    phi1 : numpy array of shape (n_samples,)
        Azimuthal angle for primary black hole
    phi2 : numpy array of shape (n_samples,)
        Azimuthal angle for secondary black hole
    """
    if isotropic:
        cos_theta1 = np.random.uniform(-1, 1, n_samples)
        cos_theta2 = np.random.uniform(-1, 1, n_samples)
    else:
        cos_theta1 = np.cos(np.random.uniform(0, np.pi, n_samples))
        cos_theta2 = np.cos(np.random.uniform(0, np.pi, n_samples))
    
    phi1 = np.random.uniform(0, 2*np.pi, n_samples)
    phi2 = np.random.uniform(0, 2*np.pi, n_samples)
    
    return cos_theta1, cos_theta2, phi1, phi2

