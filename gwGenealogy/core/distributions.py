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

def sample_isotropic_spins(n_samples, chi_min=0, chi_max=1):
    """
    Sample spin magnitudes uniformly between chi_min and chi_max for both black holes,
    and random orientations in 3D space.
    
    Parameters:
    -----------
    n_samples : int
        Number of spin pairs to generate
    chi_min : float
        Minimum dimensionless spin magnitude (default: -0.9)
    chi_max : float
        Maximum dimensionless spin magnitude (default: 0.9)
        
    Returns:
    --------
    chi1 : numpy array of shape (n_samples, 3)
        3D spin vectors for the primary black hole
    chi2 : numpy array of shape (n_samples, 3)
        3D spin vectors for the secondary black hole
    """
    # Sample spin magnitudes uniformly
    chi1_mag = np.random.uniform(chi_min, chi_max, n_samples)
    chi2_mag = np.random.uniform(chi_min, chi_max, n_samples)
    
    # Sample random directions in 3D space
    # First, generate random angles
    theta1 = np.arccos(np.random.uniform(-1, 1, n_samples))  # polar angle
    phi1 = np.random.uniform(0, 2*np.pi, n_samples)          # azimuthal angle
    theta2 = np.arccos(np.random.uniform(-1, 1, n_samples))
    phi2 = np.random.uniform(0, 2*np.pi, n_samples)
    
    # Convert to Cartesian coordinates
    chi1 = np.zeros((n_samples, 3))
    chi2 = np.zeros((n_samples, 3))
    
    # Primary black hole spins
    chi1[:, 0] = chi1_mag * np.sin(theta1) * np.cos(phi1)  # x component
    chi1[:, 1] = chi1_mag * np.sin(theta1) * np.sin(phi1)  # y component
    chi1[:, 2] = chi1_mag * np.cos(theta1)                 # z component
    
    # Secondary black hole spins
    chi2[:, 0] = chi2_mag * np.sin(theta2) * np.cos(phi2)
    chi2[:, 1] = chi2_mag * np.sin(theta2) * np.sin(phi2)
    chi2[:, 2] = chi2_mag * np.cos(theta2)
    
    return chi1, chi2
