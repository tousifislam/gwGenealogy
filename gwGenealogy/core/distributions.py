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

import numpy as np
import matplotlib.pyplot as plt

from .rcparams import set_rcparams

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

def sample_loguniform_1d(n_samples, low=1.0, high=10.0, seed=None, plot=False, bins=50, base=10):
    """
    Sample uniformly in log-space (log-uniform distribution)
    
    Parameters:
    - n_samples: Number of samples to generate
    - low: Lower bound of the distribution (default: 1.0, must be > 0)
    - high: Upper bound of the distribution (default: 10.0)
    - seed: Random seed for reproducibility (default: None)
    - plot: Whether to plot the distribution (default: False)
    - bins: Number of histogram bins for plotting (default: 50)
    - base: Logarithm base (default: 10)
    
    Returns:
    - Array of sampled points
    
    Note: low must be > 0 for log-uniform sampling
    
    Example:
    samples = sample_loguniform_1d(1000, low=1, high=100, seed=42)
    """
    if low <= 0:
        raise ValueError("low must be > 0 for log-uniform sampling")
    
    rng = np.random.default_rng(seed)
    
    # Sample uniformly in log-space
    log_low = np.log(low) / np.log(base)
    log_high = np.log(high) / np.log(base)
    log_samples = rng.uniform(log_low, log_high, n_samples)
    
    # Transform back to linear space
    samples = base ** log_samples
    
    if plot:
        plt.figure(figsize=(4,4))
        plt.hist(samples, bins=bins, density=True, alpha=0.7, histtype='stepfilled', 
                 color='C4', edgecolor='C4')
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

def sample_maxwellian_1d(n_samples, sigma=1.0, seed=None, plot=False, bins=50):
    """
    Sample from a Maxwell (speed) distribution.

    The Maxwell distribution describes the magnitude of a 3D isotropic
    Gaussian velocity vector with per-component dispersion sigma:

        f(v) = sqrt(2/pi) * (v^2 / sigma^3) * exp(-v^2 / (2*sigma^2))

    Mean: sigma * sqrt(8/pi).

    Parameters:
    - n_samples: Number of samples to generate
    - sigma: 1D velocity dispersion (default: 1.0)
    - seed: Random seed for reproducibility (default: None)
    - plot: Whether to plot the distribution (default: False)
    - bins: Number of histogram bins for plotting (default: 50)

    Returns:
    - Array of sampled speeds (>= 0)
    """
    rng = np.random.default_rng(seed)
    vx = rng.normal(0, sigma, n_samples)
    vy = rng.normal(0, sigma, n_samples)
    vz = rng.normal(0, sigma, n_samples)
    samples = np.sqrt(vx**2 + vy**2 + vz**2)

    if plot:
        plt.figure(figsize=(4,4))
        plt.hist(samples, bins=bins, density=True, alpha=0.7, histtype='stepfilled',
                 color='C5', edgecolor='C5')
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
    
    For spin magnitudes (https://arxiv.org/abs/2111.03634)
    
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
