#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: spin_distributions.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 11-13-2025
#    LAST MODIFIED: 
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from .conversions import angles_to_cartesian
from .distributions import *

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
        Default values are based on https://arxiv.org/abs/2111.03634
        
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

