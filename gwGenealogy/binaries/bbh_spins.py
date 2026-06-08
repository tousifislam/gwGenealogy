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
from ..utils.coordinates import spins_polar_to_cartesian_vectors
from ..utils.distributions import sample_uniform_1d, sample_beta_1d


def sample_spins(n_samples, chi_min=0, chi_max=1,
                 spin_magnitude='uniform', spin_angles='isotropic',
                 beta_a=1.4, beta_b=3.6,
                 tilt_beta_a=None, tilt_beta_b=None, seed=None):
    """
    Sample spin vectors for binary black hole systems with flexible distributions.

    Parameters:
    -----------
    n_samples : int
        Number of spin pairs to generate
    chi_min : float
        Minimum dimensionless spin magnitude (default: 0)
    chi_max : float
        Maximum dimensionless spin magnitude (default: 1)
    spin_magnitude : str
        Distribution for spin magnitudes. Options:
        - 'uniform': Uniform distribution between chi_min and chi_max
        - 'beta': Beta distribution with parameters beta_a and beta_b
    spin_angles : str
        Distribution for spin orientations. Options:
        - 'isotropic': Uniform distribution on the sphere (physically motivated)
        - 'uniform': Uniform distribution in theta angle
        - 'beta': Beta(tilt_beta_a, tilt_beta_b) on [0, 1] radian
    beta_a : float
        Alpha parameter for spin magnitude Beta distribution (default: 1.4)
        Only used when spin_magnitude='beta'
    beta_b : float
        Beta parameter for spin magnitude Beta distribution (default: 3.6)
        Only used when spin_magnitude='beta'
        Default values are based on https://arxiv.org/abs/2111.03634
    tilt_beta_a : float or None
        Alpha parameter for tilt Beta distribution.
        Required when spin_angles='beta'
    tilt_beta_b : float or None
        Beta parameter for tilt Beta distribution.
        Required when spin_angles='beta'
    seed : int or None
        Random seed for reproducibility (default: None)

    Returns:
    --------
    chi1 : numpy array of shape (n_samples, 3)
        3D spin vectors for the primary black hole
    chi2 : numpy array of shape (n_samples, 3)
        3D spin vectors for the secondary black hole
    """
    rng = np.random.default_rng(seed)
    seed_mag = int(rng.integers(0, 2**31))
    seed_ang = int(rng.integers(0, 2**31))

    a1, a2 = sample_spin_magnitudes(
        n_samples, chi_min, chi_max, spin_magnitude, beta_a, beta_b, seed=seed_mag
    )

    theta1, theta2, phi1, phi2 = sample_spin_angles(
        n_samples, spin_angles, tilt_beta_a=tilt_beta_a,
        tilt_beta_b=tilt_beta_b, seed=seed_ang
    )

    chi1, chi2 = spins_polar_to_cartesian_vectors(
        a1, a2, theta1, theta2, phi1, phi2
    )

    return chi1, chi2


def sample_spin_magnitudes(n_samples, chi_min, chi_max, spin_magnitude,
                           beta_a, beta_b, seed=None):
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
        Distribution type: 'uniform' or 'beta'
    beta_a : float
        Alpha parameter for Beta distribution
    beta_b : float
        Beta parameter for Beta distribution
    seed : int or None
        Random seed for reproducibility (default: None)

    Returns:
    --------
    a1 : numpy array of shape (n_samples,)
        Spin magnitudes for primary black hole
    a2 : numpy array of shape (n_samples,)
        Spin magnitudes for secondary black hole
    """
    rng = np.random.default_rng(seed)
    seed1 = int(rng.integers(0, 2**31))
    seed2 = int(rng.integers(0, 2**31))

    if spin_magnitude == 'uniform':
        a1 = sample_uniform_1d(n_samples, low=chi_min, high=chi_max, seed=seed1)
        a2 = sample_uniform_1d(n_samples, low=chi_min, high=chi_max, seed=seed2)
    elif spin_magnitude == 'beta':
        a1 = sample_beta_1d(n_samples, a=beta_a, b=beta_b, seed=seed1)
        a2 = sample_beta_1d(n_samples, a=beta_a, b=beta_b, seed=seed2)
    else:
        raise ValueError(f"Unknown spin_magnitude: '{spin_magnitude}'. "
                        f"Must be 'uniform' or 'beta'.")

    return a1, a2


def sample_spin_angles(n_samples, spin_angles='isotropic',
                       tilt_beta_a=None, tilt_beta_b=None, seed=None):
    """
    Sample spin orientation angles.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    spin_angles : str
        Distribution type (default: 'isotropic'):
        - 'isotropic': uniform on the sphere (uniform in cos theta)
        - 'uniform': uniform in theta on [0, pi]
        - 'beta': Beta(a, b) on [0, 1] radian (preferentially aligned)
    tilt_beta_a : float or None
        Alpha parameter for Beta tilt distribution (required when spin_angles='beta')
    tilt_beta_b : float or None
        Beta parameter for Beta tilt distribution (required when spin_angles='beta')
    seed : int or None
        Random seed for reproducibility (default: None)

    Returns:
    --------
    theta1, theta2 : numpy array of shape (n_samples,)
        Polar (tilt) angles in radians
    phi1, phi2 : numpy array of shape (n_samples,)
        Azimuthal angles in radians
    """
    rng = np.random.default_rng(seed)
    s1, s2, s3, s4 = (int(rng.integers(0, 2**31)) for _ in range(4))

    if spin_angles == 'isotropic':
        theta1 = np.arccos(sample_uniform_1d(n_samples, low=-1, high=1, seed=s1))
        theta2 = np.arccos(sample_uniform_1d(n_samples, low=-1, high=1, seed=s2))
    elif spin_angles == 'uniform':
        theta1 = sample_uniform_1d(n_samples, low=0, high=np.pi, seed=s1)
        theta2 = sample_uniform_1d(n_samples, low=0, high=np.pi, seed=s2)
    elif spin_angles == 'beta':
        if tilt_beta_a is None or tilt_beta_b is None:
            raise ValueError("tilt_beta_a and tilt_beta_b are required "
                             "when spin_angles='beta'")
        theta1 = sample_beta_1d(n_samples, a=tilt_beta_a, b=tilt_beta_b, seed=s1)
        theta2 = sample_beta_1d(n_samples, a=tilt_beta_a, b=tilt_beta_b, seed=s2)
    else:
        raise ValueError(f"Unknown spin_angles: '{spin_angles}'. "
                        f"Must be 'isotropic', 'uniform', or 'beta'.")

    phi1 = sample_uniform_1d(n_samples, low=0, high=2*np.pi, seed=s3)
    phi2 = sample_uniform_1d(n_samples, low=0, high=2*np.pi, seed=s4)

    return theta1, theta2, phi1, phi2

