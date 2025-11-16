#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==================================================================================
#
#    FILE: mass_distributions.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 13-11-2025
#    DESCRIPTION: Mass distribution models for 1g+1g binary BH mergers
#                 Following LIGO-Virgo Scientific Collaboration prescriptions
#                 Mostly based on Section II A of https://arxiv.org/pdf/1703.06223
#
#==================================================================================
__author__ = "Tousif Islam"

import numpy as np
from .distributions import (sample_uniform_1d, sample_uniform_in_log_1d, 
                           sample_powerlaw_1d)

def sample_mass_flat(n_samples, m_min=5.0, m_max=50.0, seed=None):
    """
    Model "flat": Uniformly distributed source-frame masses
    
    Both m1 and m2 are uniformly distributed in [m_min, m_max] with m1 > m2
    
    Parameters:
    - n_samples: Number of binary systems to generate
    - m_min: Minimum mass in solar masses (default: 5.0)
    - m_max: Maximum mass in solar masses (default: 50.0)
    - seed: Random seed for reproducibility (default: None)
    
    Returns:
    - m1: Array of primary masses (larger mass)
    - m2: Array of secondary masses (smaller mass)
    """
    # Sample both masses uniformly
    m1_samples = sample_uniform_1d(n_samples, low=m_min, high=m_max, seed=seed)
    
    # Use a different seed for m2 to ensure independence
    seed2 = None if seed is None else seed + 1
    m2_samples = sample_uniform_1d(n_samples, low=m_min, high=m_max, seed=seed2)
    
    # Ensure m1 > m2 by sorting
    m1 = np.maximum(m1_samples, m2_samples)
    m2 = np.minimum(m1_samples, m2_samples)
    
    return m1, m2

def sample_mass_log(n_samples, m_min=5.0, m_max=50.0, seed=None, base=10):
    """
    Model "log": Logarithm of source-frame masses uniformly distributed
    
    The probability distribution p(m1, m2) ∝ 1/(m1*m2)
    
    Parameters:
    - n_samples: Number of binary systems to generate
    - m_min: Minimum mass in solar masses (default: 5.0)
    - m_max: Maximum mass in solar masses (default: 50.0)
    - seed: Random seed for reproducibility (default: None)
    - base: Logarithm base (default: 10)
    
    Returns:
    - m1: Array of primary masses (larger mass)
    - m2: Array of secondary masses (smaller mass)
    """
    # Sample both masses uniformly in log-space
    m1_samples = sample_uniform_in_log_1d(n_samples, low=m_min, high=m_max, 
                                          seed=seed, base=base)
    
    # Use a different seed for m2 to ensure independence
    seed2 = None if seed is None else seed + 1
    m2_samples = sample_uniform_in_log_1d(n_samples, low=m_min, high=m_max, 
                                          seed=seed2, base=base)
    
    # Ensure m1 > m2 by sorting
    m1 = np.maximum(m1_samples, m2_samples)
    m2 = np.minimum(m1_samples, m2_samples)
    
    return m1, m2

def sample_mass_powerlaw(n_samples, m_min=5.0, m_max=50.0, alpha=-2.5, seed=None):
    """
    Model "power law": Power-law distribution for primary mass, uniform for secondary
    
    Primary mass: p(m1) ∝ m1^α with α = -2.5
    Secondary mass: uniform in [m_min, m1]
    
    Parameters:
    - n_samples: Number of binary systems to generate
    - m_min: Minimum mass in solar masses (default: 5.0)
    - m_max: Maximum mass in solar masses (default: 50.0)
    - alpha: Power-law spectral index (default: -2.5)
    - seed: Random seed for reproducibility (default: None)
    
    Returns:
    - m1: Array of primary masses (larger mass)
    - m2: Array of secondary masses (smaller mass)
    """
    # Sample primary mass from power-law distribution
    m1 = sample_powerlaw_1d(n_samples, beta=alpha, xmin=m_min, xmax=m_max, seed=seed)
    
    # Sample secondary mass uniformly in [m_min, m1] for each system
    rng = np.random.default_rng(None if seed is None else seed + 1)
    m2 = np.zeros(n_samples)
    
    for i in range(n_samples):
        m2[i] = rng.uniform(m_min, m1[i])
    
    return m1, m2

