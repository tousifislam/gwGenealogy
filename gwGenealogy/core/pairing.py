#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: pairing.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED: 
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from .distributions import sample_powerlaw_1d, sample_uniform_1d, sample_uniform_in_log_1d

def sample_mass_selective_pairing_modelA(n_samples, m_min=3.0, m_max=50.0, 
                                   alpha=-1.6, beta=6.7, seed=None,
                                   m1_distribution='powerlaw'):
    """
    Model A: Selective pairing for binary BH masses
    
    Pairing probability: p(m1) ∝ m1^α, p(m2|m1) ∝ m2^β with m1 > m2
    Default parameters α = -1.6 and β = 6.7 from Ref. [1] using current GW data
    
    This model favors mergers with m1 ≃ m2 (equal mass binaries), as predicted
    for mass-segregated clusters and supported by observational data.
    
    Parameters:
    - n_samples: Number of binary systems to generate
    - m_min: Minimum mass in solar masses (default: 3.0)
    - m_max: Maximum mass in solar masses (default: 50.0)
    - alpha: Power-law index for primary mass distribution (default: -1.6)
    - beta: Power-law index for secondary mass conditional distribution (default: 6.7)
    - seed: Random seed for reproducibility (default: None)
    - m1_distribution: Distribution for primary mass (default: 'powerlaw')
                       Options: 'uniform', 'log', 'powerlaw'
    
    Returns:
    - m1: Array of primary masses (larger mass)
    - m2: Array of secondary masses (smaller mass)
    
    References: https://arxiv.org/pdf/1906.05295
    
    """
    # Sample primary mass from specified distribution
    if m1_distribution == 'uniform':
        m1 = sample_uniform_1d(n_samples, low=m_min, high=m_max, seed=seed)
    elif m1_distribution == 'log':
        m1 = sample_uniform_in_log_1d(n_samples, low=m_min, high=m_max, seed=seed, base=10)
    elif m1_distribution == 'powerlaw':
        m1 = sample_powerlaw_1d(n_samples, beta=alpha, xmin=m_min, xmax=m_max, seed=seed)
    else:
        raise ValueError(f"Invalid m1_distribution: {m1_distribution}. Choose 'uniform', 'log', or 'powerlaw'.")
    
    # Sample secondary mass conditionally: p(m2|m1) ∝ m2^β for m2 ∈ [m_min, m1]
    rng = np.random.default_rng(None if seed is None else seed + 1)
    m2 = np.zeros(n_samples)
    
    for i in range(n_samples):
        # For each primary mass m1[i], sample m2 from power-law in [m_min, m1[i]]
        # Using inverse transform sampling for power-law
        u = rng.uniform(0, 1)
        
        if beta == -1:
            # Special case: β = -1 gives logarithmic distribution
            m2[i] = m_min * (m1[i]/m_min)**u
        else:
            if beta + 1 != 0:
                # General power-law: x = [u*(xmax^(β+1) - xmin^(β+1)) + xmin^(β+1)]^(1/(β+1))
                m2[i] = (u * (m1[i]**(beta + 1) - m_min**(beta + 1)) + m_min**(beta + 1))**(1/(beta + 1))
            else:
                m2[i] = m_min * (m1[i]/m_min)**u
    
    return m1, m2

def sample_mass_selective_pairing_modelB(n_samples, m_min_primary=3.0, m_max_primary=50.0, 
                                m_min_secondary=3.0, seed=None,
                                m1_distribution='uniform', alpha=-1.6):
    """
    O'Leary pairing model for dynamical binary black holes
    
    Primary mass m1 is randomly drawn from single BHs and BHs in loose binaries.
    Secondary mass m2 is randomly drawn in [m_min_secondary, m1] following:
    p(m2) ∝ (m1 + m2)^4
    
    This model accounts for gravitational focusing in dense stellar environments,
    where more massive binaries have larger cross-sections for interactions.
    
    Parameters:
    - n_samples: Number of binary systems to generate
    - m_min_primary: Minimum mass for primary in solar masses (default: 5.0)
    - m_max_primary: Maximum mass for primary in solar masses (default: 50.0)
    - m_min_secondary: Minimum mass for secondary in solar masses (default: 3.0)
    - seed: Random seed for reproducibility (default: None)
    - m1_distribution: Distribution for primary mass (default: 'uniform')
                       Options: 'uniform', 'log', 'powerlaw'
    - alpha: Power-law index for primary mass if m1_distribution='powerlaw' (default: -1.6)
    
    Returns:
    - m1: Array of primary masses (larger mass)
    - m2: Array of secondary masses (smaller mass)
    
    References:
    - O'Leary et al. 2016: https://arxiv.org/pdf/1602.02809
    - Mapelli et al. 2021: https://arxiv.org/pdf/2103.05016 (Eq 9)
    """
    # Sample primary mass from specified distribution
    if m1_distribution == 'uniform':
        m1 = sample_uniform_1d(n_samples, low=m_min_primary, high=m_max_primary, seed=seed)
    elif m1_distribution == 'log':
        m1 = sample_uniform_in_log_1d(n_samples, low=m_min_primary, high=m_max_primary, 
                                      seed=seed, base=10)
    elif m1_distribution == 'powerlaw':
        m1 = sample_powerlaw_1d(n_samples, beta=alpha, xmin=m_min_primary, 
                                xmax=m_max_primary, seed=seed)
    else:
        raise ValueError(f"Invalid m1_distribution: {m1_distribution}. Choose 'uniform', 'log', or 'powerlaw'.")
    
    # Sample secondary mass with p(m2) ∝ (m1 + m2)^4
    # Using rejection sampling for this conditional distribution
    rng = np.random.default_rng(None if seed is None else seed + 1)
    m2 = np.zeros(n_samples)
    
    for i in range(n_samples):
        # For each m1[i], sample m2 from [m_min_secondary, m1[i]] with p(m2) ∝ (m1 + m2)^4
        # Using inverse transform sampling
        
        m_low = m_min_secondary
        m_high = m1[i]
        
        # The PDF is p(m2) ∝ (m1 + m2)^4
        # The CDF is: F(m2) = ∫[m_low to m2] (m1 + x)^4 dx
        # After integration: F(m2) = [(m1 + m2)^5 - (m1 + m_low)^5] / 5
        # Normalized CDF: F_norm(m2) = [(m1 + m2)^5 - (m1 + m_low)^5] / [(m1 + m_high)^5 - (m1 + m_low)^5]
        
        # Inverse: m2 = [u * ((m1 + m_high)^5 - (m1 + m_low)^5) + (m1 + m_low)^5]^(1/5) - m1
        
        u = rng.uniform(0, 1)
        m2[i] = ((u * ((m1[i] + m_high)**5 - (m1[i] + m_low)**5) +
                  (m1[i] + m_low)**5)**(1/5) - m1[i])

    return m1, m2


def select_from_pool(m1, m_pool, pairing='random', beta=4.0, seed=None):
    """
    Select a secondary mass from a pre-generated BH population pool.

    This is the main workhorse for hierarchical merger loops where the
    secondary is drawn from an existing population rather than freshly sampled.

    Parameters:
    -----------
    m1 : float
        Primary mass in solar masses
    m_pool : array
        Array of available BH masses to pair with
    pairing : str
        Pairing model (default: 'random'):
        - 'random': uniform random choice from pool
        - 'fragione': weights proportional to (m1 + m_pool)^beta
        - 'gerosa_modelB': weights proportional to m_pool^beta for m_pool < m1
    beta : float
        Power-law index for weighted pairing (default: 4.0)
    seed : int or None
        Random seed for reproducibility

    Returns:
    --------
    m2 : float
        Selected secondary mass
    idx : int
        Index of selected mass in the pool
    """
    rng = np.random.default_rng(seed)
    m_pool = np.asarray(m_pool)

    if pairing == 'random':
        idx = rng.integers(0, len(m_pool))
    elif pairing == 'fragione':
        weights = (m1 + m_pool)**beta
        weights /= weights.sum()
        idx = rng.choice(len(m_pool), p=weights)
    elif pairing == 'gerosa_modelB':
        mask = m_pool < m1
        if mask.sum() == 0:
            # If no valid m2 < m1, pick closest mass
            idx = np.argmin(m_pool)
        else:
            weights = np.zeros(len(m_pool))
            weights[mask] = m_pool[mask]**beta
            weights /= weights.sum()
            idx = rng.choice(len(m_pool), p=weights)
    else:
        raise ValueError(f"Unknown pairing model: {pairing}. "
                         "Choose 'random', 'fragione', or 'gerosa_modelB'.")

    return m_pool[idx], idx


def sample_1g_masses(n_samples, m_min=3.0, m_max=60.0, imf='uniform',
                     alpha=-2.4, seed=None):
    """
    Sample first-generation BH masses from an initial mass function.

    Parameters:
    -----------
    n_samples : int
        Number of masses to sample
    m_min : float
        Minimum mass in solar masses (default: 3.0)
    m_max : float
        Maximum mass in solar masses (default: 60.0)
    imf : str
        Initial mass function: 'uniform' or 'kroupa' (default: 'uniform')
    alpha : float
        Power-law index for Kroupa IMF (default: -2.4)
    seed : int or None
        Random seed for reproducibility

    Returns:
    --------
    masses : array
        Array of sampled BH masses in solar masses
    """
    rng = np.random.default_rng(seed)

    if imf.lower() == 'uniform':
        return rng.uniform(m_min, m_max, n_samples)
    elif imf.lower() == 'kroupa':
        # Inverse CDF sampling for power-law: p(m) ∝ m^alpha
        g1 = alpha + 1
        if abs(g1) < 1e-10:
            # Special case: p(m) ∝ 1/m -> log-uniform
            return np.exp(rng.uniform(np.log(m_min), np.log(m_max), n_samples))
        else:
            u = rng.uniform(0, 1, n_samples)
            return (m_min**g1 + u * (m_max**g1 - m_min**g1))**(1.0 / g1)
    else:
        raise ValueError(f"Unknown IMF: {imf}. Choose 'uniform' or 'kroupa'.")
