#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: misc.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np


def select_from_pool(m1, m_pool, pairing='random', beta=4.0, seed=None):
    """
    Select a secondary mass from a pre-generated BH population pool.

    This is the main workhorse for hierarchical merger loops where the
    secondary is drawn from an existing population rather than freshly sampled.

    Parameters
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

    Returns
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

    Parameters
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

    Returns
    --------
    masses : array
        Array of sampled BH masses in solar masses
    """
    rng = np.random.default_rng(seed)

    if imf.lower() == 'uniform':
        return rng.uniform(m_min, m_max, n_samples)
    elif imf.lower() == 'kroupa':
        g1 = alpha + 1
        if abs(g1) < 1e-10:
            return np.exp(rng.uniform(np.log(m_min), np.log(m_max), n_samples))
        else:
            u = rng.uniform(0, 1, n_samples)
            return (m_min**g1 + u * (m_max**g1 - m_min**g1))**(1.0 / g1)
    else:
        raise ValueError(f"Unknown IMF: {imf}. Choose 'uniform' or 'kroupa'.")
