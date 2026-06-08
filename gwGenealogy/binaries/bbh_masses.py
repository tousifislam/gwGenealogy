#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==================================================================================
#
#    FILE: mass_distributions.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 13-11-2025
#    DESCRIPTION: Mass distribution models for binary BH systems
#                 Following LIGO-Virgo Scientific Collaboration prescriptions
#                 Mostly based on Section II A of https://arxiv.org/pdf/1703.06223
#
#==================================================================================
__author__ = "Tousif Islam"

import numpy as np
from ..utils.distributions import (sample_uniform_1d, sample_loguniform_1d,
                                  sample_powerlaw_1d)


def sample_masses(n_samples, m_min=5.0, m_max=50.0,
                  m1_distribution='uniform', pairing='random',
                  alpha=-2.5, beta=6.7, seed=None):
    """
    Sample binary black hole masses with flexible m1 distribution and pairing.

    Parameters
    ----------
    n_samples : int
        Number of binary systems to generate
    m_min : float
        Minimum mass in solar masses (default: 5.0)
    m_max : float
        Maximum mass in solar masses (default: 50.0)
    m1_distribution : str
        Distribution for primary mass (default: 'uniform'):
        - 'uniform': uniform in [m_min, m_max]
        - 'loguniform': uniform in log-space
        - 'powerlaw': p(m1) ∝ m1^alpha
    pairing : str
        Pairing model for secondary mass (default: 'random'):
        - 'random': m2 from same distribution as m1, then sort m1 > m2
        - 'secondary_mass_power_law': p(m2|m1) ∝ m2^beta on [m_min, m1]
        - 'total_mass_power_law': p(m2|m1) ∝ (m1+m2)^4 on [m_min, m1]
    alpha : float
        Power-law index for m1 when m1_distribution='powerlaw' (default: -2.5)
    beta : float
        Power-law index for secondary_mass_power_law pairing (default: 6.7)
    seed : int or None
        Random seed for reproducibility (default: None)

    Returns
    -------
    m1 : array
        Primary masses in solar masses (m1 >= m2)
    m2 : array
        Secondary masses in solar masses

    References
    ----------
    Flat/log/powerlaw models: Abbott et al. (2017), https://arxiv.org/abs/1703.06223
    secondary_mass_power_law: Fragione & Silk (2020), https://arxiv.org/abs/1906.05295
    total_mass_power_law: O'Leary et al. (2016), https://arxiv.org/abs/1602.02809
    """
    rng = np.random.default_rng(seed)
    seed_m1 = int(rng.integers(0, 2**31))
    seed_m2 = int(rng.integers(0, 2**31))

    m1 = _sample_single_mass(n_samples, m_min, m_max, m1_distribution, alpha,
                             seed=seed_m1)

    rng2 = np.random.default_rng(seed_m2)

    if pairing == 'random':
        m2 = _sample_single_mass(n_samples, m_min, m_max, m1_distribution,
                                 alpha, seed=seed_m2)
        m1_out = np.maximum(m1, m2)
        m2_out = np.minimum(m1, m2)
        return m1_out, m2_out

    elif pairing == 'secondary_mass_power_law':
        u = rng2.uniform(0, 1, n_samples)
        if beta == -1:
            m2 = m_min * (m1 / m_min) ** u
        else:
            bp1 = beta + 1
            m2 = (u * (m1**bp1 - m_min**bp1) + m_min**bp1) ** (1.0 / bp1)

    elif pairing == 'total_mass_power_law':
        u = rng2.uniform(0, 1, n_samples)
        m2 = (u * ((2 * m1)**5 - (m1 + m_min)**5)
              + (m1 + m_min)**5) ** (1.0 / 5) - m1

    else:
        raise ValueError(
            f"Unknown pairing: '{pairing}'. Choose 'random', "
            "'secondary_mass_power_law', or 'total_mass_power_law'.")

    return m1, m2


def _sample_single_mass(n_samples, m_min, m_max, distribution, alpha, seed=None):
    if distribution == 'uniform':
        return sample_uniform_1d(n_samples, low=m_min, high=m_max, seed=seed)
    elif distribution == 'loguniform':
        return sample_loguniform_1d(n_samples, low=m_min, high=m_max, seed=seed)
    elif distribution == 'powerlaw':
        return sample_powerlaw_1d(n_samples, beta=alpha, xmin=m_min, xmax=m_max,
                                  seed=seed)
    else:
        raise ValueError(
            f"Unknown m1_distribution: '{distribution}'. "
            "Choose 'uniform', 'loguniform', or 'powerlaw'.")
