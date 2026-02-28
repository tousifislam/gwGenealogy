#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: 1gbh.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 02-28-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from .stellar_evolution import sample_zams_masses, evolve_stars


def sample_1g_bh_masses_from_stellar_collapse(n_samples, Z=0.0002, model='F12d',
                                              m_zams_min=10.0, m_zams_max=150.0,
                                              imf='salpeter', imf_alpha=-2.35,
                                              seed=None, **kwargs):
    """
    Generate a population of 1st-generation black holes from stellar evolution.

    Convenience function that chains sample_zams_masses -> evolve_stars ->
    filter mass-gap zeros.

    Parameters:
    -----------
    n_samples : int
        Number of ZAMS masses to sample (actual BH count will be smaller
        after filtering mass-gap zeros)
    Z : float
        Metallicity (default: 0.0002)
    model : str
        Stellar evolution model: 'F12d' or 'SEVNdelayed' (default: 'F12d')
    m_zams_min : float
        Minimum ZAMS mass in solar masses (default: 10.0)
    m_zams_max : float
        Maximum ZAMS mass in solar masses (default: 150.0)
    imf : str
        Initial mass function: 'salpeter' or 'uniform' (default: 'salpeter')
    imf_alpha : float
        Power-law index for Salpeter IMF (default: -2.35)
    seed : int or None
        Random seed for reproducibility
    **kwargs
        Additional keyword arguments passed to evolve_stars
        (e.g., mass_gap_low, mass_gap_high)

    Returns:
    --------
    m_bh : array
        Array of BH masses in solar masses (mass-gap zeros removed)
    """
    M_ZAMS = sample_zams_masses(n_samples, m_zams_min=m_zams_min,
                                m_zams_max=m_zams_max, imf=imf,
                                imf_alpha=imf_alpha, seed=seed)
    m_rem = evolve_stars(M_ZAMS, Z, model=model, **kwargs)
    m_rem = np.atleast_1d(m_rem)
    return m_rem[m_rem > 0]
