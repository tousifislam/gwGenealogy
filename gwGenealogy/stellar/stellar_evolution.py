#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: stellar_evolution.py
#
#    High-level stellar evolution convenience functions.
#
#    AUTHOR: Tousif Islam
#    CREATED: 02-28-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from .collapse import compute_Mrem_Fryer12_delayed_rapster, compute_Mrem_SEVN_delayed_rapster


def IMF_kroupa(m, alpha3=-2.3):
    """
    Kroupa (2002) initial mass function (broken power law).

    Parameters
    ----------
    m : float or array
        Stellar mass in solar masses
    alpha3 : float
        High-mass spectral index (default: -2.3)

    Returns
    -------
    float or array
        dN/dm in units of 1/Msun

    References
    ----------
    Kroupa (2002): https://arxiv.org/abs/astro-ph/0201098
    """
    m = np.asarray(m, dtype=float)
    scalar_input = (m.ndim == 0)
    m = np.atleast_1d(m)

    m1, m2, m3 = 0.08, 0.50, 1.00
    a0, a1, a2, a3 = -0.3, -1.3, -2.3, alpha3

    c1 = m1**(a0 - a1)
    c2 = c1 * m2**(a1 - a2)
    c3 = c2 * m3**(a2 - a3)

    out = np.piecewise(m, [
        m <= m1,
        (m > m1) & (m <= m2),
        (m > m2) & (m <= m3),
        m > m3,
    ], [
        lambda x: x**a0,
        lambda x: c1 * x**a1,
        lambda x: c2 * x**a2,
        lambda x: c3 * x**a3,
    ])

    return float(out[0]) if scalar_input else out


def sample_kroupa_masses(n_samples, m_min=0.08, m_max=150.0, alpha3=-2.3,
                         seed=None):
    """
    Sample masses from the Kroupa (2002) IMF via inverse CDF on a fine grid.

    Parameters
    ----------
    n_samples : int
        Number of masses to draw
    m_min : float
        Minimum mass in solar masses (default: 0.08)
    m_max : float
        Maximum mass in solar masses (default: 150.0)
    alpha3 : float
        High-mass spectral index (default: -2.3)
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    array
        Sampled masses in solar masses
    """
    rng = np.random.default_rng(seed)

    n_grid = 10000
    m_grid = np.linspace(m_min, m_max, n_grid)
    pdf = IMF_kroupa(m_grid, alpha3=alpha3)

    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]

    u = rng.uniform(0, 1, n_samples)
    return np.interp(u, cdf, m_grid)


def sample_zams_masses(n_samples, m_zams_min=10.0, m_zams_max=150.0,
                       imf='salpeter', imf_alpha=-2.35, seed=None):
    """
    Sample zero-age main sequence (ZAMS) masses from an initial mass function.

    Parameters:
    -----------
    n_samples : int
        Number of ZAMS masses to sample
    m_zams_min : float
        Minimum ZAMS mass in solar masses (default: 10.0)
    m_zams_max : float
        Maximum ZAMS mass in solar masses (default: 150.0)
    imf : str
        Initial mass function: 'salpeter', 'kroupa', or 'uniform'
        (default: 'salpeter')
    imf_alpha : float
        Power-law index for Salpeter IMF (default: -2.35), or high-mass
        slope alpha3 for Kroupa IMF (default: -2.3 if not specified)
    seed : int or None
        Random seed for reproducibility

    Returns:
    --------
    M_ZAMS : array
        Array of ZAMS masses in solar masses
    """
    rng = np.random.default_rng(seed)

    if imf.lower() == 'uniform':
        return rng.uniform(m_zams_min, m_zams_max, n_samples)
    elif imf.lower() == 'salpeter':
        alpha = imf_alpha
        g1 = alpha + 1
        u = rng.uniform(0, 1, n_samples)
        return (m_zams_min**g1 + u * (m_zams_max**g1 - m_zams_min**g1))**(1.0 / g1)
    elif imf.lower() == 'kroupa':
        alpha3 = imf_alpha if imf_alpha != -2.35 else -2.3
        return sample_kroupa_masses(n_samples, m_min=m_zams_min, m_max=m_zams_max,
                                    alpha3=alpha3, seed=seed)
    else:
        raise ValueError(f"Unknown IMF: {imf}. Choose 'salpeter', 'kroupa', or 'uniform'.")


def evolve_stars(M_ZAMS, Z, model='Fryer12_delayed', **kwargs):
    """
    Evolve ZAMS masses to remnant masses using a stellar evolution model.

    Parameters:
    -----------
    M_ZAMS : float or array
        ZAMS masses in solar masses
    Z : float
        Absolute metallicity
    model : str
        Stellar evolution model: 'Fryer12_delayed' or 'SEVN_delayed'
        (default: 'Fryer12_delayed')
    **kwargs
        Additional keyword arguments passed to the underlying model
        (e.g., mass_gap_low, mass_gap_high)

    Returns:
    --------
    M_rem : float or array
        Remnant masses in solar masses. Returns 0 for objects in the
        pair-instability mass gap.
    """
    M_ZAMS = np.atleast_1d(np.asarray(M_ZAMS, dtype=float))

    if model.lower() == 'fryer12_delayed':
        return compute_Mrem_Fryer12_delayed_rapster(M_ZAMS, Z, **kwargs)
    elif model.lower() == 'sevn_delayed':
        M_ZAMS = np.clip(M_ZAMS, 15.0, 340.0)
        return compute_Mrem_SEVN_delayed_rapster(M_ZAMS, Z, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model}. Choose 'Fryer12_delayed' or 'SEVN_delayed'.")


def sample_1g_bh_masses_from_stellar_collapse(n_samples, Z=0.0002,
                                              model='Fryer12_delayed',
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
        Stellar evolution model: 'Fryer12_delayed' or 'SEVN_delayed'
        (default: 'Fryer12_delayed')
    m_zams_min : float
        Minimum ZAMS mass in solar masses (default: 10.0)
    m_zams_max : float
        Maximum ZAMS mass in solar masses (default: 150.0)
    imf : str
        Initial mass function: 'salpeter', 'kroupa', or 'uniform'
        (default: 'salpeter')
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
