#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: stellar_evolution.py
#
#    Rewritten from Rapster code by Konstantinos Kritos
#    https://github.com/Kkritos/Rapster/
#
#    AUTHOR: Tousif Islam
#    CREATED: 02-28-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import os
import numpy as np
from scipy import interpolate

_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Lazy-loaded interpolators (initialized on first call)
_MremInterpol_F12d = None
_MremInterpol_SEVNdelayed = None


def _load_F12d_interpolator():
    """Load the Fryer 2012 delayed interpolator on first use."""
    global _MremInterpol_F12d
    if _MremInterpol_F12d is not None:
        return _MremInterpol_F12d

    N_grid = 700
    M_grid = np.linspace(10, 340, N_grid)
    Z_grid = np.logspace(np.log10(1e-4), np.log10(2e-2), N_grid)

    data_path = os.path.join(_DATA_DIR, 'MzamsMrem_F12d.txt')
    Mremnants_F12d = np.loadtxt(data_path, unpack=True)

    _MremInterpol_F12d = interpolate.RegularGridInterpolator(
        (M_grid, Z_grid), Mremnants_F12d, method='linear', bounds_error=True
    )
    return _MremInterpol_F12d


def _load_SEVNdelayed_interpolator():
    """Load the SEVN delayed interpolator on first use."""
    global _MremInterpol_SEVNdelayed
    if _MremInterpol_SEVNdelayed is not None:
        return _MremInterpol_SEVNdelayed

    # Load 12 metallicity files
    Mrem_delayed_list = []
    for i in range(1, 13):
        fpath = os.path.join(_DATA_DIR, f'MzamsMrem{i}_delayed.npz')
        data = np.load(fpath)
        Mrem_delayed_list.append(data[f'Mrem{i}'])

    # Stack: shape (Npoints, 12) — ZAMS x metallicity
    Mrem_delayed = np.array(Mrem_delayed_list).T

    # Metallicity grid
    Zvalues = np.array([1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 4.0e-3,
                        6.0e-3, 8.0e-3, 1.0e-2, 1.4e-2, 1.7e-2, 2.0e-2])

    # ZAMS mass grid
    Npoints = 500
    Mzams = np.linspace(15, 340, Npoints)

    _MremInterpol_SEVNdelayed = interpolate.RegularGridInterpolator(
        (Mzams, Zvalues), Mrem_delayed, method='linear', bounds_error=True
    )
    return _MremInterpol_SEVNdelayed


def Mrem_F12d(M, Z, mass_gap_low=45.0, mass_gap_high=120.0):
    """
    Fryer et al. (2012) delayed remnant mass prescription model.

    Uses RAPSTER stellar evolution data interpolated on a regular grid.

    Parameters:
    -----------
    M : float or array
        ZAMS mass in solar masses (valid range: [10, 340])
    Z : float or array
        Absolute metallicity (valid range: [1e-4, 2e-2])
    mass_gap_low : float
        Lower edge of the upper mass gap in solar masses (default: 45.0)
    mass_gap_high : float
        Upper edge of the upper mass gap in solar masses (default: 120.0)

    Returns:
    --------
    float or array
        Remnant mass in solar masses. Returns 0 for BHs in the mass gap.

    References:
    -----------
    Fryer et al. (2012): https://arxiv.org/abs/1110.1726
    """
    interp = _load_F12d_interpolator()

    M = np.atleast_1d(np.asarray(M, dtype=float))
    Z = np.atleast_1d(np.asarray(Z, dtype=float))

    # Broadcast Z to match M if Z is scalar
    if Z.size == 1:
        Z = np.full_like(M, Z[0])

    points = np.column_stack((M, Z))
    Mrem = interp(points)

    # Apply pair-instability mass gap
    out = Mrem * (np.heaviside(mass_gap_low - Mrem, 0)
                  + np.heaviside(Mrem - mass_gap_high, 0))

    if out.size == 1:
        return float(out[0])
    return out


def Mrem_SEVNdelayed(M, Z, mass_gap_low=55.0, mass_gap_high=120.0):
    """
    SEVN delayed SN engine remnant mass prescription.

    Uses RAPSTER SEVN data interpolated on a regular grid across
    12 metallicity values.

    Parameters:
    -----------
    M : float or array
        ZAMS mass in solar masses (valid range: [15, 340])
    Z : float or array
        Absolute metallicity (valid range: [1e-4, 2e-2])
    mass_gap_low : float
        Lower edge of the upper mass gap in solar masses (default: 55.0)
    mass_gap_high : float
        Upper edge of the upper mass gap in solar masses (default: 120.0)

    Returns:
    --------
    float or array
        Remnant mass in solar masses. Returns 0 for BHs in the mass gap.

    References:
    -----------
    Spera & Mapelli (2017) — SEVN code
    """
    interp = _load_SEVNdelayed_interpolator()

    M = np.atleast_1d(np.asarray(M, dtype=float))
    Z = np.atleast_1d(np.asarray(Z, dtype=float))

    # Broadcast Z to match M if Z is scalar
    if Z.size == 1:
        Z = np.full_like(M, Z[0])

    points = np.column_stack((M, Z))
    Mrem = interp(points)

    # Apply pair-instability mass gap
    out = Mrem * (np.heaviside(mass_gap_low - Mrem, 0)
                  + np.heaviside(Mrem - mass_gap_high, 0))

    if out.size == 1:
        return float(out[0])
    return out


def sample_1g_bh_population(n_samples, Z=0.0002, model='F12d',
                            m_zams_min=10.0, m_zams_max=150.0,
                            imf='salpeter', imf_alpha=-2.35, seed=None):
    """
    Generate a population of 1st-generation black holes from stellar evolution.

    Samples ZAMS masses from an initial mass function, applies a stellar
    evolution model to get remnant masses, and filters out mass-gap zeros.

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

    Returns:
    --------
    m_bh : array
        Array of BH masses in solar masses (mass-gap zeros removed)
    """
    rng = np.random.default_rng(seed)

    # Sample ZAMS masses from IMF
    if imf.lower() == 'uniform':
        M_ZAMS = rng.uniform(m_zams_min, m_zams_max, n_samples)
    elif imf.lower() == 'salpeter':
        # Inverse CDF sampling for power-law: p(m) ∝ m^alpha
        alpha = imf_alpha
        g1 = alpha + 1
        u = rng.uniform(0, 1, n_samples)
        M_ZAMS = (m_zams_min**g1 + u * (m_zams_max**g1 - m_zams_min**g1))**(1.0 / g1)
    else:
        raise ValueError(f"Unknown IMF: {imf}. Choose 'salpeter' or 'uniform'.")

    # Apply stellar evolution model
    if model.lower() == 'f12d':
        m_bh = Mrem_F12d(M_ZAMS, Z)
    elif model.lower() == 'sevndelayed':
        # SEVN delayed requires M >= 15
        M_ZAMS = np.clip(M_ZAMS, 15.0, 340.0)
        m_bh = Mrem_SEVNdelayed(M_ZAMS, Z)
    else:
        raise ValueError(f"Unknown model: {model}. Choose 'F12d' or 'SEVNdelayed'.")

    # Filter out mass-gap zeros
    m_bh = m_bh[m_bh > 0]

    return m_bh
