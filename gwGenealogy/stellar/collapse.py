#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: collapse.py
#
#    Stellar collapse remnant mass prescriptions.
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
import h5py
from scipy import interpolate

_DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'stellar_evolution_data.h5')

# Lazy-loaded interpolators (initialized on first call)
_MremInterpol_F12d = None
_MremInterpol_SEVNdelayed = None


def _load_F12d_interpolator():
    """Load the Fryer 2012 delayed interpolator on first use."""
    global _MremInterpol_F12d
    if _MremInterpol_F12d is not None:
        return _MremInterpol_F12d

    with h5py.File(_DATA_FILE, 'r') as f:
        grp = f['Fryer12_delayed']
        M_grid = grp['Mzams'][:]
        Z_grid = grp['Z'][:]
        Mrem = grp['Mrem'][:]

    _MremInterpol_F12d = interpolate.RegularGridInterpolator(
        (M_grid, Z_grid), Mrem, method='linear', bounds_error=True
    )
    return _MremInterpol_F12d


def _load_SEVNdelayed_interpolator():
    """Load the SEVN delayed interpolator on first use."""
    global _MremInterpol_SEVNdelayed
    if _MremInterpol_SEVNdelayed is not None:
        return _MremInterpol_SEVNdelayed

    with h5py.File(_DATA_FILE, 'r') as f:
        grp = f['SEVN_delayed']
        Mzams = grp['Mzams'][:]
        Zvalues = grp['Z'][:]
        Mrem_delayed = grp['Mrem'][:]

    _MremInterpol_SEVNdelayed = interpolate.RegularGridInterpolator(
        (Mzams, Zvalues), Mrem_delayed, method='linear', bounds_error=True
    )
    return _MremInterpol_SEVNdelayed


def compute_Mrem_Fryer12_delayed_rapster(M, Z, mass_gap_low=45.0, mass_gap_high=120.0):
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

    if Z.size == 1:
        Z = np.full_like(M, Z[0])

    points = np.column_stack((M, Z))
    Mrem = interp(points)

    out = Mrem * (np.heaviside(mass_gap_low - Mrem, 0)
                  + np.heaviside(Mrem - mass_gap_high, 0))

    if out.size == 1:
        return float(out[0])
    return out


def compute_Mrem_SEVN_delayed_rapster(M, Z, mass_gap_low=55.0, mass_gap_high=120.0):
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

    if Z.size == 1:
        Z = np.full_like(M, Z[0])

    points = np.column_stack((M, Z))
    Mrem = interp(points)

    out = Mrem * (np.heaviside(mass_gap_low - Mrem, 0)
                  + np.heaviside(Mrem - mass_gap_high, 0))

    if out.size == 1:
        return float(out[0])
    return out


