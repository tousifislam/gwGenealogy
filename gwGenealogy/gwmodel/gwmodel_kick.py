#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: gwmodel_kick.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 02-28-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np


def gwModel_kick_aligned(q, s1z, s2z):
    """
    Updated RIT aligned-spin recoil (kick) for binaries with spins along +/-z.

    Uses parameters fitted on combined dataset (SXS NR + RIT NR + BHPT data).

    Parameters:
    -----------
    q : float or array
        Mass ratio m1/m2 >= 1 (primary is m1)
    s1z : float or array
        Dimensionless spin of primary along z in [-1, 1]
    s2z : float or array
        Dimensionless spin of secondary along z in [-1, 1]

    Returns:
    --------
    V_kick : float or array
        Kick velocity in km/s

    Notes:
    ------
    R^2 = 0.9974, RMS = 108.5 km/s on training data.

    Reference:
    ----------
    Based on RIT formula from arXiv:1406.7295, refitted to expanded dataset
    including extreme mass ratios from BHPT (q up to ~128).
    """
    # Fitted parameters (full training)
    A = 1.177151e+04
    B = -9.281482e-01
    H = 7.410261e+03
    H2a = 5.845639e+00
    H2b = -7.440300e-01
    H3a = -6.095334e-01
    H3b = -1.321148e+00
    H3c = -1.442264e+00
    H3d = -1.790316e-02
    H3e = 6.691620e+00
    H4a = -8.580474e-01
    H4b = -2.668094e+00
    H4c = 3.622004e+00
    H4d = -2.214556e+00
    H4e = 1.395472e+00
    H4f = 2.920338e-01
    a_deg = 1.468588e+02
    b_deg = 1.107239e+02
    c_deg = 1.346647e+02

    # Convert to arrays
    q = np.asarray(q, dtype=float)
    s1z = np.asarray(s1z, dtype=float)
    s2z = np.asarray(s2z, dtype=float)

    # Input validation
    if np.any(q < 1):
        raise ValueError("q must be >= 1 (m1/m2).")
    if np.any((s1z < -1) | (s1z > 1) | (s2z < -1) | (s2z > 1)):
        raise ValueError("s1z and s2z must be in [-1, 1].")

    # Mass-ratio quantities
    eta = q / (1.0 + q)**2
    delta_m = (q - 1.0) / (q + 1.0)

    # Tilded spin combinations (aligned case; z-components only)
    S_tilde_par = (s1z + q**2 * s2z) / (1.0 + q)**2
    Delta_tilde_par = (s1z - q * s2z) / (1.0 + q)

    # In-plane (orbital-plane) spin-orbit recoil polynomial
    poly = (
        Delta_tilde_par
        + H2a * S_tilde_par * delta_m
        + H2b * Delta_tilde_par * S_tilde_par
        + H3a * (Delta_tilde_par**2) * delta_m
        + H3b * (S_tilde_par**2) * delta_m
        + H3c * Delta_tilde_par * (S_tilde_par**2)
        + H3d * (Delta_tilde_par**3)
        + H3e * Delta_tilde_par * (delta_m**2)
        + H4a * S_tilde_par * (Delta_tilde_par**2) * delta_m
        + H4b * (S_tilde_par**3) * delta_m
        + H4c * S_tilde_par * (delta_m**3)
        + H4d * Delta_tilde_par * S_tilde_par * (delta_m**2)
        + H4e * Delta_tilde_par * (S_tilde_par**3)
        + H4f * S_tilde_par * (Delta_tilde_par**3)
    )
    Vperp = H * (eta**2) * poly

    # Mass-asymmetry contribution (nonspinning baseline)
    Vm = A * (eta**2) * delta_m * (1.0 + B * eta)

    # Phase angle between Vm and Vperp
    xi = np.deg2rad(a_deg + b_deg * S_tilde_par + c_deg * delta_m * Delta_tilde_par)

    # Total recoil magnitude (aligned spins have no V_parallel term)
    V_kick = np.sqrt(Vm**2 + Vperp**2 + 2.0 * Vm * Vperp * np.cos(xi))

    return V_kick


def finalkick_gwModel(q, chi1, chi2):
    """
    Compute kick velocity using the gwModel aligned-spin formula.

    This is a convenience wrapper that handles q < 1 by inverting it,
    and projects scalar spin magnitudes to z-components for the aligned model.

    Parameters:
    -----------
    q : float
        Mass ratio (either convention; will be converted to q >= 1)
    chi1 : float
        Dimensionless spin magnitude of primary
    chi2 : float
        Dimensionless spin magnitude of secondary

    Returns:
    --------
    v_kick : float
        Kick velocity in km/s
    """
    if q < 1:
        q = 1.0 / q
    return gwModel_kick_aligned(q, chi1, chi2)
