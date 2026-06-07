#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: conversions.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np


# ---- Mass conversions ----

def m1_m2_to_mchirp(m1, m2):
    """
    Chirp mass from component masses.

    Parameters
    ----------
    m1 : float or array
        Mass of the heavier object
    m2 : float or array
        Mass of the lighter object

    Returns
    -------
    float or array
        Chirp mass of the binary
    """
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def m1_m2_to_q(m1, m2):
    """
    Mass ratio q = m1/m2 >= 1.

    If m1 < m2 for any element, the masses are swapped so that q >= 1.
    """
    m1, m2 = np.asarray(m1, dtype=float), np.asarray(m2, dtype=float)
    return np.maximum(m1/m2, m2/m1)


def m1_m2_to_eta(m1, m2):
    """
    Symmetric mass ratio eta = m1*m2 / (m1+m2)^2.
    """
    m1, m2 = np.asarray(m1, dtype=float), np.asarray(m2, dtype=float)
    return (m1 * m2) / (m1 + m2)**2


# ---- Derived spin quantities (all use q = m1/m2 >= 1) ----

def chi_eff(q, chi1z, chi2z):
    """
    Effective inspiral spin parameter.

    chi_eff = (q * chi1z + chi2z) / (1 + q)

    Parameters
    ----------
    q : float or array
        Mass ratio q = m1/m2 >= 1
    chi1z : float or array
        z-component of primary dimensionless spin
    chi2z : float or array
        z-component of secondary dimensionless spin

    Returns
    -------
    float or array
    """
    q = np.asarray(q, dtype=float)
    return (q * chi1z + chi2z) / (1 + q)


def chi_p(q, chi1_perp, chi2_perp):
    """
    Effective precession spin parameter.

    chi_p = max(chi1_perp, (4 + 3q) / (q(4q + 3)) * chi2_perp)

    Parameters
    ----------
    q : float or array
        Mass ratio q = m1/m2 >= 1
    chi1_perp : float or array
        In-plane spin magnitude of primary
    chi2_perp : float or array
        In-plane spin magnitude of secondary

    Returns
    -------
    float or array
    """
    q = np.asarray(q, dtype=float)
    coeff = (4 + 3*q) / (q * (4*q + 3))
    return np.maximum(chi1_perp, coeff * chi2_perp)


def delta_parallel(q, chi1, chi2, theta_1, theta_2):
    """
    Parallel component of the asymmetric spin combination Delta.

    Delta_parallel = |q * chi1 * cos(theta_1) - chi2 * cos(theta_2)| / (1 + q)

    Parameters
    ----------
    q : float or array
        Mass ratio q = m1/m2 >= 1
    chi1, chi2 : float or array
        Dimensionless spin magnitudes
    theta_1, theta_2 : float or array
        Tilt angles in radians

    Returns
    -------
    float or array
    """
    q = np.asarray(q, dtype=float)
    return np.abs(q * chi1 * np.cos(theta_1) - chi2 * np.cos(theta_2)) / (1 + q)


def delta_perp(q, chi1, chi2, theta_1, theta_2, deltaphi):
    """
    Perpendicular component of the asymmetric spin combination Delta.

    Parameters
    ----------
    q : float or array
        Mass ratio q = m1/m2 >= 1
    chi1, chi2 : float or array
        Dimensionless spin magnitudes
    theta_1, theta_2 : float or array
        Tilt angles in radians
    deltaphi : float or array
        Azimuthal angle difference phi_1 - phi_2 in radians

    Returns
    -------
    float or array
    """
    q = np.asarray(q, dtype=float)
    s1 = np.sin(theta_1)
    s2 = np.sin(theta_2)
    val = (q**2 * chi1**2 * s1**2 + chi2**2 * s2**2
           - 2 * q * chi1 * chi2 * s1 * s2 * np.cos(deltaphi)) / (1 + q)**2
    return np.sqrt(np.maximum(0, val))


def chi_tilde_parallel(q, chi1, chi2, theta_1, theta_2):
    """
    Parallel component of the symmetric spin combination chi-tilde.

    chi_tilde_parallel = (q^2 * chi1 * cos(theta_1) + chi2 * cos(theta_2)) / (1 + q)^2

    Parameters
    ----------
    q : float or array
        Mass ratio q = m1/m2 >= 1
    chi1, chi2 : float or array
        Dimensionless spin magnitudes
    theta_1, theta_2 : float or array
        Tilt angles in radians

    Returns
    -------
    float or array
    """
    q = np.asarray(q, dtype=float)
    return (q**2 * chi1 * np.cos(theta_1) + chi2 * np.cos(theta_2)) / (1 + q)**2


def chi_tilde_perp(q, chi1, chi2, theta_1, theta_2, deltaphi):
    """
    Perpendicular component of the symmetric spin combination chi-tilde.

    Parameters
    ----------
    q : float or array
        Mass ratio q = m1/m2 >= 1
    chi1, chi2 : float or array
        Dimensionless spin magnitudes
    theta_1, theta_2 : float or array
        Tilt angles in radians
    deltaphi : float or array
        Azimuthal angle difference phi_1 - phi_2 in radians

    Returns
    -------
    float or array
    """
    q = np.asarray(q, dtype=float)
    s1 = np.sin(theta_1)
    s2 = np.sin(theta_2)
    val = (q**4 * chi1**2 * s1**2 + chi2**2 * s2**2
           + 2 * q**2 * chi1 * chi2 * s1 * s2 * np.cos(deltaphi)) / (1 + q)**4
    return np.sqrt(np.maximum(0, val))
