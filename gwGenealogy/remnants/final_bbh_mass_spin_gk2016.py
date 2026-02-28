#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: final_bbh_mass_spin_gk2016.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 02-28-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np

# Fitting coefficients from Gerosa & Kesden (2016) and references therein
# Spin coefficients
_t0 = -2.8904
_t2 = -3.51712
_t3 = 2.5763
_s4 = -0.1229
_s5 = 0.4537

# Mass coefficients
_p0 = 0.04827
_p1 = 0.01707


def bbh_final_spin_precessing_GK2016(m1, m2, chi1, chi2, theta1, theta2, dPhi):
    """
    Dimensionless merger remnant spin parameter in [0, 1].
    From Gerosa & Kesden (2016) and references therein.
    Vectorized version.

    Parameters:
    -----------
    m1, m2 : float or array
        Component masses in solar masses
    chi1, chi2 : float or array
        Dimensionless spin magnitudes
    theta1, theta2 : float or array
        Angles between orbital angular momentum and spin vectors (radians)
    dPhi : float or array
        Azimuthal angle difference between spins (radians)

    Returns:
    --------
    chi_f : float or array
        Dimensionless remnant spin magnitude, clipped to [0, 1]

    References:
    -----------
    Gerosa & Kesden (2016): https://arxiv.org/abs/1605.01067
    Rezzolla et al. (2008): https://arxiv.org/abs/0708.3999
    Barausse & Rezzolla (2009): https://arxiv.org/abs/0904.2577
    """
    m1, m2 = np.atleast_1d(m1).astype(float), np.atleast_1d(m2).astype(float)
    chi1, chi2 = np.atleast_1d(chi1).astype(float), np.atleast_1d(chi2).astype(float)
    theta1, theta2 = np.atleast_1d(theta1).astype(float), np.atleast_1d(theta2).astype(float)
    dPhi = np.atleast_1d(dPhi).astype(float)

    # Ensure m1 >= m2 (swap if needed)
    swap = m2 > m1
    m1, m2 = np.where(swap, m2, m1), np.where(swap, m1, m2)
    chi1, chi2 = np.where(swap, chi2, chi1), np.where(swap, chi1, chi2)
    theta1, theta2 = np.where(swap, theta2, theta1), np.where(swap, theta1, theta2)

    q = m2 / m1
    eta = q / (1 + q)**2

    cost1, cost2 = np.cos(theta1), np.cos(theta2)
    sint1, sint2 = np.sqrt(1 - cost1**2), np.sqrt(1 - cost2**2)

    chi1_para, chi2_para = chi1 * cost1, chi2 * cost2
    chi1_perp, chi2_perp = chi1 * sint1, chi2 * sint2

    chi_para = (chi1_para + q**2 * chi2_para) / (1 + q)**2

    chi1_chi2 = chi1_para * chi2_para + chi1_perp * chi2_perp * np.cos(dPhi)
    chi_squared = (chi1**2 + q**4 * chi2**2 + 2 * q**2 * chi1_chi2) / (1 + q)**4

    el = (2 * np.sqrt(3) + _t2 * eta + _t3 * eta**2
          + _s4 * (1 + q)**4 / (1 + q**2) * chi_squared
          + (_s5 * eta + _t0 + 2) * (1 + q)**2 / (1 + q**2) * chi_para)

    chi_f = np.sqrt(q**2 * el**2 / (1 + q)**4 + chi_squared
                    + 2 * q * el * chi_para / (1 + q)**2)
    chi_f = np.clip(chi_f, 0, 1)

    return chi_f[0] if chi_f.size == 1 else chi_f


def bbh_final_mass_precessing_GK2016(m1, m2, chi1, chi2, theta1, theta2, dPhi):
    """
    Merger remnant mass in solar masses.
    From Gerosa & Kesden (2016) and references therein.
    Vectorized version.

    Parameters:
    -----------
    m1, m2 : float or array
        Component masses in solar masses
    chi1, chi2 : float or array
        Dimensionless spin magnitudes
    theta1, theta2 : float or array
        Angles between orbital angular momentum and spin vectors (radians)
    dPhi : float or array
        Azimuthal angle difference between spins (radians)

    Returns:
    --------
    m_f : float or array
        Remnant mass in solar masses

    References:
    -----------
    Gerosa & Kesden (2016): https://arxiv.org/abs/1605.01067
    Barausse, Morozova & Rezzolla (2012): https://arxiv.org/abs/1206.3803
    """
    m1, m2 = np.atleast_1d(m1).astype(float), np.atleast_1d(m2).astype(float)
    chi1, chi2 = np.atleast_1d(chi1).astype(float), np.atleast_1d(chi2).astype(float)
    theta1, theta2 = np.atleast_1d(theta1).astype(float), np.atleast_1d(theta2).astype(float)
    dPhi = np.atleast_1d(dPhi).astype(float)

    # Ensure m1 >= m2 (swap if needed)
    swap = m2 > m1
    m1, m2 = np.where(swap, m2, m1), np.where(swap, m1, m2)
    chi1, chi2 = np.where(swap, chi2, chi1), np.where(swap, chi1, chi2)
    theta1, theta2 = np.where(swap, theta2, theta1), np.where(swap, theta1, theta2)

    q = m2 / m1
    eta = q / (1 + q)**2

    cost1, cost2 = np.cos(theta1), np.cos(theta2)

    chi1_para, chi2_para = chi1 * cost1, chi2 * cost2
    chi_para = (chi1_para + q**2 * chi2_para) / (1 + q)**2

    Z1 = 1 + (1 - chi_para**2)**(1/3) * ((1 + chi_para)**(1/3) + (1 - chi_para)**(1/3))
    Z2 = np.sqrt(3 * chi_para**2 + Z1**2)

    r_isco = 3 + Z2 - np.sign(chi_para) * np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))
    E_isco = np.sqrt(1 - 2 / 3 / r_isco)

    m_f = (m1 + m2) * (1 - eta * (1 - 4 * eta) * (1 - E_isco)
                       - 16 * eta**2 * (_p0 + 4 * _p1 * chi_para * (chi_para + 1)))

    return m_f[0] if m_f.size == 1 else m_f


def bbh_final_state_precessing_GK2016(m1, m2, chi1, chi2, theta1, theta2, dPhi):
    """
    Convenience wrapper returning both remnant mass and spin.

    Parameters:
    -----------
    m1, m2 : float or array
        Component masses in solar masses
    chi1, chi2 : float or array
        Dimensionless spin magnitudes
    theta1, theta2 : float or array
        Angles between orbital angular momentum and spin vectors (radians)
    dPhi : float or array
        Azimuthal angle difference between spins (radians)

    Returns:
    --------
    (M_f, chi_f) : tuple
        Remnant mass (solar masses) and dimensionless spin magnitude
    """
    M_f = bbh_final_mass_precessing_GK2016(m1, m2, chi1, chi2, theta1, theta2, dPhi)
    chi_f = bbh_final_spin_precessing_GK2016(m1, m2, chi1, chi2, theta1, theta2, dPhi)
    return M_f, chi_f
