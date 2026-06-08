#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==================================================================================
#
#    FILE: smbbh.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 06-07-2026
#    DESCRIPTION: Supermassive BBH binary parameter sampling
#
#    Three accretion-type models for SMBH binary progenitors:
#
#    Total mass:
#      M_total ~ LogUniform(1e5, 1e11) Msun
#      m1 = M_total / (1 + small_q),  m2 = M_total * small_q / (1 + small_q)
#
#    Mass ratio:
#      agnostic: small_q = U(0.1, 1)
#      hot/cold/dry: p(small_q) ∝ small_q^{-0.3} (1 - small_q)  =>  Beta(0.7, 2)
#      where small_q = m2/m1 in (0, 1], q = m1/m2 = 1/small_q >= 1
#
#    Spin magnitude (iid for chi1, chi2):
#      p(chi) = Beta(chi; a, b)
#        agnostic: Uniform on [0, 1]
#        hot:     a=3.212,   b=1.563    (high spins, peaked ~0.7)
#        cold:    a=5.935,   b=1.856    (very high spins, peaked ~0.8)
#        dry:     a=10.5868, b=4.66884  (moderate, narrow, peaked ~0.7)
#
#    Spin tilt theta1, theta2:
#      agnostic: isotropic, theta = arccos(U(-1, 1))   (random orientations)
#      hot:     Beta(2.018, 5.244) on [0, 1] radian   (preferentially aligned)
#      cold:    Beta(2.544, 19.527) on [0, 1] radian   (strongly aligned)
#      dry:     isotropic, theta = arccos(U(-1, 1))    (random orientations)
#
#    Azimuthal angles phi1, phi2:
#      Uniform on [0, 2*pi]
#
#==================================================================================
__author__ = "Tousif Islam"

import numpy as np
from ..utils.distributions import sample_beta_1d, sample_uniform_1d, sample_loguniform_1d
from .bbh_spins import sample_spin_angles

# Beta distribution parameters for spin magnitude: p(chi) = Beta(chi; a, b)
# 'agnostic' uses uniform on [0, 1] instead
_SPIN_MAG_PARAMS = {"agnostic": None,
                    "hot":     (3.212,   1.563),
                    "cold":    (5.935,   1.856),
                    "dry":     (10.5868, 4.66884)}

# Beta distribution parameters for spin tilt on [0, 1] radian
# dry uses isotropic instead
_SPIN_TILT_PARAMS = {"hot":  (2.018, 5.244),
                     "cold": (2.544, 19.527)}


def sample_smbbh(n_samples, accretion="hot",
                 m_total_min=1e5, m_total_max=1e11, seed=None):
    """
    Sample supermassive BBH binary parameters for a given accretion model.

    Parameters
    ----------
    n_samples : int
        Number of binary systems to generate
    accretion : str
        Accretion type (default: 'hot'):
        - 'agnostic': uniform spins on [0,1], isotropic orientations
        - 'hot':  hot wet accretion (high spins, preferentially aligned)
        - 'cold': cold wet accretion (very high spins, strongly aligned)
        - 'dry':  dry merger (moderate spins, isotropic orientations)
    m_total_min : float
        Minimum total mass in Msun (default: 1e5)
    m_total_max : float
        Maximum total mass in Msun (default: 1e11)
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary with keys:
        - m_total:   total mass in Msun
        - mass_1:    primary mass in Msun (m1 >= m2)
        - mass_2:    secondary mass in Msun
        - q:         mass ratio m1/m2 >= 1
        - small_q:   mass ratio m2/m1 in (0, 1]
        - a1:  primary spin magnitude
        - a2:  secondary spin magnitude
        - theta1:   primary spin tilt angle (rad)
        - theta2:   secondary spin tilt angle (rad)
        - cos_theta1, cos_theta2: cosines of tilt angles
        - phi1:     primary azimuthal angle (rad)
        - phi2:     secondary azimuthal angle (rad)
    """
    accretion = accretion.lower()
    if accretion not in _SPIN_MAG_PARAMS:
        raise ValueError(
            f"Unknown accretion type: '{accretion}'. "
            f"Choose from {list(_SPIN_MAG_PARAMS.keys())}")

    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(6)]

    # Total mass: M_total ~ LogUniform(m_total_min, m_total_max)
    m_total = sample_loguniform_1d(n_samples, low=m_total_min, high=m_total_max,
                                   seed=seeds[5])

    # Mass ratio
    if accretion == "agnostic":
        # Uniform small_q in [0.1, 1]
        small_q = sample_uniform_1d(n_samples, low=0.1, high=1.0, seed=seeds[0])
    else:
        # p(small_q) ∝ small_q^{-0.3} (1 - small_q) => Beta(0.7, 2)
        small_q = sample_beta_1d(n_samples, a=0.7, b=2.0, seed=seeds[0])
    q = 1.0 / small_q

    # Component masses: m1 = M/(1+small_q), m2 = M*small_q/(1+small_q)
    mass_1 = m_total / (1.0 + small_q)
    mass_2 = m_total * small_q / (1.0 + small_q)

    # Spin magnitudes
    mag_params = _SPIN_MAG_PARAMS[accretion]
    if mag_params is None:
        # agnostic: uniform on [0, 1]
        a1 = sample_uniform_1d(n_samples, low=0, high=1, seed=seeds[1])
        a2 = sample_uniform_1d(n_samples, low=0, high=1, seed=seeds[2])
    else:
        # p(chi) = Beta(chi; a, b)
        a_mag, b_mag = mag_params
        a1 = sample_beta_1d(n_samples, a=a_mag, b=b_mag, seed=seeds[1])
        a2 = sample_beta_1d(n_samples, a=a_mag, b=b_mag, seed=seeds[2])

    # Spin tilts and azimuthal angles
    if accretion in ("dry", "agnostic"):
        theta1, theta2, phi1, phi2 = sample_spin_angles(n_samples, spin_angles='isotropic', 
                                                            seed=seeds[3])
    else:
        a_tilt, b_tilt = _SPIN_TILT_PARAMS[accretion]
        theta1, theta2, phi1, phi2 = sample_spin_angles(n_samples, spin_angles='beta',
                                                            tilt_beta_a=a_tilt, tilt_beta_b=b_tilt, 
                                                            seed=seeds[3])

    cos_theta1 = np.cos(theta1)
    cos_theta2 = np.cos(theta2)

    return {
        "m_total": m_total,
        "mass_1": mass_1,
        "mass_2": mass_2,
        "q": q,
        "small_q": small_q,
        "a1": a1,
        "a2": a2,
        "theta1": theta1,
        "theta2": theta2,
        "cos_theta1": cos_theta1,
        "cos_theta2": cos_theta2,
        "phi1": phi1,
        "phi2": phi2,
    }
