#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: star_clusters.py
#
#    Star cluster property sampling and escape velocity formulas.
#
#    Escape velocity formulas:
#      Mcl_rh_to_vescape: virial theorem, v_esc = 2 sqrt(0.4 G M / r_h)
#        Reference: https://arxiv.org/pdf/2210.10055, Equation 1
#      Mcl_rho_to_vescape: mass-density scaling relation,
#        v_esc = 40 km/s (M/1e5)^(1/3) (rho/1e5)^(1/6)
#        References: Georgiev et al. (2009a,b), Fragione & Silk (2020)
#
#    Cluster sampling (Mapelli et al. 2021, Sec 2.5):
#      Three cluster flavours — NSC, GC, YSC — each defined by
#      log-normal M_cluster and rho distributions, plus t_SC and f_bin.
#      Core density rho_c = 20 * rho.
#
#    References:
#      Mapelli et al. (2021): https://arxiv.org/abs/2106.07179
#      Georgiev et al. (2009a,b)
#      Fragione & Silk (2020)
#
#    AUTHOR: Tousif Islam
#    CREATED: 06-07-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np

G_SI = 6.67430e-11
M_SUN_KG = 1.98892e30
PC_TO_M = 3.08567758149137e16


def Mcl_rh_to_vescape(Mcl, r_h):
    """Escape velocity from cluster mass and half-mass radius (virial theorem).

    v_esc = 2 * sqrt(0.4 G M_cl / r_h)

    Parameters
    ----------
    Mcl : float or array
        Total cluster mass [Msun]
    r_h : float or array
        Half-mass radius [pc]

    Returns
    -------
    v_esc : float or array
        Escape velocity [km/s]

    Reference: https://arxiv.org/pdf/2210.10055, Equation 1
    """
    Mcl_kg = np.asarray(Mcl, dtype=float) * M_SUN_KG
    r_h_m = np.asarray(r_h, dtype=float) * PC_TO_M
    v_rms = np.sqrt(0.4 * G_SI * Mcl_kg / r_h_m)
    return 2.0 * v_rms / 1000.0


def Mcl_rho_to_vescape(Mcl, rho):
    """Escape velocity from cluster mass and half-mass density (scaling relation).

    v_esc = 40 km/s * (M_tot / 1e5 Msun)^(1/3) * (rho / 1e5 Msun pc^-3)^(1/6)

    Parameters
    ----------
    Mcl : float or array
        Total cluster mass [Msun]
    rho : float or array
        Density at the half-mass radius [Msun pc^-3]

    Returns
    -------
    v_esc : float or array
        Escape velocity [km/s]

    References: Georgiev et al. (2009a,b), Fragione & Silk (2020),
    https://arxiv.org/pdf/2103.05016 Equation 22
    """
    return 40.0 * (np.asarray(Mcl, dtype=float) / 1e5)**(1.0/3.0) * (np.asarray(rho, dtype=float) / 1e5)**(1.0/6.0)


_STAR_CLUSTER_PARAMS = {
    'NSC': {'log_M_mean': 6.18, 'log_rho_mean': 5.0, 't_SC': 13.6, 'f_bin': 0.01},
    'GC':  {'log_M_mean': 5.6,  'log_rho_mean': 3.7, 't_SC': 13.6, 'f_bin': 0.1},
    'YSC': {'log_M_mean': 4.3,  'log_rho_mean': 3.3, 't_SC': 1.0,  'f_bin': 1.0},
}


def sample_star_clusters_mapelli2021(n_samples, cluster_type='GC',
                                     sigma_M=0.4, sigma_rho=0.4, seed=None):
    """Sample star cluster properties following Mapelli et al. (2021) Sec 2.5.

    Parameters
    ----------
    n_samples : int
        Number of clusters to draw.
    cluster_type : str
        'NSC', 'GC', or 'YSC'.
    sigma_M : float
        Std dev of log10(M_cluster/Msun) distribution (default: 0.4).
    sigma_rho : float
        Std dev of log10(rho/(Msun pc^-3)) distribution (default: 0.4).
    seed : int or None
        Random seed.

    Returns
    -------
    dict
        M_cluster : total mass [Msun]
        rho : half-mass density [Msun pc^-3]
        rho_c : core density = 20*rho [Msun pc^-3]
        v_esc : escape velocity [km/s]
        t_SC : cluster lifetime [Gyr]
        f_bin : binary fraction
    """
    cluster_type = cluster_type.upper()
    if cluster_type not in _STAR_CLUSTER_PARAMS:
        raise ValueError(f"Unknown cluster_type: '{cluster_type}'. "
                         f"Choose from {list(_STAR_CLUSTER_PARAMS.keys())}")

    params = _STAR_CLUSTER_PARAMS[cluster_type]
    rng = np.random.default_rng(seed)

    log_M = rng.normal(params['log_M_mean'], sigma_M, size=n_samples)
    log_rho = rng.normal(params['log_rho_mean'], sigma_rho, size=n_samples)

    M_cluster = 10.0**log_M
    rho = 10.0**log_rho
    rho_c = 20.0 * rho

    v_esc = Mcl_rho_to_vescape(M_cluster, rho)

    return {
        'M_cluster': M_cluster,
        'rho': rho,
        'rho_c': rho_c,
        'v_esc': v_esc,
        't_SC': params['t_SC'],
        'f_bin': params['f_bin'],
    }


def sample_star_clusters(n_samples, M_cluster_min=1e3, M_cluster_max=1e8,
                              r_h_min=0.1, r_h_max=100.0,
                              Z_min=1e-4, Z_max=0.02, seed=None):
    """Sample generic star cluster properties from log-uniform priors.

    Parameters
    ----------
    n_samples : int
        Number of clusters to draw.
    M_cluster_min : float
        Minimum cluster mass [Msun] (default: 1e3).
    M_cluster_max : float
        Maximum cluster mass [Msun] (default: 1e8).
    r_h_min : float
        Minimum half-mass radius [pc] (default: 0.1).
    r_h_max : float
        Maximum half-mass radius [pc] (default: 100).
    Z_min : float
        Minimum metallicity (default: 1e-4).
    Z_max : float
        Maximum metallicity (default: 0.02).
    seed : int or None
        Random seed.

    Returns
    -------
    dict
        M_cluster : cluster mass [Msun]
        r_h : half-mass radius [pc]
        Z : metallicity
        v_esc : escape velocity [km/s]
    """
    rng = np.random.default_rng(seed)

    log_M = rng.uniform(np.log10(M_cluster_min), np.log10(M_cluster_max), size=n_samples)
    log_rh = rng.uniform(np.log10(r_h_min), np.log10(r_h_max), size=n_samples)
    log_Z = rng.uniform(np.log10(Z_min), np.log10(Z_max), size=n_samples)

    M_cluster = 10.0**log_M
    r_h = 10.0**log_rh
    Z = 10.0**log_Z

    v_esc = Mcl_rh_to_vescape(M_cluster, r_h)

    return {
        'M_cluster': M_cluster,
        'r_h': r_h,
        'Z': Z,
        'v_esc': v_esc,
    }
