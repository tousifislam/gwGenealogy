#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: plummer.py
#
#    Plummer sphere dynamics: potential, density, escape speed, apocentre,
#    Chandrasekhar dynamical friction, and retention/hierarchical-merger
#    probability.  All functions take physical parameters — no hard-coded
#    environment assumptions.  Use the preset dicts (GC_POPULATION_DEFAULTS,
#    NSC_POPULATION_DEFAULTS, etc.) for common environments, or pass your own.
#
#    AUTHOR: Tousif Islam
#    CREATED: 06-06-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from scipy.special import erf

# ---- Physical constants (pc-km/s-Msun units) ----

G_PC = 4.302e-3
PC_KMS_TO_MYR = 0.978


# ---- Common environment presets ----

GC_POPULATION_DEFAULTS = {'Mcl_range': (1e4, 1e6), 'rh_range': (0.5, 50.0)}
NSC_POPULATION_DEFAULTS = {'Mcl_range': (1e6, 1e8), 'rh_range': (1.0, 20.0)}

GC_HIER_DEFAULTS = {'epsilon': 0.2, 'tau_gyr': 0.3}
NSC_HIER_DEFAULTS = {'epsilon': 0.4, 'tau_gyr': 0.5}


# ===========================================================================
#  Plummer sphere model
# ===========================================================================

def plummer_scale_radius(r_h):
    """Convert half-mass radius to Plummer scale radius: a = r_h / 1.305.

    Parameters
    ----------
    r_h : float or array
        Half-mass radius [pc]

    Returns
    -------
    a : float or array
        Scale radius [pc]
    """
    return np.asarray(r_h, dtype=float) / 1.305


def plummer_potential(r, Mcl, a):
    """Plummer gravitational potential Phi(r).

    Parameters
    ----------
    r : float or array
        Radial distance [pc]
    Mcl : float
        Cluster mass [Msun]
    a : float
        Scale radius [pc]

    Returns
    -------
    Phi : float or array
        Potential [(km/s)^2]
    """
    return -G_PC * Mcl / np.sqrt(np.asarray(r, dtype=float)**2 + a**2)


def plummer_density(r, Mcl, a):
    """Plummer density profile rho(r).

    Parameters
    ----------
    r : float or array
        Radial distance [pc]
    Mcl : float
        Cluster mass [Msun]
    a : float
        Scale radius [pc]

    Returns
    -------
    rho : float or array
        Density [Msun / pc^3]
    """
    return (3.0 * Mcl) / (4.0 * np.pi * a**3) * (1.0 + (np.asarray(r, dtype=float) / a)**2)**(-2.5)


def plummer_velocity_dispersion(r, Mcl, a):
    """Isotropic 1-D velocity dispersion sigma(r) for a Plummer sphere.

    Parameters
    ----------
    r : float or array
        Radial distance [pc]
    Mcl : float
        Cluster mass [Msun]
    a : float
        Scale radius [pc]

    Returns
    -------
    sigma : float or array
        Velocity dispersion [km/s]
    """
    return np.sqrt(G_PC * Mcl / (6.0 * a) * (1.0 + (np.asarray(r, dtype=float) / a)**2)**(-0.5))


def plummer_escape_speed(Mcl, a):
    """Central escape speed v_esc = sqrt(2 G Mcl / a).

    Parameters
    ----------
    Mcl : float or array
        Cluster mass [Msun]
    a : float or array
        Scale radius [pc]

    Returns
    -------
    v_esc : float or array
        Escape speed [km/s]
    """
    return np.sqrt(2.0 * G_PC * np.asarray(Mcl, dtype=float) / np.asarray(a, dtype=float))


def plummer_core_speed(Mcl, a):
    """Core-crossing speed v_c ~ 0.54 v_esc.

    The minimum kick to reach the scale radius r = a from the center.

    Parameters
    ----------
    Mcl : float or array
        Cluster mass [Msun]
    a : float or array
        Scale radius [pc]

    Returns
    -------
    v_c : float or array
        Core-crossing speed [km/s]
    """
    return plummer_escape_speed(Mcl, a) * np.sqrt(1.0 - 1.0 / np.sqrt(2.0))


def plummer_tidal_radius(r_h, rt_over_rh=5.0):
    """Tidal truncation radius r_t = rt_over_rh * r_h.

    Parameters
    ----------
    r_h : float or array
        Half-mass radius [pc]
    rt_over_rh : float
        Tidal radius in units of r_h (default: 5.0)

    Returns
    -------
    r_t : float or array
        Tidal radius [pc]
    """
    return rt_over_rh * np.asarray(r_h, dtype=float)


def plummer_apocentre(v_kick, v_esc, a):
    """Radial-orbit apocentre r_max in a Plummer potential.

    Parameters
    ----------
    v_kick : float or array
        Kick velocity (must be < v_esc for bound orbits) [km/s]
    v_esc : float
        Central escape speed [km/s]
    a : float
        Scale radius [pc]

    Returns
    -------
    r_max : float or array
        Apocentre distance [pc]
    """
    u = np.asarray(v_kick, dtype=float) / v_esc
    return a * np.sqrt((1.0 - u**2)**(-2) - 1.0)


# ===========================================================================
#  Cluster population sampling
# ===========================================================================

def sample_cluster_population(n_clusters, Mcl_range=(1e4, 1e6),
                              rh_range=(0.5, 50.0), rt_over_rh=5.0, seed=None):
    """Sample a population of Plummer clusters with log-uniform priors.

    Parameters
    ----------
    n_clusters : int
        Number of clusters
    Mcl_range : tuple (float, float)
        (min, max) cluster mass [Msun], sampled log-uniformly.
        Typical: (1e4, 1e6) for GCs, (1e6, 1e8) for NSCs.
    rh_range : tuple (float, float)
        (min, max) half-mass radius [pc], sampled log-uniformly.
        Typical: (0.5, 50) for GCs, (1, 20) for NSCs.
    rt_over_rh : float
        Tidal radius in units of r_h (default: 5.0)
    seed : int or None
        Random seed

    Returns
    -------
    dict with keys:
        'Mcl', 'rh', 'a', 'vesc', 'vc', 'r_t' (arrays),
        'n_clusters' (int)
    """
    rng = np.random.default_rng(seed)
    Mcl = 10.0**rng.uniform(*np.log10(Mcl_range), n_clusters)
    rh = 10.0**rng.uniform(*np.log10(rh_range), n_clusters)
    a = plummer_scale_radius(rh)
    return {
        'Mcl': Mcl,
        'rh': rh,
        'a': a,
        'vesc': plummer_escape_speed(Mcl, a),
        'vc': plummer_core_speed(Mcl, a),
        'r_t': plummer_tidal_radius(rh, rt_over_rh),
        'n_clusters': n_clusters,
    }


# ===========================================================================
#  Chandrasekhar dynamical friction
# ===========================================================================

def chandrasekhar_F(X):
    """Chandrasekhar velocity term F(X) = erf(X) - (2X/sqrt(pi)) exp(-X^2).

    Uses a series expansion for X < 0.2 to avoid numerical cancellation.

    Parameters
    ----------
    X : float or array
        Velocity ratio v / (sqrt(2) sigma)

    Returns
    -------
    F : float or array
    """
    X = np.asarray(X, dtype=float)
    series = (4.0 / (3.0 * np.sqrt(np.pi))) * X**3 * (1.0 - 0.6 * X**2 + (3.0 / 14.0) * X**4)
    direct = erf(X) - (2.0 * X / np.sqrt(np.pi)) * np.exp(-X**2)
    return np.where(X < 0.2, series, direct)


def orbit_shape_factor(u, n_xi=400):
    """Dimensionless orbit-shape factor R(u) for a radial Plummer orbit.

    The dynamical-friction return time factorizes as t_DF = K * R(u),
    where K depends on cluster/BH parameters and R depends only on
    u = v_kick / v_esc.

    Parameters
    ----------
    u : float
        Ratio v_kick / v_esc (0 < u < 1)
    n_xi : int
        Number of quadrature points (default: 400)

    Returns
    -------
    R : float
        Dimensionless orbit-shape factor
    """
    s_max = np.sqrt((1.0 - u**2)**(-2) - 1.0)
    xi = (np.arange(n_xi) + 0.5) / n_xi
    s = s_max * (2.0 * xi - xi**2)
    dsdxi = 2.0 * s_max * (1.0 - xi)
    g = 1.0 + s**2
    w = np.sqrt(np.clip(u**2 - 1.0 + g**(-0.5), 1e-300, None))
    X = np.sqrt(6.0) * w * g**0.25
    h = w**3 * g**2.5 / chandrasekhar_F(X)
    integrand = dsdxi / w
    return (np.sum(integrand) / n_xi) / (np.sum(integrand / h) / n_xi)


# Lazy-loaded interpolation grid for orbit_shape_factor
_u_grid = None
_R_grid = None


def _ensure_orbit_grid():
    """Build the interpolation grid on first use."""
    global _u_grid, _R_grid
    if _u_grid is None:
        _u_grid = np.linspace(1e-3, 1.0 - 1e-4, 400)
        _R_grid = np.array([orbit_shape_factor(u) for u in _u_grid])


def dynamical_friction_time(v_kick, v_esc, Mcl, a, M_bh, ln_lambda=2.5):
    """Orbit-averaged Chandrasekhar dynamical-friction return time.

    Parameters
    ----------
    v_kick : float or array
        Kick velocities of RETAINED remnants (v_kick < v_esc) [km/s]
    v_esc : float
        Central escape speed [km/s]
    Mcl : float
        Cluster mass [Msun]
    a : float
        Plummer scale radius [pc]
    M_bh : float or array (matching v_kick)
        Remnant BH mass(es) [Msun]
    ln_lambda : float
        Coulomb logarithm (default: 2.5)

    Returns
    -------
    t_DF : float or array
        Return time [Myr]
    """
    _ensure_orbit_grid()
    u = np.asarray(v_kick, dtype=float) / v_esc
    R = np.interp(u, _u_grid, _R_grid)
    K = (2.0**1.5 * np.sqrt(Mcl * a**3 / G_PC)
         / (3.0 * ln_lambda * np.asarray(M_bh, dtype=float)) * PC_KMS_TO_MYR)
    return K * R


# ===========================================================================
#  Retention analysis
# ===========================================================================

def retained_mask(v_kick, v_esc, a, r_t):
    """Boolean mask: bound (v_kick < v_esc) AND apocentre within tidal radius.

    Parameters
    ----------
    v_kick : array
        Kick velocities [km/s]
    v_esc : float
        Central escape speed [km/s]
    a : float
        Plummer scale radius [pc]
    r_t : float
        Tidal truncation radius [pc]

    Returns
    -------
    mask : boolean array (same shape as v_kick)
    """
    v_kick = np.asarray(v_kick, dtype=float)
    bound = v_kick < v_esc
    result = bound.copy()
    if bound.any():
        rmax = plummer_apocentre(v_kick[bound], v_esc, a)
        result[bound] = rmax <= r_t
    return result


def compute_p_retention(v_kick, v_esc, a, r_t):
    """Retention probability: fraction of kicks retained within tidal radius.

    Parameters
    ----------
    v_kick : array
        Kick velocities [km/s]
    v_esc : float
        Central escape speed [km/s]
    a : float
        Plummer scale radius [pc]
    r_t : float
        Tidal truncation radius [pc]

    Returns
    -------
    P_ret : float
    """
    return float(np.mean(retained_mask(v_kick, v_esc, a, r_t)))


def compute_p_core(v_kick, v_esc, Mcl, a):
    """Prompt core-occupancy probability: fraction with v_kick < v_core.

    Parameters
    ----------
    v_kick : array
        Kick velocities [km/s]
    v_esc : float
        Central escape speed [km/s]
    Mcl : float
        Cluster mass [Msun]
    a : float
        Plummer scale radius [pc]

    Returns
    -------
    P_core : float
    """
    vc = plummer_core_speed(Mcl, a)
    return float(np.mean(np.asarray(v_kick) < vc))


def compute_p_hier(v_kick, M_bh, v_esc, Mcl, a, r_t,
                   epsilon, tau_gyr, ln_lambda=2.5):
    """Hierarchical-merger probability weighted by dynamical-friction return.

    P_hier = (epsilon / N) * sum_{retained} exp(-t_DF / tau)

    Parameters
    ----------
    v_kick : array
        Kick velocities [km/s]
    M_bh : float or array (same length as v_kick)
        Remnant BH mass(es) [Msun]
    v_esc : float
        Central escape speed [km/s]
    Mcl : float
        Cluster mass [Msun]
    a : float
        Plummer scale radius [pc]
    r_t : float
        Tidal truncation radius [pc]
    epsilon : float
        Repeat-merger efficiency (e.g., 0.2 for GCs, 0.4 for NSCs)
    tau_gyr : float
        Repeat-merger timescale [Gyr] (e.g., 0.3 for GCs, 0.5 for NSCs)
    ln_lambda : float
        Coulomb logarithm (default: 2.5)

    Returns
    -------
    P_hier : float
    """
    v_kick = np.asarray(v_kick, dtype=float)
    M_bh = np.asarray(M_bh, dtype=float)
    mask = retained_mask(v_kick, v_esc, a, r_t)
    if not mask.any():
        return 0.0
    M_retained = M_bh[mask] if M_bh.ndim > 0 else M_bh
    tdf_myr = dynamical_friction_time(v_kick[mask], v_esc, Mcl, a, M_retained, ln_lambda)
    tdf_gyr = tdf_myr / 1000.0
    return float(epsilon * np.sum(np.exp(-tdf_gyr / tau_gyr)) / len(v_kick))


def retention_curve(v_kick, v_esc_values):
    """Retention probability as a function of escape speed: P_ret(v_esc).

    Parameters
    ----------
    v_kick : array
        Kick velocity samples [km/s]
    v_esc_values : array
        Escape speed grid [km/s]

    Returns
    -------
    P_ret : array (same shape as v_esc_values)
    """
    v_kick = np.asarray(v_kick)
    v_esc_values = np.asarray(v_esc_values)
    return np.array([float(np.mean(v_kick < ve)) for ve in v_esc_values])


def population_retention(v_kick, clusters, M_bh=None,
                         epsilon=None, tau_gyr=None, ln_lambda=2.5):
    """Retention, core-occupancy, and P_hier across a cluster population.

    Parameters
    ----------
    v_kick : array
        Kick velocity samples [km/s]
    clusters : dict
        Output of ``sample_cluster_population``
    M_bh : float or array (same length as v_kick), optional
        Remnant BH mass(es) [Msun].  Required for P_hier.
    epsilon : float, optional
        Repeat-merger efficiency.  Required for P_hier.
    tau_gyr : float, optional
        Repeat-merger timescale [Gyr].  Required for P_hier.
    ln_lambda : float
        Coulomb logarithm (default: 2.5)

    Returns
    -------
    dict with keys:
        'P_ret'  : array of shape (n_clusters,)
        'P_core' : array of shape (n_clusters,)
        'P_hier' : array of shape (n_clusters,) — only if M_bh, epsilon,
                   and tau_gyr are all provided
    """
    n = clusters['n_clusters']
    P_ret = np.zeros(n)
    P_core = np.zeros(n)
    compute_hier = (M_bh is not None and epsilon is not None and tau_gyr is not None)
    P_hier = np.zeros(n) if compute_hier else None

    for i in range(n):
        vi, ai, ri = clusters['vesc'][i], clusters['a'][i], clusters['r_t'][i]
        Mi = clusters['Mcl'][i]
        P_ret[i] = compute_p_retention(v_kick, vi, ai, ri)
        P_core[i] = compute_p_core(v_kick, vi, Mi, ai)
        if compute_hier:
            P_hier[i] = compute_p_hier(v_kick, M_bh, vi, Mi, ai, ri,
                                       epsilon, tau_gyr, ln_lambda)

    result = {'P_ret': P_ret, 'P_core': P_core}
    if P_hier is not None:
        result['P_hier'] = P_hier
    return result
