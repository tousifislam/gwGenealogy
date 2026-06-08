#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: plummer.py
#
#    Plummer sphere cluster model and merger retention analysis.
#
#    PlummerCluster class: initialize with (Mcl, r_h, cluster_type) to get
#    all structural quantities (scale radius, escape speed, core speed,
#    tidal radius, potential, density, velocity dispersion). Then call
#    merger_analysis(v_kick, M_bh) to compute per-kick retention, apocentre,
#    dynamical friction time, P_ret, P_core, P_hier.
#
#    Standalone functions are also available for use outside the class.
#
#    Cluster type presets (epsilon, tau_gyr for hierarchical mergers):
#      GC:  epsilon=0.2, tau_gyr=0.3
#      NSC: epsilon=0.4, tau_gyr=0.5
#
#    AUTHOR: Tousif Islam
#    CREATED: 06-06-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from scipy.special import erf

G_PC = 4.302e-3
PC_KMS_TO_MYR = 0.978

_CLUSTER_TYPE_DEFAULTS = {
    'GC':  {'epsilon': 0.2, 'tau_gyr': 0.3, 'rt_over_rh': 5.0},
    'NSC': {'epsilon': 0.4, 'tau_gyr': 0.5, 'rt_over_rh': 5.0},
}


class PlummerCluster:
    """Plummer sphere cluster model with merger retention analysis.

    Parameters
    ----------
    Mcl : float
        Cluster mass [Msun]
    r_h : float
        Half-mass radius [pc]
    cluster_type : str, optional
        'GC' or 'NSC'. Sets epsilon, tau_gyr, rt_over_rh defaults.
    epsilon : float, optional
        Repeat-merger efficiency. Overrides cluster_type default.
    tau_gyr : float, optional
        Repeat-merger timescale [Gyr]. Overrides cluster_type default.
    rt_over_rh : float, optional
        Tidal radius in units of r_h. Overrides cluster_type default.
    ln_lambda : float
        Coulomb logarithm (default: 2.5)
    """

    def __init__(self, Mcl, r_h, cluster_type=None,
                 epsilon=None, tau_gyr=None, rt_over_rh=None, ln_lambda=2.5):
        self.Mcl = float(Mcl)
        self.r_h = float(r_h)
        self.ln_lambda = ln_lambda

        defaults = _CLUSTER_TYPE_DEFAULTS.get(cluster_type, {}) if cluster_type else {}
        self.cluster_type = cluster_type
        self.epsilon = epsilon if epsilon is not None else defaults.get('epsilon', 0.2)
        self.tau_gyr = tau_gyr if tau_gyr is not None else defaults.get('tau_gyr', 0.3)
        rt_rh = rt_over_rh if rt_over_rh is not None else defaults.get('rt_over_rh', 5.0)

        self.a = plummer_scale_radius(self.r_h)
        self.r_c = self.a
        self.v_esc = plummer_escape_speed(self.Mcl, self.a)
        self.v_core = plummer_core_speed(self.Mcl, self.a)
        self.r_t = plummer_tidal_radius(self.r_h, rt_rh)

    def potential(self, r):
        """Gravitational potential at distance r [pc]. Returns (km/s)^2."""
        return plummer_potential(r, self.Mcl, self.a)

    def density(self, r):
        """Density at distance r [pc]. Returns Msun/pc^3."""
        return plummer_density(r, self.Mcl, self.a)

    def sigma(self, r):
        """1-D velocity dispersion at distance r [pc]. Returns km/s."""
        return plummer_velocity_dispersion(r, self.Mcl, self.a)

    def merger_analysis(self, v_kick, M_bh):
        """Run full merger retention analysis for a set of kicks.

        Parameters
        ----------
        v_kick : array
            Kick velocity samples [km/s]
        M_bh : array (same length as v_kick)
            Remnant BH masses [Msun]

        Sets attributes
        ---------------
        v_kick : array
        M_bh : array
        retained : boolean array
        r_max : array (NaN for unbound kicks)
        t_df : array (NaN for ejected kicks) [Myr]
        P_ret : float
        P_core : float
        P_hier : float
        """
        self.v_kick = np.asarray(v_kick, dtype=float)
        self.M_bh = np.asarray(M_bh, dtype=float)

        self.retained = retained_mask(self.v_kick, self.v_esc, self.a, self.r_t)

        self.r_max = np.full_like(self.v_kick, np.nan)
        bound = self.v_kick < self.v_esc
        if bound.any():
            self.r_max[bound] = plummer_apocentre(self.v_kick[bound], self.v_esc, self.a)

        self.t_df = np.full_like(self.v_kick, np.nan)
        if self.retained.any():
            M_ret = self.M_bh[self.retained] if self.M_bh.ndim > 0 else self.M_bh
            self.t_df[self.retained] = dynamical_friction_time(
                self.v_kick[self.retained], self.v_esc, self.Mcl, self.a,
                M_ret, self.ln_lambda)

        self.P_ret = float(np.mean(self.retained))
        self.P_core = float(np.mean(self.v_kick < self.v_core))

        if self.retained.any():
            tdf_gyr = self.t_df[self.retained] / 1000.0
            self.P_hier = float(self.epsilon * np.sum(np.exp(-tdf_gyr / self.tau_gyr))
                                / len(self.v_kick))
        else:
            self.P_hier = 0.0

    def __repr__(self):
        s = (f"PlummerCluster(Mcl={self.Mcl:.2e}, r_h={self.r_h:.2f}, "
             f"a={self.a:.2f}, v_esc={self.v_esc:.1f} km/s, "
             f"v_core={self.v_core:.1f} km/s, r_t={self.r_t:.1f} pc")
        if self.cluster_type:
            s += f", type={self.cluster_type}"
        if hasattr(self, 'P_ret'):
            s += f", P_ret={self.P_ret:.4f}, P_core={self.P_core:.4f}, P_hier={self.P_hier:.4f}"
        return s + ")"


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


def chandrasekhar_F(X):
    """Chandrasekhar velocity term F(X) = erf(X) - (2X/sqrt(pi)) exp(-X^2).

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
