#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: host_retention.py
#
#    Environment escape-speed distributions for retention analysis.
#    Provides escape-speed CDFs for common astrophysical environments
#    and a generic interface for computing environment-marginalised
#    retention fractions from kick samples.
#
#    Predefined environment presets:
#      GC:  log-normal v_esc with mu=1.5, sig=0.30 (base-10)
#           from Antonini & Rasio (2016), Mapelli et al. (2021)
#      NSC: log-normal v_esc with mu=2.2, sig=0.36 (base-10)
#           from Antonini & Rasio (2016), Mapelli et al. (2021)
#      EG:  uniform v_esc in [400, 2500] km/s (early-type galaxies)
#      DG:  uniform v_esc in [100, 250] km/s (dwarf galaxies)
#      MW:  Gaussian v_esc with mean=497, std=8 km/s
#           from Koppelman & Helmi (2021)
#      M31: Gaussian v_esc with mean=470, std=40 km/s
#           from Kafle et al. (2018)
#
#    References:
#      Antonini & Rasio (2016): https://arxiv.org/abs/1606.04889
#      Mapelli et al. (2021): https://arxiv.org/abs/2106.07179
#      Koppelman & Helmi (2021): https://arxiv.org/abs/2006.16283
#      Kafle et al. (2018): https://doi.org/10.1093/mnras/sty082
#
#    AUTHOR: Tousif Islam
#    CREATED: 06-06-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from scipy.stats import lognorm, norm, gaussian_kde
from scipy.integrate import trapezoid

GC_ENVIRONMENT = {'kind': 'lognormal', 'mu': 1.5, 'sig': 0.30, 'label': 'GCs'}
NSC_ENVIRONMENT = {'kind': 'lognormal', 'mu': 2.2, 'sig': 0.36, 'label': 'NSCs'}
EG_ENVIRONMENT = {'kind': 'uniform', 'vmin': 400, 'vmax': 2500, 'label': 'EGs'}
DG_ENVIRONMENT = {'kind': 'uniform', 'vmin': 100, 'vmax': 250, 'label': 'DGs'}
MW_ENVIRONMENT = {'kind': 'gaussian', 'mean': 497, 'std': 8, 'label': 'MW'}
M31_ENVIRONMENT = {'kind': 'gaussian', 'mean': 470, 'std': 40, 'label': 'M31'}

ENVIRONMENTS = {'GC': GC_ENVIRONMENT, 'NSC': NSC_ENVIRONMENT,
                'EG': EG_ENVIRONMENT, 'DG': DG_ENVIRONMENT,
                'MW': MW_ENVIRONMENT, 'M31': M31_ENVIRONMENT}


def sample_escape_speed(n_samples, environment, seed=None):
    """Draw escape velocity samples from an environment's distribution.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw.
    environment : dict
        Environment specification (see ``escape_speed_cdf``).
    seed : int or None
        Random seed.

    Returns
    -------
    v_esc : numpy array
        Escape velocity samples [km/s]
    """
    rng = np.random.default_rng(seed)
    kind = environment['kind']
    if kind == 'lognormal':
        s = environment['sig'] * np.log(10)
        scale = np.exp(environment['mu'] * np.log(10))
        return lognorm.rvs(s, scale=scale, size=n_samples, random_state=rng)
    elif kind == 'uniform':
        return rng.uniform(environment['vmin'], environment['vmax'], size=n_samples)
    elif kind == 'gaussian':
        return np.abs(rng.normal(environment['mean'], environment['std'], size=n_samples))
    else:
        raise ValueError(f"Unknown environment kind: {kind!r}")


def sample_multi_escape_speed(n_samples, environments=None, seed=None):
    """Draw escape velocity samples for multiple environments.

    Parameters
    ----------
    n_samples : int
        Number of samples per environment.
    environments : dict of {name: environment_dict}, optional
        Defaults to all predefined environments.
    seed : int or None
        Random seed.

    Returns
    -------
    dict of {name: numpy array of v_esc}
    """
    if environments is None:
        environments = ENVIRONMENTS
    rng = np.random.default_rng(seed)
    return {name: sample_escape_speed(n_samples, env,
            seed=int(rng.integers(0, 2**31)))
            for name, env in environments.items()}


def escape_speed_cdf(v, environment):
    """CDF of the escape-speed distribution for a given environment.

    Parameters
    ----------
    v : float or array
        Escape speed value(s) [km/s]
    environment : dict
        Environment specification with 'kind' key:

        - ``{'kind': 'lognormal', 'mu': float, 'sig': float}``
          mu and sig are base-10 log mean and std dev
          (v_esc ~ 10^N(mu, sig))
        - ``{'kind': 'uniform', 'vmin': float, 'vmax': float}``
        - ``{'kind': 'gaussian', 'mean': float, 'std': float}``

    Returns
    -------
    cdf : float or array
    """
    kind = environment['kind']
    if kind == 'lognormal':
        return lognorm.cdf(v, s=environment['sig'] * np.log(10),
                           scale=np.exp(environment['mu'] * np.log(10)))
    elif kind == 'uniform':
        return np.clip(
            (np.asarray(v, dtype=float) - environment['vmin'])
            / (environment['vmax'] - environment['vmin']),
            0.0, 1.0)
    elif kind == 'gaussian':
        return norm.cdf(v, loc=environment['mean'], scale=environment['std'])
    else:
        raise ValueError(f"Unknown environment kind: {kind!r}")


def compute_environment_retention(v_kick, environment):
    """Per-kick retention probability for a given environment.

    p_ret_i = 1 - CDF(v_kick_i)

    Parameters
    ----------
    v_kick : array
        Kick velocity samples [km/s]
    environment : dict
        Environment specification (see ``escape_speed_cdf``)

    Returns
    -------
    p_ret : numpy array
        Individual retention probability for each kick sample
    """
    p_ret = 1.0 - escape_speed_cdf(np.asarray(v_kick), environment)
    return p_ret


def compute_multi_environment_retention(v_kick, environments=None):
    """Per-kick retention probabilities across multiple environments.

    Parameters
    ----------
    v_kick : array
        Kick velocity samples [km/s]
    environments : dict of {name: environment_dict}, optional
        Defaults to all predefined environments (GC, NSC, EG, DG)

    Returns
    -------
    dict of {name: numpy array of p_ret}
    """
    if environments is None:
        environments = ENVIRONMENTS
    return {name: compute_environment_retention(v_kick, env)
            for name, env in environments.items()}


def retention_curve(v_kick, v_esc_array):
    """Retention probability as a function of escape speed.

    p_ret(v_esc) = P(v_kick < v_esc), the empirical CDF of kick samples
    evaluated on v_esc_array.

    Parameters
    ----------
    v_kick : array
        Kick velocity samples [km/s]
    v_esc_array : array
        Escape speed values at which to evaluate retention [km/s]

    Returns
    -------
    p_ret : numpy array
        Retention probability at each v_esc value
    """
    v_sorted = np.sort(np.asarray(v_kick))
    cdf = np.arange(1, len(v_sorted) + 1) / len(v_sorted)
    return np.interp(v_esc_array, v_sorted, cdf, left=0.0, right=1.0)


def compute_environment_cumulative_retention(v_kick, environment,
                                             method='kde', vmax=5000.0, ngrid=5000):
    """Integrated retention fraction P_ret for a given environment.

    P_ret = ∫ p_kick(v) [1 - CDF_esc(v)] dv

    Parameters
    ----------
    v_kick : array
        Kick velocity samples [km/s]
    environment : dict
        Environment specification (see ``escape_speed_cdf``)
    method : str
        'kde' (default): fit a KDE to the kick samples and integrate
        analytically. More accurate for posterior/MCMC samples.
        'mc': Monte Carlo mean (1/N) Σ p_ret_i. Exact when samples
        are iid draws from the kick distribution.
    vmax : float
        Upper integration limit for KDE method [km/s] (default: 5000).
    ngrid : int
        Number of grid points for KDE integration (default: 5000).

    Returns
    -------
    P_ret : float
        Integrated retention fraction
    """
    v_kick = np.asarray(v_kick)
    if method == 'kde':
        kde = gaussian_kde(v_kick)
        v_grid = np.linspace(0.0, vmax, ngrid)
        p_kick = kde(v_grid)
        survival = 1.0 - escape_speed_cdf(v_grid, environment)
        P_ret = float(trapezoid(p_kick * survival, v_grid))
    elif method == 'mc':
        p_ret = compute_environment_retention(v_kick, environment)
        P_ret = float(np.mean(p_ret))
    else:
        raise ValueError(f"Unknown method: '{method}'. Choose 'kde' or 'mc'.")
    return P_ret


def compute_multi_environment_cumulative_retention(v_kick, environments=None,
                                                   method='kde', **kwargs):
    """Integrated retention fraction P_ret across multiple environments.

    Parameters
    ----------
    v_kick : array
        Kick velocity samples [km/s]
    environments : dict of {name: environment_dict}, optional
        Defaults to all predefined environments.
    method : str
        'kde' (default) or 'mc'. See ``compute_environment_cumulative_retention``.
    **kwargs
        Additional keyword arguments passed to
        ``compute_environment_cumulative_retention`` (vmax, ngrid).

    Returns
    -------
    dict of {name: P_ret}
    """
    if environments is None:
        environments = ENVIRONMENTS
    return {name: compute_environment_cumulative_retention(v_kick, env,
            method=method, **kwargs)
            for name, env in environments.items()}
