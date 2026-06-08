#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: environments.py
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
#
#    References:
#      Antonini & Rasio (2016): https://arxiv.org/abs/1606.04889
#      Mapelli et al. (2021): https://arxiv.org/abs/2106.07179
#
#    AUTHOR: Tousif Islam
#    CREATED: 06-06-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from scipy.stats import lognorm, norm

GC_ENVIRONMENT = {'kind': 'lognormal', 'mu': 1.5, 'sig': 0.30, 'label': 'GCs'}
NSC_ENVIRONMENT = {'kind': 'lognormal', 'mu': 2.2, 'sig': 0.36, 'label': 'NSCs'}
EG_ENVIRONMENT = {'kind': 'uniform', 'vmin': 400, 'vmax': 2500, 'label': 'EGs'}
DG_ENVIRONMENT = {'kind': 'uniform', 'vmin': 100, 'vmax': 250, 'label': 'DGs'}

ENVIRONMENTS = {'GC': GC_ENVIRONMENT, 'NSC': NSC_ENVIRONMENT, 
                'EG': EG_ENVIRONMENT, 'DG': DG_ENVIRONMENT}


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


def compute_environment_cumulative_retention(v_kick, environment):
    """Integrated retention fraction P_ret for a given environment.

    P_ret = ∫ p_ret(v) f(v) dv  ≈  (1/N) Σᵢ p_ret(vᵢ)
    where vᵢ ~ f(v) are kick samples drawn from the physical kick distribution.
    The MC mean is exact because f(v) weighting enters at the sampling stage.

    Parameters
    ----------
    v_kick : array
        Kick velocity samples [km/s]
    environment : dict
        Environment specification (see ``escape_speed_cdf``)

    Returns
    -------
    P_ret : float
        Integrated retention fraction
    """
    p_ret = compute_environment_retention(v_kick, environment)
    P_ret = float(np.mean(p_ret))
    return P_ret


def compute_multi_environment_cumulative_retention(v_kick, environments=None):
    """Integrated retention fraction P_ret across multiple environments.

    Parameters
    ----------
    v_kick : array
        Kick velocity samples [km/s]
    environments : dict of {name: environment_dict}, optional
        Defaults to all predefined environments (GC, NSC, EG, DG)

    Returns
    -------
    dict of {name: P_ret}
    """
    if environments is None:
        environments = ENVIRONMENTS
    return {name: compute_environment_cumulative_retention(v_kick, env)
            for name, env in environments.items()}
