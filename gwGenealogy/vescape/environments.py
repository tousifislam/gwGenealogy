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
#    Users can define custom environments as dictionaries:
#        {'kind': 'lognormal', 'mu': ..., 'sig': ...}
#        {'kind': 'uniform',   'vmin': ..., 'vmax': ...}
#        {'kind': 'gaussian',  'mean': ..., 'std': ...}
#
#    AUTHOR: Tousif Islam
#    CREATED: 06-06-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from scipy.stats import lognorm, norm


# ===========================================================================
#  Predefined environment presets
# ===========================================================================

GC_ENVIRONMENT = {'kind': 'lognormal', 'mu': 1.5, 'sig': 0.30, 'label': 'GCs'}
NSC_ENVIRONMENT = {'kind': 'lognormal', 'mu': 2.2, 'sig': 0.36, 'label': 'NSCs'}
EG_ENVIRONMENT = {'kind': 'uniform', 'vmin': 400, 'vmax': 2500, 'label': 'EGs'}
DG_ENVIRONMENT = {'kind': 'uniform', 'vmin': 100, 'vmax': 250, 'label': 'DGs'}

ENVIRONMENTS = {
    'GC': GC_ENVIRONMENT,
    'NSC': NSC_ENVIRONMENT,
    'EG': EG_ENVIRONMENT,
    'DG': DG_ENVIRONMENT,
}


# ===========================================================================
#  Escape-speed CDF
# ===========================================================================

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


# ===========================================================================
#  Environment-marginalised retention
# ===========================================================================

def compute_environment_retention(v_kick, environment):
    """Mean retention fraction marginalised over the environment's v_esc distribution.

    F_ret = <P(v_esc > v_kick)> = <1 - CDF(v_kick)>

    Parameters
    ----------
    v_kick : array
        Kick velocity samples [km/s]
    environment : dict
        Environment specification (see ``escape_speed_cdf``)

    Returns
    -------
    F_ret : float
        Mean retention fraction
    """
    return float(np.mean(1.0 - escape_speed_cdf(np.asarray(v_kick), environment)))


def compute_multi_environment_retention(v_kick, environments=None):
    """Retention fraction across multiple environments.

    Parameters
    ----------
    v_kick : array
        Kick velocity samples [km/s]
    environments : dict of {name: environment_dict}, optional
        Defaults to all predefined environments (GC, NSC, EG, DG)

    Returns
    -------
    dict of {name: F_ret}
    """
    if environments is None:
        environments = ENVIRONMENTS
    return {name: compute_environment_retention(v_kick, env)
            for name, env in environments.items()}
