#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: natal_kick.py
#
#    Supernova natal kick prescriptions for compact objects.
#
#    Based on Equation 5 of Giacobbo & Mapelli (2018):
#    https://arxiv.org/pdf/1805.11100
#
#    Maxwellian velocity distribution with sigma = 265 km/s from:
#    Hobbs et al. (2005), MNRAS, 360, 974
#    https://arxiv.org/abs/astro-ph/0507584
#
#    AUTHOR: Tousif Islam
#    CREATED: 02-28-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from ..core.distributions import sample_maxwellian_1d


def sample_maxwellian_kick(sigma, n_samples=1, seed=None):
    """
    Sample natal kick velocities from a Maxwellian velocity distribution.

    Based on Equation 5 of Giacobbo & Mapelli (2018):

        f(v) ∝ (1/sigma^3) * v^2 * exp(-v^2 / (2 * sigma^2))

    where sigma is the one-dimensional root-mean-square (1D rms) velocity
    and v is the modulus of the 3D velocity.

    The default sigma = 265 km/s was derived by Hobbs et al. (2005) from
    the proper motions of 233 young isolated Galactic pulsars, corresponding
    to an average natal kick speed <v> = sigma * sqrt(8/pi) ~ 420 km/s.

    Parameters:
    -----------
    sigma : float
        1D rms velocity dispersion in km/s (default for CCSNe: 265 km/s)
    n_samples : int
        Number of kick velocities to draw (default: 1)
    seed : int or None
        Random seed for reproducibility

    Returns:
    --------
    v_kick : float or array
        Kick velocity(ies) in km/s

    References:
    -----------
    Giacobbo & Mapelli (2018), Eq. 5: https://arxiv.org/abs/1805.11100
    Hobbs et al. (2005): https://arxiv.org/abs/astro-ph/0507584
    """
    v_kick = sample_maxwellian_1d(n_samples, sigma=sigma, seed=seed)

    if n_samples == 1:
        return float(v_kick[0])
    return v_kick
