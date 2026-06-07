#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: model_comparison.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 02-28-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from ..core.statistics import compute_jensen_shannon_divergence


def compare_kick_models(model_a_fn, model_b_fn, q_values, chi_max=1.0,
                        n_samples=10000, seed=None):
    """
    Compare two kick models by computing Jensen-Shannon divergence
    of their kick velocity distributions at each mass ratio.

    Parameters:
    -----------
    model_a_fn : callable
        First kick model: fn(q, chi1, chi2, theta1, theta2, deltaphi, Theta) -> v_kick
    model_b_fn : callable
        Second kick model with same signature
    q_values : array
        Mass ratio values to compare at (q = m1/m2 >= 1)
    chi_max : float
        Maximum spin magnitude (default: 1.0)
    n_samples : int
        Number of samples per q value (default: 10000)
    seed : int or None
        Random seed for reproducibility

    Returns:
    --------
    dict with keys:
        'q_values': array of mass ratios
        'jsd': array of JSD values per q
        'kicks_a': dict {q: kick_samples} for model A
        'kicks_b': dict {q: kick_samples} for model B
    """
    rng = np.random.default_rng(seed)
    q_values = np.asarray(q_values)

    jsd_values = np.zeros(len(q_values))
    kicks_a = {}
    kicks_b = {}

    for i, q in enumerate(q_values):
        # Sample spins and angles
        chi1 = rng.uniform(0, chi_max, n_samples)
        chi2 = rng.uniform(0, chi_max, n_samples)
        theta1 = np.arccos(rng.uniform(-1, 1, n_samples))
        theta2 = np.arccos(rng.uniform(-1, 1, n_samples))
        phi1 = rng.uniform(0, 2 * np.pi, n_samples)
        phi2 = rng.uniform(0, 2 * np.pi, n_samples)
        deltaphi = phi1 - phi2
        Theta = rng.uniform(0, 2 * np.pi, n_samples)

        vk_a = model_a_fn(q, chi1, chi2, theta1=theta1, theta2=theta2,
                          deltaphi=deltaphi, Theta=Theta)
        vk_b = model_b_fn(q, chi1, chi2, theta1=theta1, theta2=theta2,
                          deltaphi=deltaphi, Theta=Theta)

        kicks_a[q] = np.asarray(vk_a)
        kicks_b[q] = np.asarray(vk_b)

        jsd_values[i] = compute_jensen_shannon_divergence(kicks_a[q], kicks_b[q])

    return {
        'q_values': q_values,
        'jsd': jsd_values,
        'kicks_a': kicks_a,
        'kicks_b': kicks_b,
    }
