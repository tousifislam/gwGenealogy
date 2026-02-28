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


def compute_jensen_shannon_divergence(samples_a, samples_b, n_bins=100):
    """
    Compute the Jensen-Shannon divergence between two sets of samples.

    JSD is a symmetric, bounded measure of the similarity between two
    probability distributions, based on the KL divergence.

    Parameters:
    -----------
    samples_a : array
        First set of samples
    samples_b : array
        Second set of samples
    n_bins : int
        Number of histogram bins (default: 100)

    Returns:
    --------
    jsd : float
        Jensen-Shannon divergence (0 = identical, ln(2) = maximally different)
    """
    samples_a = np.asarray(samples_a)
    samples_b = np.asarray(samples_b)

    # Common bin edges
    all_samples = np.concatenate([samples_a, samples_b])
    edges = np.linspace(all_samples.min(), all_samples.max(), n_bins + 1)

    # Histograms -> probability distributions
    p, _ = np.histogram(samples_a, bins=edges, density=True)
    q, _ = np.histogram(samples_b, bins=edges, density=True)

    # Add small epsilon to avoid log(0)
    eps = 1e-12
    p = p + eps
    q = q + eps

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    # Mixture distribution
    m = 0.5 * (p + q)

    # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    return 0.5 * kl_pm + 0.5 * kl_qm


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
        Mass ratio values to compare at (q <= 1)
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
