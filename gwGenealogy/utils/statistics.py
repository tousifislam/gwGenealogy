#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: statistics.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 02-28-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np


def _samples_to_distributions(samples_a, samples_b, n_bins=100):
    """
    Convert two sets of samples into normalized probability distributions
    on a common binning.

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
    p : array
        Normalized distribution for samples_a
    q : array
        Normalized distribution for samples_b
    """
    samples_a = np.asarray(samples_a)
    samples_b = np.asarray(samples_b)

    all_samples = np.concatenate([samples_a, samples_b])
    edges = np.linspace(all_samples.min(), all_samples.max(), n_bins + 1)

    p, _ = np.histogram(samples_a, bins=edges, density=True)
    q, _ = np.histogram(samples_b, bins=edges, density=True)

    eps = 1e-12
    p = p + eps
    q = q + eps

    p = p / p.sum()
    q = q / q.sum()

    return p, q


def compute_kullback_leibler_divergence(samples_a, samples_b, n_bins=100):
    """
    Compute the Kullback-Leibler divergence KL(P || Q) from two sets of samples.

    Note: KL divergence is asymmetric — KL(P||Q) != KL(Q||P).

    Parameters:
    -----------
    samples_a : array
        Samples from distribution P
    samples_b : array
        Samples from distribution Q
    n_bins : int
        Number of histogram bins (default: 100)

    Returns:
    --------
    kl : float
        KL divergence (>= 0; 0 iff P == Q)
    """
    p, q = _samples_to_distributions(samples_a, samples_b, n_bins)
    return np.sum(p * np.log(p / q))


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
    p, q = _samples_to_distributions(samples_a, samples_b, n_bins)

    m = 0.5 * (p + q)

    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    return 0.5 * kl_pm + 0.5 * kl_qm
