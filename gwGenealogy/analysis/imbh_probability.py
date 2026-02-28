#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: imbh_probability.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 02-28-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from ..clusters.generations import run_hierarchical_merger_mc


def compute_imbh_probability_grid(seed_mass_values, chi_max_values, v_esc_values,
                                  m_target=100.0, n_experiments=10000,
                                  **mc_kwargs):
    """
    Sweep over a parameter grid of (seed_mass, chi_max, v_esc) and compute
    IMBH formation probability at each point.

    At each grid point, runs a full Monte Carlo hierarchical merger simulation
    and records the probability that the seed BH reaches the target mass.

    Parameters:
    -----------
    seed_mass_values : array
        Seed BH masses to sweep over (solar masses)
    chi_max_values : array
        Maximum natal spin values to sweep over
    v_esc_values : array
        Escape velocities to sweep over (km/s)
    m_target : float
        Target IMBH mass (default: 100.0 M_sun)
    n_experiments : int
        Number of MC experiments per grid point (default: 10000)
    **mc_kwargs
        Additional keyword arguments passed to run_hierarchical_merger_mc
        (e.g., pairing, pairing_beta, max_generations, m_pool, kick_fn, etc.)

    Returns:
    --------
    dict with keys:
        'p_imbh': 3D array of shape (len(seed_mass_values), len(chi_max_values),
                  len(v_esc_values)) — IMBH formation probability
        'p_retention': 3D array — retention probability at each grid point
        'seed_mass_values': array
        'chi_max_values': array
        'v_esc_values': array
    """
    seed_mass_values = np.asarray(seed_mass_values)
    chi_max_values = np.asarray(chi_max_values)
    v_esc_values = np.asarray(v_esc_values)

    shape = (len(seed_mass_values), len(chi_max_values), len(v_esc_values))
    p_imbh = np.zeros(shape)
    p_retention = np.zeros(shape)

    for i, m_seed in enumerate(seed_mass_values):
        for j, chi_max in enumerate(chi_max_values):
            for k, v_esc in enumerate(v_esc_values):
                results = run_hierarchical_merger_mc(
                    n_experiments=n_experiments,
                    seed_mass=m_seed,
                    v_esc=v_esc,
                    m_target=m_target,
                    chi_max_secondary=chi_max,
                    **mc_kwargs,
                )
                p_imbh[i, j, k] = results['p_target']
                p_retention[i, j, k] = results['p_retention']

    return {
        'p_imbh': p_imbh,
        'p_retention': p_retention,
        'seed_mass_values': seed_mass_values,
        'chi_max_values': chi_max_values,
        'v_esc_values': v_esc_values,
    }
