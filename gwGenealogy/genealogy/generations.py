#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: generations.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED: 02-28-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np

def _get_default_remnant_mass_fn():
    from gwModels.remnants import bbh_final_mass_precessing_BMR2012
    return bbh_final_mass_precessing_BMR2012

def _get_default_remnant_spin_fn():
    from gwModels.remnants import bbh_final_spin_precessing_HBR2016
    return bbh_final_spin_precessing_HBR2016

def _get_default_kick_fn():
    from gwModels.remnants import bbh_final_kick_precessing_CLZM2007
    return bbh_final_kick_precessing_CLZM2007


def _sample_isotropic_angles(rng):
    """
    Sample isotropic spin orientations and orbital-plane angle.

    Parameters:
    -----------
    rng : numpy.random.Generator
        Random number generator instance

    Returns:
    --------
    theta1, theta2, delta_phi, Theta : float
        Spin tilt angles, azimuthal difference, and orbital-plane angle (all radians)
    """
    theta1 = np.arccos(rng.uniform(-1, 1))
    theta2 = np.arccos(rng.uniform(-1, 1))
    phi1 = rng.uniform(0, 2 * np.pi)
    phi2 = rng.uniform(0, 2 * np.pi)
    delta_phi = phi1 - phi2
    Theta = rng.uniform(0, 2 * np.pi)
    return theta1, theta2, delta_phi, Theta


def run_single_merger_chain(seed_mass, seed_spin, m_pool, v_esc, m_target=100.0,
                            max_generations=20, chi_max_secondary=0.2,
                            pairing='random', pairing_beta=4.0,
                            remnant_mass_fn=None, remnant_spin_fn=None,
                            kick_fn=None, seed=None, store_history=False):
    """
    Simulate a single BH seed growing through repeated mergers with pool partners.

    The seed BH repeatedly merges with a partner drawn from m_pool until either:
    - It is ejected (kick > v_esc)
    - It reaches the target mass
    - It exceeds max_generations

    Parameters:
    -----------
    seed_mass : float
        Initial seed BH mass in solar masses
    seed_spin : float
        Initial seed BH dimensionless spin
    m_pool : array
        Pool of available BH masses for pairing
    v_esc : float
        Cluster escape velocity in km/s
    m_target : float
        Target mass to reach (default: 100.0 M_sun)
    max_generations : int
        Maximum number of mergers (default: 20)
    chi_max_secondary : float
        Maximum natal spin for secondary BHs (default: 0.2)
    pairing : str
        Pairing model: 'random', 'fragione', 'gerosa_modelB' (default: 'random')
    pairing_beta : float
        Power-law index for weighted pairing (default: 4.0)
    remnant_mass_fn : callable or None
        fn(m1, m2, chi1, chi2, theta1, theta2, dPhi) -> M_rem.
        Default: bbh_final_mass_precessing_GK2016
    remnant_spin_fn : callable or None
        fn(m1, m2, chi1, chi2, theta1, theta2, dPhi) -> chi_rem.
        Default: bbh_final_spin_precessing_GK2016
    kick_fn : callable or None
        fn(q, chi1, chi2, theta1, theta2, delta_phi, Theta) -> v_kick (km/s).
        Default: bbh_final_kick_precessing_CLZM2007
    seed : int or None
        Random seed for reproducibility
    store_history : bool
        If True, store per-generation history (default: False)

    Returns:
    --------
    dict with keys:
        'final_mass', 'final_spin', 'n_generations', 'retained',
        'reached_target', 'history' (if store_history=True)
    """
    if remnant_mass_fn is None:
        remnant_mass_fn = _get_default_remnant_mass_fn()
    if remnant_spin_fn is None:
        remnant_spin_fn = _get_default_remnant_spin_fn()
    if kick_fn is None:
        kick_fn = _get_default_kick_fn()

    rng = np.random.default_rng(seed)

    m_current = seed_mass
    chi_current = seed_spin
    generation = 1
    is_retained = True
    history = [] if store_history else None

    while is_retained and m_current < m_target and generation <= max_generations:
        # Draw secondary from pool
        m2, _ = select_from_pool(m_current, m_pool, pairing=pairing,
                                 beta=pairing_beta, seed=rng.integers(0, 2**31))

        # Secondary natal spin
        chi2 = rng.uniform(0, chi_max_secondary)

        # Isotropic angles
        theta1, theta2, delta_phi, Theta = _sample_isotropic_angles(rng)

        # Ensure q <= 1 for kick formula
        m1_ord = max(m_current, m2)
        m2_ord = min(m_current, m2)
        q = m2_ord / m1_ord

        # Order spins to match mass ordering
        if m_current >= m2:
            chi1_ord, chi2_ord = chi_current, chi2
        else:
            chi1_ord, chi2_ord = chi2, chi_current

        # Remnant mass and spin (these handle m1>=m2 swap internally)
        m_rem = remnant_mass_fn(m_current, m2, chi_current, chi2,
                                theta1, theta2, delta_phi)
        chi_rem = remnant_spin_fn(m_current, m2, chi_current, chi2,
                                  theta1, theta2, delta_phi)

        # Kick velocity
        v_kick = kick_fn(q, chi1_ord, chi2_ord, theta1=theta1, theta2=theta2,
                         delta_phi=delta_phi, Theta=Theta)

        if store_history:
            history.append({
                'generation': generation,
                'mass_before': m_current, 'spin_before': chi_current,
                'm2': m2, 'chi2': chi2, 'q': q,
                'mass_after': float(m_rem), 'spin_after': float(chi_rem),
                'v_kick': float(v_kick), 'v_esc': v_esc,
            })

        # Check retention
        if v_kick > v_esc:
            is_retained = False
        else:
            m_current = float(m_rem)
            chi_current = float(chi_rem)
            generation += 1

    result = {
        'final_mass': m_current,
        'final_spin': chi_current,
        'n_generations': generation,
        'retained': is_retained,
        'reached_target': (m_current >= m_target) and is_retained,
    }
    if store_history:
        result['history'] = history

    return result


def run_hierarchical_merger_mc(n_experiments, seed_mass=10.0, seed_spin=None,
                               m_pool=None, v_esc=100.0, m_target=100.0,
                               max_generations=20, chi_max_secondary=0.2,
                               spin_dist='uniform', pairing='random',
                               pairing_beta=4.0, remnant_mass_fn=None,
                               remnant_spin_fn=None, kick_fn=None,
                               seed=None, store_history=False, verbose=False):
    """
    Monte Carlo simulation of BH seed growth through repeated mergers.

    Runs n_experiments independent merger chains, each starting from
    seed_mass/seed_spin and repeatedly merging with partners from m_pool.

    Parameters:
    -----------
    n_experiments : int
        Number of independent MC experiments
    seed_mass : float
        Initial seed BH mass in solar masses (default: 10.0)
    seed_spin : float or None
        Initial seed BH spin. If None, drawn from spin_dist each experiment
    m_pool : array or None
        Pool of BH masses for pairing. If None, uses uniform [3, 60] M_sun
    v_esc : float
        Cluster escape velocity in km/s (default: 100.0)
    m_target : float
        Target mass for IMBH formation (default: 100.0 M_sun)
    max_generations : int
        Maximum number of mergers per chain (default: 20)
    chi_max_secondary : float
        Maximum natal spin for secondary BHs (default: 0.2)
    spin_dist : str
        Distribution for seed spin if seed_spin is None: 'uniform' (default)
    pairing : str
        Pairing model: 'random', 'fragione', 'gerosa_modelB' (default: 'random')
    pairing_beta : float
        Power-law index for weighted pairing (default: 4.0)
    remnant_mass_fn : callable or None
        Remnant mass function. Default: GK2016
    remnant_spin_fn : callable or None
        Remnant spin function. Default: GK2016
    kick_fn : callable or None
        Kick velocity function. Default: CLZM2007
    seed : int or None
        Master random seed for reproducibility
    store_history : bool
        If True, store per-chain merger history (default: False)
    verbose : bool
        If True, print progress (default: False)

    Returns:
    --------
    dict with keys:
        'final_masses', 'final_spins', 'final_generations', 'retained',
        'reached_target', 'p_retention', 'p_target', 'config',
        'histories' (if store_history=True)
    """
    if m_pool is None:
        rng_pool = np.random.default_rng(seed)
        m_pool = sample_1g_masses(1000, m_min=3.0, m_max=60.0, imf='uniform',
                                  seed=rng_pool.integers(0, 2**31))

    final_masses = np.zeros(n_experiments)
    final_spins = np.zeros(n_experiments)
    final_generations = np.zeros(n_experiments, dtype=int)
    retained = np.zeros(n_experiments, dtype=bool)
    reached_target = np.zeros(n_experiments, dtype=bool)
    histories = [] if store_history else None

    master_rng = np.random.default_rng(seed)

    for i in range(n_experiments):
        # Determine seed spin for this experiment
        if seed_spin is not None:
            this_spin = seed_spin
        else:
            if spin_dist == 'uniform':
                this_spin = master_rng.uniform(0, chi_max_secondary)
            else:
                this_spin = master_rng.uniform(0, chi_max_secondary)

        # Per-experiment seed for reproducibility
        exp_seed = master_rng.integers(0, 2**31)

        result = run_single_merger_chain(
            seed_mass=seed_mass, seed_spin=this_spin, m_pool=m_pool,
            v_esc=v_esc, m_target=m_target, max_generations=max_generations,
            chi_max_secondary=chi_max_secondary, pairing=pairing,
            pairing_beta=pairing_beta, remnant_mass_fn=remnant_mass_fn,
            remnant_spin_fn=remnant_spin_fn, kick_fn=kick_fn,
            seed=exp_seed, store_history=store_history,
        )

        final_masses[i] = result['final_mass']
        final_spins[i] = result['final_spin']
        final_generations[i] = result['n_generations']
        retained[i] = result['retained']
        reached_target[i] = result['reached_target']

        if store_history:
            histories.append(result.get('history', []))

        if verbose and (i + 1) % max(1, n_experiments // 10) == 0:
            print(f"  {i+1}/{n_experiments} experiments done "
                  f"(p_ret={retained[:i+1].mean():.3f})")

    output = {
        'final_masses': final_masses,
        'final_spins': final_spins,
        'final_generations': final_generations,
        'retained': retained,
        'reached_target': reached_target,
        'p_retention': retained.mean(),
        'p_target': reached_target.mean(),
        'config': {
            'n_experiments': n_experiments,
            'seed_mass': seed_mass,
            'v_esc': v_esc,
            'm_target': m_target,
            'max_generations': max_generations,
            'chi_max_secondary': chi_max_secondary,
            'pairing': pairing,
            'pairing_beta': pairing_beta,
        },
    }
    if store_history:
        output['histories'] = histories

    return output


def run_population_mergers(n_samples, chi_max, merge_fn=None, m_pool=None,
                           v_esc_min=1.0, v_esc_max=300.0,
                           pairing='random', pairing_kwargs=None,
                           max_gen=5, m_min=3.0, m_max=60.0, imf='uniform',
                           kick_fn=None, remnant_mass_fn=None,
                           remnant_spin_fn=None, seed=None):
    """
    Population-level merger simulation tracking mass/spin distributions
    across generations (1g -> 2g -> ... -> max_gen).

    At each generation, BHs from previous generations merge pairwise.
    Only retained BHs (kick < v_esc) survive to the next generation.

    Parameters:
    -----------
    n_samples : int
        Number of 1g BHs to generate
    chi_max : float
        Maximum natal spin for 1g BHs
    merge_fn : callable or None
        Custom merge function with signature merge_fn(m1, m2, spin1, spin2, v_esc).
        If None, uses built-in GK2016 + CLZM2007 merger
    m_pool : array or None
        External mass pool (unused in population mode; kept for API compatibility)
    v_esc_min, v_esc_max : float
        Escape velocity range in km/s (default: 1-300)
    pairing : str
        Pairing model: 'random', 'gerosa_modelB', 'fragione' (default: 'random')
    pairing_kwargs : dict or None
        Additional kwargs for pairing function (e.g., {'beta': 6.7})
    max_gen : int
        Maximum generation to simulate (default: 5)
    m_min, m_max : float
        1g BH mass range in solar masses (default: 3-60)
    imf : str
        Initial mass function: 'uniform' or 'kroupa' (default: 'uniform')
    kick_fn : callable or None
        Kick function. Default: CLZM2007
    remnant_mass_fn : callable or None
        Remnant mass function. Default: GK2016
    remnant_spin_fn : callable or None
        Remnant spin function. Default: GK2016
    seed : int or None
        Random seed for reproducibility

    Returns:
    --------
    dict : {generation: {'m': array, 'spin': array, 'q': array}}
        Mass, spin, and mass ratio arrays for each generation
    """
    if remnant_mass_fn is None:
        remnant_mass_fn = _get_default_remnant_mass_fn()
    if remnant_spin_fn is None:
        remnant_spin_fn = _get_default_remnant_spin_fn()
    if kick_fn is None:
        kick_fn = _get_default_kick_fn()
    if pairing_kwargs is None:
        pairing_kwargs = {}

    rng = np.random.default_rng(seed)

    data = {g: {'q': np.array([]), 'm': np.array([]), 'spin': np.array([])}
            for g in range(1, max_gen + 1)}

    def _get_pair(m1_arr, m_pool_arr, spin_pool_arr):
        """Select partners using the chosen pairing model."""
        n = len(m1_arr)
        if pairing == 'random':
            idx = rng.integers(0, len(m_pool_arr), size=n)
            return m_pool_arr[idx], spin_pool_arr[idx]
        elif pairing == 'gerosa_modelB':
            beta = pairing_kwargs.get('beta', 6.7)
            # Vectorized: weights[i,j] = m_pool[j]^beta if m_pool[j] < m1[i]
            weights = np.tile(m_pool_arr**beta, (n, 1))
            mask = m_pool_arr[np.newaxis, :] >= m1_arr[:, np.newaxis]
            weights[mask] = 0
            zero_rows = weights.sum(axis=1) == 0
            if zero_rows.any():
                weights[zero_rows, :] = 1.0
            weights /= weights.sum(axis=1, keepdims=True)
            cumsum = np.cumsum(weights, axis=1)
            rand = rng.random(n)[:, np.newaxis]
            idx = (cumsum < rand).sum(axis=1)
            idx = np.clip(idx, 0, len(m_pool_arr) - 1)
            return m_pool_arr[idx], spin_pool_arr[idx]
        elif pairing == 'fragione':
            beta = pairing_kwargs.get('beta', 4.0)
            weights = (m1_arr[:, np.newaxis] + m_pool_arr[np.newaxis, :])**beta
            weights /= weights.sum(axis=1, keepdims=True)
            cumsum = np.cumsum(weights, axis=1)
            rand = rng.random(n)[:, np.newaxis]
            idx = (cumsum < rand).sum(axis=1)
            idx = np.clip(idx, 0, len(m_pool_arr) - 1)
            return m_pool_arr[idx], spin_pool_arr[idx]
        else:
            raise ValueError(f"Unknown pairing: {pairing}")

    def _do_merge(m1, m2, spin1, spin2, v_esc):
        """Merge two populations and return retained remnants."""
        n = len(m1)

        # Ensure m1 >= m2 ordering for kick function
        m1_f = np.maximum(m1, m2)
        m2_f = np.minimum(m1, m2)
        spin1_f = np.where(m1 >= m2, spin1, spin2)
        spin2_f = np.where(m1 >= m2, spin2, spin1)
        q = m2_f / m1_f

        # Isotropic spin orientations
        theta1 = np.arccos(rng.uniform(-1, 1, n))
        theta2 = np.arccos(rng.uniform(-1, 1, n))
        phi1 = rng.uniform(0, 2 * np.pi, n)
        phi2 = rng.uniform(0, 2 * np.pi, n)
        delta_phi = phi1 - phi2
        Theta = rng.uniform(0, 2 * np.pi, n)

        # Remnant properties
        mf = remnant_mass_fn(m1_f, m2_f, spin1_f, spin2_f, theta1, theta2, delta_phi)
        chif = remnant_spin_fn(m1_f, m2_f, spin1_f, spin2_f, theta1, theta2, delta_phi)

        # Kick velocity
        vk = kick_fn(q, spin1_f, spin2_f, theta1=theta1, theta2=theta2,
                     delta_phi=delta_phi, Theta=Theta)

        retained_mask = vk < v_esc
        return mf, chif, q, retained_mask

    def _do_merger(gen1, gen2, target_gen, append=False):
        """Merge generations and store results."""
        n1 = len(data[gen1]['m'])
        n2 = len(data[gen2]['m'])
        if n1 == 0 or n2 == 0:
            return

        n_tmp = min(n1, n2, n_samples)

        m1 = rng.choice(data[gen1]['m'], n_tmp, replace=True)
        spin1 = rng.choice(data[gen1]['spin'], n_tmp, replace=True)
        m2, spin2 = _get_pair(m1, data[gen2]['m'], data[gen2]['spin'])
        v_esc_arr = rng.uniform(v_esc_min, v_esc_max, n_tmp)

        mf, chif, q_arr, retained_mask = _do_merge(m1, m2, spin1, spin2, v_esc_arr)

        # Compute q >= 1 for storage
        q_store = np.maximum(m1, m2) / np.minimum(m1, m2)

        if append and len(data[target_gen]['m']) > 0:
            data[target_gen]['q'] = np.concatenate(
                (data[target_gen]['q'], q_store[retained_mask]))
            data[target_gen]['m'] = np.concatenate(
                (data[target_gen]['m'], mf[retained_mask]))
            data[target_gen]['spin'] = np.concatenate(
                (data[target_gen]['spin'], chif[retained_mask]))
        else:
            data[target_gen]['q'] = q_store[retained_mask]
            data[target_gen]['m'] = mf[retained_mask]
            data[target_gen]['spin'] = chif[retained_mask]

    # 1g BHs
    data[1]['m'] = sample_1g_masses(n_samples, m_min=m_min, m_max=m_max, imf=imf,
                                    seed=rng.integers(0, 2**31))
    data[1]['spin'] = rng.uniform(0, chi_max, n_samples)

    # Build generation hierarchy: gen_i + gen_j -> gen_(i+j) where i+j = target
    # 1g + 1g -> 2g
    if max_gen >= 2:
        _do_merger(1, 1, 2)

    # For higher generations, combine all possible parent pairs
    for target in range(3, max_gen + 1):
        first = True
        for g1 in range(target - 1, 0, -1):
            g2 = target - g1
            if g2 > g1:
                break
            _do_merger(g1, g2, target, append=not first)
            first = False

    return data


def compute_retention_probability(q_values, chi_max_values, v_esc_values,
                                  n_samples=10000, kick_fn=None, seed=None):
    """
    Compute retention probability on a 3D grid of (q, chi_max, v_esc).

    For each grid point, samples n_samples isotropic spin configurations
    and computes the fraction retained (kick < v_esc).

    Parameters:
    -----------
    q_values : array
        Mass ratio values (q <= 1)
    chi_max_values : array
        Maximum spin magnitude values
    v_esc_values : array
        Escape velocity values in km/s
    n_samples : int
        Number of spin samples per grid point (default: 10000)
    kick_fn : callable or None
        Kick function. Default: CLZM2007
    seed : int or None
        Random seed for reproducibility

    Returns:
    --------
    p_ret : 3D array of shape (len(q_values), len(chi_max_values), len(v_esc_values))
        Retention probability at each grid point
    """
    if kick_fn is None:
        kick_fn = _get_default_kick_fn()

    rng = np.random.default_rng(seed)

    q_values = np.asarray(q_values)
    chi_max_values = np.asarray(chi_max_values)
    v_esc_values = np.asarray(v_esc_values)

    p_ret = np.zeros((len(q_values), len(chi_max_values), len(v_esc_values)))

    for i, q in enumerate(q_values):
        for j, chi_max in enumerate(chi_max_values):
            # Sample spins
            chi1 = rng.uniform(0, chi_max, n_samples)
            chi2 = rng.uniform(0, chi_max, n_samples)

            # Isotropic angles
            theta1 = np.arccos(rng.uniform(-1, 1, n_samples))
            theta2 = np.arccos(rng.uniform(-1, 1, n_samples))
            phi1 = rng.uniform(0, 2 * np.pi, n_samples)
            phi2 = rng.uniform(0, 2 * np.pi, n_samples)
            delta_phi = phi1 - phi2
            Theta = rng.uniform(0, 2 * np.pi, n_samples)

            # Compute kicks
            vk = kick_fn(q, chi1, chi2, theta1=theta1, theta2=theta2,
                         delta_phi=delta_phi, Theta=Theta)

            for k, v_esc in enumerate(v_esc_values):
                p_ret[i, j, k] = (vk < v_esc).mean()

    return p_ret
