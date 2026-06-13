#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: seed_growth.py
#
#    Monte Carlo simulation of BH seed growth through hierarchical mergers
#    in dense star clusters. Implements the experiment described in
#    Section III.B of Islam, Wadekar & Kritos (2026, arXiv:2603.10170).
#
#    A seed BH repeatedly merges with partners drawn from a metallicity-
#    dependent 1G BH population (Kroupa IMF -> stellar evolution -> BH masses).
#    Partners are selected with mass-weighted pairing P ∝ (m_seed + m_partner)^beta.
#    After each merger, the remnant kick is compared to the cluster escape
#    velocity to determine retention. The chain ends when the seed is ejected,
#    reaches a target mass, or hits the generation limit.
#
#    Kick models:
#      'hlz'     — Campanelli-Lousto-Zlochower-Merritt analytic formula
#      'gwmodel' — IW2025 normalizing-flow model (gwModel_flow_prec)
#
#    AUTHOR: Tousif Islam
#    CREATED: 06-08-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import matplotlib.pyplot as plt

from ..stellar import sample_1g_bh_masses_from_stellar_collapse
from ..binaries.bbh_spins import sample_spin_angles
from ..binaries.bbh_remnant import BBHRemnant, preload_kick_model
from ..utils.distributions import sample_uniform_1d, seed_legacy_rng


class MonteCarloBHSeedGrowth:
    """Monte Carlo simulation of BH seed growth through hierarchical mergers.

    Parameters
    ----------
    v_esc : float
        Cluster escape velocity [km/s].
    Z : float
        Metallicity (default: 0.005). Controls the 1G BH mass pool via
        stellar evolution. Ignored if m_pool is provided.
    chi_max : float
        Maximum natal spin magnitude for both seed and 1G BHs (default: 0.2).
    m_seed : float
        Initial seed BH mass [Msun] (default: 10.0).
    m_targets : list of float or None
        Target BBH masses [Msun] to track (default: [100, 250]).
    beta : float
        Pairing steepness: P(partner) ∝ (m_seed + m_partner)^beta (default: 4.0).
        beta=0 gives random (uniform) pairing.
    max_generations : int
        Maximum number of mergers per chain (default: 20).
    precessing : bool
        If True (default), use precessing remnant models. If False,
        use nonprecessing (aligned-spin) models.
    mass_spin_model : str or None
        Remnant mass/spin model passed to BBHRemnant. None uses the default
        ('hbr' for precessing, 'uib' for nonprecessing).
    evolve_v_esc : bool
        If True, escape velocity decays with generation as
        v_esc(g) = v_esc * g^(v_esc_decay_index). Default: False.
    v_esc_decay_index : float
        Power-law index for v_esc evolution (default: -0.35).
        Only used when evolve_v_esc=True.
    kick_model : str
        Kick model passed to BBHRemnant: 'gwmodel' (default) or 'hlz'
        for precessing; 'gwmodel_kick_q200' (default) or 'hlz' for
        nonprecessing.
    m_pool : array or None
        Pre-built 1G BH mass pool. If provided, skips stellar evolution
        and uses this array directly. Useful when sweeping over v_esc
        with the same metallicity.
    n_pool : int
        Number of ZAMS stars to sample for the 1G pool (default: 5000).
        Ignored if m_pool is provided.
    seed : int or None
        Master random seed for reproducibility.
    """

    def __init__(self, v_esc=100.0, Z=0.005, chi_max=0.2, m_seed=10.0,
                 m_targets=None, beta=4.0, max_generations=20,
                 precessing=True, mass_spin_model=None, kick_model=None,
                 evolve_v_esc=False, v_esc_decay_index=-0.35,
                 m_pool=None, n_pool=5000, seed=None):
        self.v_esc = float(v_esc)
        self.Z = float(Z)
        self.chi_max = float(chi_max)
        self.m_seed = float(m_seed)
        self.m_targets = sorted(m_targets or [100.0, 250.0])
        self.beta = float(beta)
        self.max_generations = max_generations
        self.precessing = precessing
        self.mass_spin_model = mass_spin_model
        self.kick_model = kick_model or ('gwmodel' if precessing else 'gwmodel_kick_q200')
        self.evolve_v_esc = evolve_v_esc
        self.v_esc_decay_index = float(v_esc_decay_index)
        self.seed = seed

        rng = np.random.default_rng(seed)

        # Use pre-built pool or generate from stellar evolution
        if m_pool is not None:
            self.m_pool = np.asarray(m_pool, dtype=float)
        else:
            # Build 1G BH mass pool: Kroupa IMF -> ZAMS -> Fryer12 delayed -> filter to BHs
            # m_zams_min=10 since ZAMS < 10 Msun never form BHs (Kroupa IMF starts at 0.03 Msun)
            self.m_pool = sample_1g_bh_masses_from_stellar_collapse(
                n_pool, Z=Z, imf='kroupa', m_zams_min=10.0, m_zams_max=150.0,
                seed=rng.integers(0, 2**31))

    def _v_esc_at_generation(self, v_esc_0, generation):
        """Return escape velocity at a given generation (1-indexed)."""
        if self.evolve_v_esc:
            return v_esc_0 * generation ** self.v_esc_decay_index
        return v_esc_0

    def _run_single_chain(self, rng, store_history=False, v_esc_override=None):
        """Run one seed growth chain. Returns a result dict."""
        v_esc_0 = v_esc_override if v_esc_override is not None else self.v_esc
        m = self.m_seed
        # Initial seed spin drawn uniformly in [0, chi_max]
        chi = float(sample_uniform_1d(1, low=0, high=self.chi_max, seed=int(rng.integers(0, 2**31)))[0])
        m_target_max = max(self.m_targets)
        crossed = {mt: False for mt in self.m_targets}
        history = [] if store_history else None

        generation = 0
        is_retained = True

        # Keep merging while seed is retained, below target mass, and under gen limit
        while is_retained and m < m_target_max and generation < self.max_generations:
            # Mark any target masses already crossed before this merger
            for mt in self.m_targets:
                if m >= mt:
                    crossed[mt] = True

            # Partner from 1G pool with mass-weighted pairing
            weights = (m + self.m_pool) ** self.beta
            weights /= weights.sum()
            m2 = rng.choice(self.m_pool, p=weights)
            # Partner natal spin drawn uniformly in [0, chi_max]
            chi2 = float(sample_uniform_1d(1, low=0, high=self.chi_max, seed=int(rng.integers(0, 2**31)))[0])

            # Isotropic spin orientations
            theta1, theta2, phi1, phi2 = sample_spin_angles(1, 'isotropic', seed=int(rng.integers(0, 2**31)))
            theta1, theta2 = float(theta1[0]), float(theta2[0])
            phi1, phi2 = float(phi1[0]), float(phi2[0])

            # Mass ordering: m1 >= m2, spins follow
            if m >= m2:
                m1_ord, m2_ord, a1_ord, a2_ord = m, m2, chi, chi2
            else:
                m1_ord, m2_ord, a1_ord, a2_ord = m2, m, chi2, chi

            # Seed legacy RNG backends so kick models with internal randomness
            # (CLZM2007 Theta, gwmodel flow) are reproducible from our rng chain.
            seed_legacy_rng(rng)

            # Compute remnant mass, spin, and kick via BBHRemnant
            rem = BBHRemnant(m1=m1_ord, m2=m2_ord, a1=a1_ord, a2=a2_ord,
                             theta1=theta1, theta2=theta2, phi1=phi1, phi2=phi2,
                             precessing=self.precessing,
                             mass_spin_model=self.mass_spin_model,
                             kick_model=self.kick_model)
            m_rem, chi_rem, v_kick = float(rem.Mf[0]), abs(float(rem.af[0])), float(rem.vkick[0])

            merger_gen = generation + 1
            v_esc = self._v_esc_at_generation(v_esc_0, merger_gen)

            if store_history:
                history.append({
                    'generation': merger_gen,
                    'mass_before': m, 'spin_before': chi,
                    'm2': m2, 'chi2': chi2, 'q': m1_ord / m2_ord,
                    'mass_after': m_rem, 'spin_after': chi_rem,
                    'v_kick': v_kick, 'v_esc': v_esc,
                })

            # Seed ejected if kick exceeds escape velocity; otherwise grow
            if v_kick > v_esc:
                is_retained = False
            else:
                m = m_rem
                chi = chi_rem
                generation += 1

        # Final target check after loop exit (last merger may have crossed a target)
        for mt in self.m_targets:
            if m >= mt:
                crossed[mt] = True

        result = {
            'final_mass': m,
            'final_spin': chi,
            'n_generations': generation,
            'retained': is_retained,
            'reached_target': {mt: crossed[mt] and is_retained
                               for mt in self.m_targets},
        }
        if store_history:
            result['history'] = history
        return result

    def simulate(self, n_experiments=10000, store_history=False, verbose=False):
        """Run the full MC simulation.

        Parameters
        ----------
        n_experiments : int
            Number of independent seed growth chains (default: 10000).
        store_history : bool
            If True, store per-generation merger details for each chain
            (default: False).
        verbose : bool
            Print progress every 10% (default: False).

        Returns
        -------
        dict with keys:
            final_masses, final_spins, final_generations : arrays
            retained : bool array
            reached_target : dict of {m_target: bool array}
            P_ret : float (retention probability)
            P_target : dict of {m_target: float}
            median_mass : float (median final mass of retained seeds)
            params : dict (simulation parameters)
            histories : list of list-of-dicts (only if store_history=True)
        """
        rng = np.random.default_rng(self.seed)
        # Preload so the one-time lazy flow load can't perturb the seeded path.
        preload_kick_model(self.kick_model)

        # Pre-allocate output arrays for all experiments
        final_masses = np.zeros(n_experiments)
        final_spins = np.zeros(n_experiments)
        final_gens = np.zeros(n_experiments, dtype=int)

        # Whether each seed survived all mergers without ejection
        retained = np.zeros(n_experiments, dtype=bool)
        # Whether each seed crossed each target mass while still retained
        reached_target = {mt: np.zeros(n_experiments, dtype=bool)
                         for mt in self.m_targets}
        # Per-chain merger-by-merger records (only when store_history=True)
        histories = [] if store_history else None

        # Run independent seed growth chains
        for i in range(n_experiments):

            # Single chain with its own RNG for reproducibility
            exp_rng = np.random.default_rng(rng.integers(0, 2**63))
            result = self._run_single_chain(exp_rng, store_history=store_history)

            # Collect results from this chain
            final_masses[i] = result['final_mass']
            final_spins[i] = result['final_spin']
            final_gens[i] = result['n_generations']
            retained[i] = result['retained']
            for mt in self.m_targets:
                reached_target[mt][i] = result['reached_target'][mt]
            if store_history:
                histories.append(result['history'])

            # Print progress every 10% of experiments
            if verbose and (i + 1) % max(1, n_experiments // 10) == 0:
                print(f"  {i+1}/{n_experiments} done "
                      f"(P_ret={retained[:i+1].mean():.3f})")

        # Aggregate statistics across all chains
        output = {
            'final_masses': final_masses,
            'final_spins': final_spins,
            'final_generations': final_gens,
            'retained': retained,
            'reached_target': reached_target,
            'P_ret': float(retained.mean()),
            'P_target': {mt: float(reached_target[mt].mean())
                         for mt in self.m_targets},
            'median_mass': (float(np.median(final_masses[retained]))
                           if retained.any() else np.nan),
            'params': {
                'v_esc': self.v_esc, 'Z': self.Z,
                'chi_max': self.chi_max, 'm_seed': self.m_seed,
                'm_targets': self.m_targets, 'beta': self.beta,
                'kick_model': self.kick_model,
                'n_pool': len(self.m_pool),
                'evolve_v_esc': self.evolve_v_esc,
                'v_esc_decay_index': self.v_esc_decay_index,
            },
        }
        if store_history:
            output['histories'] = histories
        return output

    def simulate_grid(self, v_esc_values, n_experiments=10000, verbose=False):
        """Sweep over escape velocities, reusing the same 1G pool.

        Parameters
        ----------
        v_esc_values : array-like
            Escape velocities [km/s] to sweep over.
        n_experiments : int
            Number of chains per v_esc value (default: 10000).
        verbose : bool
            Print progress per v_esc value (default: False).

        Returns
        -------
        dict with keys:
            v_esc_values : array
            P_ret : array of retention probabilities
            P_target : dict of {m_target: array of probabilities}
            median_mass : array of median retained masses
        """
        v_esc_values = np.atleast_1d(np.asarray(v_esc_values, dtype=float))
        n_v = len(v_esc_values)

        P_ret = np.zeros(n_v)
        P_target = {mt: np.zeros(n_v) for mt in self.m_targets}
        median_mass = np.full(n_v, np.nan)
        # Preload so the one-time lazy flow load can't perturb the seeded path.
        preload_kick_model(self.kick_model)

        # Loop over each escape velocity value
        for j, v_esc in enumerate(v_esc_values):
            # Reset RNG per v_esc so each gets the same chain seeds
            rng = np.random.default_rng(self.seed)
            # Per-v_esc accumulators
            retained = np.zeros(n_experiments, dtype=bool)
            reached = {mt: np.zeros(n_experiments, dtype=bool) for mt in self.m_targets}
            masses = np.zeros(n_experiments)

            # Run all chains at this v_esc
            for i in range(n_experiments):
                exp_rng = np.random.default_rng(rng.integers(0, 2**63))
                result = self._run_single_chain(exp_rng, v_esc_override=v_esc)
                retained[i] = result['retained']
                masses[i] = result['final_mass']
                for mt in self.m_targets:
                    reached[mt][i] = result['reached_target'][mt]

            # Retention probability at this v_esc
            P_ret[j] = retained.mean()
            # Per-target crossing probability at this v_esc
            for mt in self.m_targets:
                P_target[mt][j] = reached[mt].mean()
            # Median final mass of retained seeds at this v_esc
            if retained.any():
                median_mass[j] = np.median(masses[retained])

            if verbose:
                print(f"  v_esc={v_esc:.0f} km/s: P_ret={P_ret[j]:.3f}")

        return {
            'v_esc_values': v_esc_values,
            'P_ret': P_ret,
            'P_target': P_target,
            'median_mass': median_mass,
        }

    def __repr__(self):
        v_esc_str = f"v_esc={self.v_esc}"
        if self.evolve_v_esc:
            v_esc_str += f"*g^({self.v_esc_decay_index})"
        return (f"MonteCarloBHSeedGrowth({v_esc_str}, Z={self.Z}, "
                f"chi_max={self.chi_max}, m_seed={self.m_seed}, "
                f"kick={self.kick_model}, pool={len(self.m_pool)} BHs)")


class IMBHFormationProbability:
    """IMBH formation probability on a (seed_mass, chi_max, v_esc) grid.

    At each grid point, runs a full Monte Carlo seed growth simulation
    via MonteCarloBHSeedGrowth and records the probability that the seed
    BH reaches the target mass.

    Implements the parameter sweep from Section III.C / Figure 4 of
    Islam, Wadekar & Kritos (2026, arXiv:2603.10170).

    Parameters
    ----------
    seed_mass_values : array-like
        Seed BH masses to sweep over [Msun].
    chi_max_values : array-like
        Maximum natal spin values to sweep over.
    v_esc_values : array-like
        Escape velocities to sweep over [km/s].
    m_target : float
        Target IMBH mass [Msun] (default: 100.0).
    n_experiments : int
        Number of MC chains per grid point (default: 10000).
    **mc_kwargs
        Additional keyword arguments passed to MonteCarloBHSeedGrowth
        (e.g., Z, beta, max_generations, kick_model, precessing,
        n_pool, evolve_v_esc, v_esc_decay_index, seed).
    """

    def __init__(self, seed_mass_values, chi_max_values, v_esc_values,
                 m_target=100.0, n_experiments=10000, **mc_kwargs):
        self.seed_mass_values = np.atleast_1d(np.asarray(seed_mass_values, dtype=float))
        self.chi_max_values = np.atleast_1d(np.asarray(chi_max_values, dtype=float))
        self.v_esc_values = np.atleast_1d(np.asarray(v_esc_values, dtype=float))
        self.m_target = float(m_target)
        self.n_experiments = n_experiments
        self.mc_kwargs = mc_kwargs

    def compute(self, verbose=False):
        """Run the IMBH probability grid computation.

        Returns
        -------
        dict with keys:
            'p_imbh' : 3D array (n_seed, n_chi, n_vesc)
            'p_retention' : 3D array (n_seed, n_chi, n_vesc)
            'seed_mass_values' : array
            'chi_max_values' : array
            'v_esc_values' : array
        """
        ns = len(self.seed_mass_values)
        nc = len(self.chi_max_values)
        nv = len(self.v_esc_values)

        p_imbh = np.zeros((ns, nc, nv))
        p_ret = np.zeros((ns, nc, nv))

        total = ns * nc * nv
        count = 0

        for i, m_seed in enumerate(self.seed_mass_values):
            for j, chi_max in enumerate(self.chi_max_values):
                for k, v_esc in enumerate(self.v_esc_values):
                    mc = MonteCarloBHSeedGrowth(
                        v_esc=v_esc, chi_max=chi_max, m_seed=m_seed,
                        m_targets=[self.m_target], **self.mc_kwargs)
                    res = mc.simulate(n_experiments=self.n_experiments)

                    p_imbh[i, j, k] = res['P_target'][self.m_target]
                    p_ret[i, j, k] = res['P_ret']

                    count += 1
                    if verbose and count % max(1, total // 10) == 0:
                        print(f"  {count}/{total} grid points done")

        self.results = {
            'p_imbh': p_imbh,
            'p_retention': p_ret,
            'seed_mass_values': self.seed_mass_values,
            'chi_max_values': self.chi_max_values,
            'v_esc_values': self.v_esc_values,
        }
        return self.results

    def plot_heatmap(self, v_esc, axes='seed_chi', ax=None,
                    cmap='inferno', vmin=None, vmax=None):
        """Plot a single P_IMBH heatmap slice at a given v_esc.

        Parameters
        ----------
        v_esc : float
            Escape velocity value (must be in v_esc_values).
        axes : str
            Which 2D slice: 'seed_chi' (default) plots seed_mass vs chi_max.
        ax : matplotlib.axes.Axes or None
            Axes to plot on. If None, creates a new figure.
        cmap : str
            Colormap (default: 'inferno').
        vmin, vmax : float or None
            Colorbar limits.

        Returns
        -------
        fig, ax, im : Figure, Axes, and ScalarMappable
        """
        if not hasattr(self, 'results'):
            raise RuntimeError("Call .compute() before plotting.")

        k = np.argmin(np.abs(self.v_esc_values - v_esc))
        p_2d = self.results['p_imbh'][:, :, k]

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = ax.figure

        im = ax.pcolormesh(self.seed_mass_values, self.chi_max_values, p_2d.T,
                           cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.set_xlabel(r'$m_{\rm seed}$ $[M_{\odot}]$')
        ax.set_ylabel(r'$\chi_{\rm max}$')
        ax.set_title(f'$v_{{\\rm esc}}$={self.v_esc_values[k]:.0f} km/s')

        return fig, ax, im

    def plot_heatmap_all_vesc(self, cmap='inferno', vmin=None, vmax=None, figsize=None):
        """Multi-panel P_IMBH heatmap across v_esc values.

        Parameters
        ----------
        cmap : str
            Colormap (default: 'inferno').
        vmin, vmax : float or None
            Colorbar limits.
        figsize : tuple or None
            Figure size. Default scales with number of panels.

        Returns
        -------
        fig, axes : Figure and array of Axes
        """
        if not hasattr(self, 'results'):
            raise RuntimeError("Call .compute() before plotting.")

        nv = len(self.v_esc_values)
        ncols = min(nv, 4)
        nrows = int(np.ceil(nv / ncols))
        if figsize is None:
            figsize = (5 * ncols + 1, 4 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)

        for idx, v_esc in enumerate(self.v_esc_values):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            p_2d = self.results['p_imbh'][:, :, idx]
            im = ax.pcolormesh(self.seed_mass_values, self.chi_max_values,
                               p_2d.T, cmap=cmap, vmin=vmin, vmax=vmax,
                               shading='auto')
            ax.text(0.05, 0.95,
                    f'$v_{{\\rm esc}}$={v_esc:.0f} km/s',
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', color='w')
            if r == nrows - 1:
                ax.set_xlabel(r'$m_{\rm seed}$ $[M_{\odot}]$')
            if c == 0:
                ax.set_ylabel(r'$\chi_{\rm max}$')

        # Hide unused axes
        for idx in range(nv, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_visible(False)

        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(r'$p_{\rm IMBH}$')

        plt.tight_layout(rect=[0, 0, 0.88, 1])
        return fig, axes

    def __repr__(self):
        return (f"IMBHFormationProbability("
                f"m_seed=[{self.seed_mass_values[0]:.1f},{self.seed_mass_values[-1]:.1f}]"
                f"x{len(self.seed_mass_values)}, "
                f"chi=[{self.chi_max_values[0]:.2f},{self.chi_max_values[-1]:.2f}]"
                f"x{len(self.chi_max_values)}, "
                f"v_esc=[{self.v_esc_values[0]:.0f},{self.v_esc_values[-1]:.0f}]"
                f"x{len(self.v_esc_values)}, "
                f"m_target={self.m_target}, n={self.n_experiments})")
