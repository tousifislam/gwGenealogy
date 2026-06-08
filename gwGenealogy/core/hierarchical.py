#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: hierarchical_mergers.py
#
#    Monte Carlo simulation of hierarchical BH mergers across generations
#    in dense star clusters. Implements the experiment described in
#    Section III.C of Islam, Wadekar & Kritos (2026, arXiv:2603.10170).
#
#    A population of 1G BHs (from an IMF) merges pairwise to produce
#    higher-generation remnants. Generation convention follows the paper:
#      remnant_gen = max(parent_gen1, parent_gen2) + 1
#    This means 2G+2G -> 3G (not 4G). Only retained remnants (kick < v_esc)
#    survive to participate in subsequent mergers.
#
#    Generation hierarchy (for max_gen=5):
#      2G: (1G+1G)
#      3G: (2G+1G), (2G+2G)
#      4G: (3G+1G), (3G+2G), (3G+3G)
#      5G: (4G+1G), (4G+2G), (4G+3G), (4G+4G)
#
#    Supported configurations:
#      1G masses:  'uniform' in [m_min, m_max], 'powerlaw' p(m) ~ m^gamma
#      1G spins:   'uniform' in [0, chi_max], 'beta' Beta(a, b)
#      v_esc:      'uniform' in [v_min, v_max], 'gaussian' N(mu, sigma)
#      Pairing:    'random', 'secondary_mass_power_law' p(m2|m1) ~ m2^beta (m2<m1),
#                  'total_mass_power_law' p(m2) ~ (m1+m2)^beta
#      Remnants:   via BBHRemnant (precessing or nonprecessing)
#
#    AUTHOR: Tousif Islam
#    CREATED: 06-08-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ..binaries.bbh_remnant import BBHRemnant
from ..binaries.bbh_spins import sample_spin_angles
from ..utils.distributions import (sample_uniform_1d, sample_powerlaw_1d,
                                   sample_gaussian_1d, sample_beta_1d)


class HierarchicalMergersInClusters:
    """Monte Carlo simulation of hierarchical BH mergers across generations.

    Generation convention: remnant_gen = max(parent_gen1, parent_gen2) + 1.
    This means 2G+2G -> 3G, not 4G.

    Parameters
    ----------
    n_samples : int
        Number of 1G BHs to generate (default: 5000).
    chi_max : float
        Maximum natal spin magnitude for 1G BHs (default: 0.2).
    m_min, m_max : float
        1G BH mass range [Msun] (default: 3.0-60.0).
    imf : str
        Initial mass function for 1G BH masses:
        'uniform' (default) or 'powerlaw' (p(m) ~ m^imf_gamma).
    imf_gamma : float
        Power-law index when imf='powerlaw' (default: -2.5).
    spin_dist : str
        1G spin magnitude distribution: 'uniform' (default) or 'beta'.
    beta_a, beta_b : float
        Beta distribution parameters for spin_dist='beta'
        (default: 1.4, 3.6; from arXiv:2111.03634).
    v_esc_min, v_esc_max : float
        Escape velocity range [km/s] for uniform sampling (default: 1-300).
    v_esc_dist : str
        Escape velocity distribution: 'uniform' (default) or 'gaussian'.
    v_esc_mean, v_esc_sigma : float
        Mean and std for Gaussian v_esc (default: 150, 45 km/s).
    pairing : str
        Binary pairing model: 'random' (default), 'secondary_mass_power_law'
        (p(m2|m1) ~ m2^beta, m2 < m1), or 'total_mass_power_law'
        (p(m2) ~ (m1+m2)^beta).
    pairing_beta : float or None
        Power-law index for pairing. None uses defaults:
        6.7 for secondary_mass_power_law, 4.0 for total_mass_power_law.
    evolve_v_esc : bool
        If True, escape velocity decays with generation as
        v_esc(g) = v_esc_sampled * g^(v_esc_decay_index). Default: False.
    v_esc_decay_index : float
        Power-law index for v_esc evolution (default: -0.35).
        Only used when evolve_v_esc=True.
    max_gen : int
        Maximum generation to simulate (default: 5).
    precessing : bool
        If True (default), use precessing remnant models.
    mass_spin_model : str or None
        Remnant mass/spin model for BBHRemnant (default: 'hbr').
    kick_model : str or None
        Kick model for BBHRemnant. Default: 'gwmodel' (precessing)
        or 'gwmodel_kick_q200' (nonprecessing).
    seed : int or None
        Master random seed for reproducibility.
    """

    def __init__(self, n_samples=5000, chi_max=0.2,
                 m_min=3.0, m_max=60.0, imf='uniform', imf_gamma=-2.5,
                 spin_dist='uniform', beta_a=1.4, beta_b=3.6,
                 v_esc_min=1.0, v_esc_max=300.0, v_esc_dist='uniform',
                 v_esc_mean=150.0, v_esc_sigma=45.0,
                 pairing='random', pairing_beta=None,
                 evolve_v_esc=False, v_esc_decay_index=-0.35,
                 max_gen=5,
                 precessing=True, mass_spin_model=None, kick_model=None,
                 seed=None):
        
        # 1G BH population parameters
        self.n_samples = n_samples
        self.chi_max = float(chi_max)
        self.m_min = float(m_min)
        self.m_max = float(m_max)
        self.imf = imf
        self.imf_gamma = float(imf_gamma)

        # 1G spin distribution parameters
        self.spin_dist = spin_dist
        self.beta_a = float(beta_a)
        self.beta_b = float(beta_b)

        # Escape velocity distribution parameters
        self.v_esc_min = float(v_esc_min)
        self.v_esc_max = float(v_esc_max)
        self.v_esc_dist = v_esc_dist
        self.v_esc_mean = float(v_esc_mean)
        self.v_esc_sigma = float(v_esc_sigma)

        # Escape velocity evolution
        self.evolve_v_esc = evolve_v_esc
        self.v_esc_decay_index = float(v_esc_decay_index)

        # Merger configuration
        self.pairing = pairing
        self.max_gen = max_gen
        self.precessing = precessing
        self.mass_spin_model = mass_spin_model
        self.kick_model = kick_model or ('gwmodel' if precessing else 'gwmodel_kick_q200')
        self.seed = seed

        # Auto-select pairing power-law index from model defaults
        if pairing_beta is None:
            if pairing == 'secondary_mass_power_law':
                self.pairing_beta = 6.7
            elif pairing == 'total_mass_power_law':
                self.pairing_beta = 4.0
            else:
                self.pairing_beta = 0.0
        else:
            self.pairing_beta = float(pairing_beta)

    def _sample_1g(self, rng):
        """Sample 1G BH masses and spins."""
        
        # Sub-seeds for independent mass and spin sampling
        seed_m = int(rng.integers(0, 2**31))
        seed_s = int(rng.integers(0, 2**31))

        # Sample 1G BH masses from chosen IMF
        if self.imf == 'uniform':
            masses = sample_uniform_1d(self.n_samples, low=self.m_min,
                                       high=self.m_max, seed=seed_m)
        elif self.imf == 'powerlaw':
            masses = sample_powerlaw_1d(self.n_samples, beta=self.imf_gamma,
                                        xmin=self.m_min, xmax=self.m_max,
                                        seed=seed_m)
        else:
            raise ValueError(f"Unknown IMF: {self.imf}. "
                             "Choose 'uniform' or 'powerlaw'.")

        # Sample 1G natal spin magnitudes
        if self.spin_dist == 'uniform':
            spins = sample_uniform_1d(self.n_samples, low=0,
                                      high=self.chi_max, seed=seed_s)
        elif self.spin_dist == 'beta':
            spins = sample_beta_1d(self.n_samples, a=self.beta_a,
                                   b=self.beta_b, seed=seed_s)
        else:
            raise ValueError(f"Unknown spin_dist: {self.spin_dist}. "
                             "Choose 'uniform' or 'beta'.")

        return masses, spins

    def _sample_v_esc(self, n, rng, generation=1):
        """Sample escape velocities, optionally scaled by generation."""
        seed_v = int(rng.integers(0, 2**31))
        if self.v_esc_dist == 'uniform':
            v_esc = sample_uniform_1d(n, low=self.v_esc_min,
                                      high=self.v_esc_max, seed=seed_v)
        elif self.v_esc_dist == 'gaussian':
            v_esc = sample_gaussian_1d(n, mean=self.v_esc_mean,
                                       std=self.v_esc_sigma, seed=seed_v)
            v_esc = np.clip(v_esc, self.v_esc_min, self.v_esc_max)
        else:
            raise ValueError(f"Unknown v_esc_dist: {self.v_esc_dist}. "
                             "Choose 'uniform' or 'gaussian'.")
        if self.evolve_v_esc:
            v_esc *= generation ** self.v_esc_decay_index
        return v_esc

    def _get_pair(self, m1_arr, m_pool, spin_pool, rng):
        """Select partners using the chosen pairing model.

        For 'random' pairing, mass and spin are drawn independently from the
        pool (matching the original implementation). For 'secondary_mass_power_law' and
        'total_mass_power_law', the selected spin corresponds to the selected mass index.
        """
        n = len(m1_arr)

        # Random: independent draws from pool
        if self.pairing == 'random':
            m_idx = rng.integers(0, len(m_pool), size=n)
            s_idx = rng.integers(0, len(spin_pool), size=n)
            return m_pool[m_idx], spin_pool[s_idx]

        # Gerosa & Berti (2019) Model B: p(m2|m1) ~ m2^beta for m2 < m1
        elif self.pairing == 'secondary_mass_power_law':
            weights = np.tile(m_pool**self.pairing_beta, (n, 1))
            mask = m_pool[np.newaxis, :] >= m1_arr[:, np.newaxis]
            weights[mask] = 0
            zero_rows = weights.sum(axis=1) == 0
            if zero_rows.any():
                weights[zero_rows, :] = 1.0
            weights /= weights.sum(axis=1, keepdims=True)
            cumsum = np.cumsum(weights, axis=1)
            rand = rng.random(n)[:, np.newaxis]
            idx = (cumsum < rand).sum(axis=1)
            idx = np.clip(idx, 0, len(m_pool) - 1)
            return m_pool[idx], spin_pool[idx]

        # O'Leary et al. (2016): p(m2) ~ (m1 + m2)^beta
        elif self.pairing == 'total_mass_power_law':
            weights = (m1_arr[:, np.newaxis]
                       + m_pool[np.newaxis, :])**self.pairing_beta
            weights /= weights.sum(axis=1, keepdims=True)
            cumsum = np.cumsum(weights, axis=1)
            rand = rng.random(n)[:, np.newaxis]
            idx = (cumsum < rand).sum(axis=1)
            idx = np.clip(idx, 0, len(m_pool) - 1)
            return m_pool[idx], spin_pool[idx]

        else:
            raise ValueError(f"Unknown pairing: {self.pairing}. "
                             "Choose 'random', 'secondary_mass_power_law', or 'total_mass_power_law'.")

    def _compute_remnants(self, m1, m2, spin1, spin2, v_esc, rng):
        """Compute remnant properties for paired BH arrays via BBHRemnant."""
        n = len(m1)

        # Ensure m1 >= m2 and reorder spins to match
        m1_ord = np.maximum(m1, m2)
        m2_ord = np.minimum(m1, m2)
        a1_ord = np.where(m1 >= m2, spin1, spin2)
        a2_ord = np.where(m1 >= m2, spin2, spin1)

        theta1, theta2, phi1, phi2 = sample_spin_angles(
            n, 'isotropic', seed=int(rng.integers(0, 2**31)))

        # Remnant mass, spin, and kick via BBHRemnant
        rem = BBHRemnant(m1=m1_ord, m2=m2_ord, a1=a1_ord, a2=a2_ord,
                         theta1=theta1, theta2=theta2, phi1=phi1, phi2=phi2,
                         precessing=self.precessing,
                         mass_spin_model=self.mass_spin_model,
                         kick_model=self.kick_model)

        q = m1_ord / m2_ord
        retained = rem.vkick < v_esc

        return rem.Mf, np.abs(rem.af), q, retained

    def simulate(self, verbose=False):
        """Run the hierarchical merger simulation across generations.

        Parameters
        ----------
        verbose : bool
            Print generation counts (default: False).

        Returns
        -------
        dict : {generation: {'m': array, 'spin': array, 'q': array}}
            Mass, spin, and mass ratio arrays for retained BHs at each
            generation. Generation 1 contains the initial 1G population
            (q is empty for 1G).
        """
        rng = np.random.default_rng(self.seed)

        # Pre-allocate empty arrays for each generation
        data = {g: {'m': np.array([]), 'spin': np.array([]),
                     'q': np.array([])}
                for g in range(1, self.max_gen + 1)}

        # Generate 1G BH population
        masses, spins = self._sample_1g(rng)
        data[1]['m'] = masses
        data[1]['spin'] = spins

        if verbose:
            spin_info = (f"chi~Beta({self.beta_a},{self.beta_b})"
                         if self.spin_dist == 'beta'
                         else f"chi~U(0,{self.chi_max})")
            v_esc_info = (f"v_esc~N({self.v_esc_mean},{self.v_esc_sigma})"
                          if self.v_esc_dist == 'gaussian'
                          else f"v_esc~U({self.v_esc_min},{self.v_esc_max})")
            print(f"1g: {len(data[1]['m'])} BHs "
                  f"(m=[{self.m_min},{self.m_max}] Msun, IMF={self.imf}, "
                  f"{spin_info}, {v_esc_info})")

        def _merge_generations(gen1, gen2, target_gen, append=False):
            """Pair, merge, and store retained BHs from two generation pools."""
            
            # Skip if either parent generation is empty
            n1 = len(data[gen1]['m'])
            n2 = len(data[gen2]['m'])
            if n1 == 0 or n2 == 0:
                return
            n_tmp = min(n1, n2, self.n_samples)

            # Draw primaries from gen1, pair with partners from gen2
            m1 = rng.choice(data[gen1]['m'], n_tmp, replace=True)
            spin1 = rng.choice(data[gen1]['spin'], n_tmp, replace=True)
            m2, spin2 = self._get_pair(
                m1, data[gen2]['m'], data[gen2]['spin'], rng)
            v_esc = self._sample_v_esc(n_tmp, rng, generation=target_gen)

            mf, chif, q, retained = self._compute_remnants(
                m1, m2, spin1, spin2, v_esc, rng)

            # Append to target generation or initialize it
            if append and len(data[target_gen]['m']) > 0:
                data[target_gen]['m'] = np.concatenate(
                    (data[target_gen]['m'], mf[retained]))
                data[target_gen]['spin'] = np.concatenate(
                    (data[target_gen]['spin'], chif[retained]))
                data[target_gen]['q'] = np.concatenate(
                    (data[target_gen]['q'], q[retained]))
            else:
                data[target_gen]['m'] = mf[retained]
                data[target_gen]['spin'] = chif[retained]
                data[target_gen]['q'] = q[retained]

        # Build generation hierarchy:
        # remnant_gen = max(parent_gen1, parent_gen2) + 1
        # For target gen n, merge (n-1)g with each of {1g, 2g, ..., (n-1)g}
        for target in range(2, self.max_gen + 1):
            for g2 in range(1, target):
                _merge_generations(target - 1, g2, target, append=(g2 > 1))
            if verbose:
                print(f"{target}g: {len(data[target]['m'])} BHs")

        return data

    def plot_generations(self, data, compare_data=None,
                         label=None, compare_label=None, figsize=None):
        """Paper-style figure: mass-spin scatter(s) + mass histogram.

        If compare_data is provided, creates a 3-panel figure:
        top two panels are mass-spin scatter for each dataset,
        bottom panel overlays mass histograms (bars vs step lines).
        Otherwise creates a 2-panel figure for a single dataset.

        Parameters
        ----------
        data : dict
            Output from simulate(): {gen: {'m': array, 'spin': array}}.
        compare_data : dict or None
            Second dataset for comparison (default: None).
        label : str or None
            Label for the primary dataset (default: self.kick_model).
        compare_label : str or None
            Label for the comparison dataset.
        figsize : tuple or None
            Figure size. Default scales with number of panels.

        Returns
        -------
        fig, axes : Figure and array of Axes
        """
        _MARKERS = {1: '.', 2: 's', 3: 'v', 4: '*', 5: '+'}
        _MSIZES = {1: 6, 2: 6, 3: 6, 4: 8, 5: 8}
        _ALPHAS = {1: 0.4, 2: 0.6, 3: 0.8, 4: 1.0, 5: 1.0}
        _COLORS = {1: 'C0', 2: 'C1', 3: 'C2', 4: 'C3', 5: 'C4'}

        label = label or self.kick_model
        max_gen = max(data.keys())

        def _scatter_panel(ax, d, panel_label):
            for g in range(1, max_gen + 1):
                if g not in d or len(d[g]['m']) == 0:
                    continue
                marker = _MARKERS.get(g, 'o')
                ax.plot(d[g]['m'], d[g]['spin'], marker,
                        alpha=_ALPHAS.get(g, 0.8),
                        markersize=_MSIZES.get(g, 6),
                        color=_COLORS.get(g, f'C{g-1}'))
            ax.set_ylabel(r'$\chi_{\rm BH}^{\rm retained}$')
            ax.set_xlim(left=0)
            ax.text(0.95, 0.05, panel_label, transform=ax.transAxes,
                    fontsize=14, va='bottom', ha='right', color='r')

        if compare_data is not None:
            compare_label = compare_label or 'compare'
            n_panels = 3
            if figsize is None:
                figsize = (8, 9)
            fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True,
                                     gridspec_kw={'hspace': 0,
                                                  'height_ratios': [3, 3, 4]})
            _scatter_panel(axes[0], data, label)
            _scatter_panel(axes[1], compare_data, compare_label)
            hist_ax = axes[2]

            # Mass histograms: bars for primary, step lines for compare
            all_masses = []
            for g in range(1, max_gen + 1):
                if g in data and len(data[g]['m']) > 0:
                    all_masses.extend(data[g]['m'])
                if g in compare_data and len(compare_data[g]['m']) > 0:
                    all_masses.extend(compare_data[g]['m'])
            max_mass = max(all_masses) if all_masses else 200
            max_bin = np.ceil(max_mass / 50) * 50
            bins = np.linspace(0, max_bin, 50)
            bin_width = bins[1] - bins[0]
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            # Alternate up/down: even generations up, odd generations down
            for g in range(2, max_gen + 1):
                color = _COLORS.get(g, f'C{g-1}')
                sign = 1 if g % 2 == 0 else -1

                m1 = np.asarray(data[g]['m'], dtype=float)
                m1 = m1[np.isfinite(m1)]
                if len(m1) > 1:
                    counts, _ = np.histogram(m1, bins=bins)
                    hist_ax.bar(bin_centers, sign * counts, width=bin_width,
                                color=color, alpha=0.3, align='center')

                m2 = np.asarray(compare_data[g]['m'], dtype=float)
                m2 = m2[np.isfinite(m2)]
                if len(m2) > 1:
                    counts, _ = np.histogram(m2, bins=bins)
                    hist_ax.step(np.append(bins[:-1], bins[-1]),
                                 sign * np.append(counts, counts[-1]),
                                 where='post', color=color, linewidth=3)

            hist_ax.set_yscale('symlog', linthresh=1)
            y_max = max(abs(v) for v in hist_ax.get_ylim())
            hist_ax.set_ylim(-y_max, y_max)
            hist_ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'{abs(int(x))}' if x != 0 else '0'))

            legend_gen = [Line2D([0], [0], color=_COLORS.get(g, f'C{g-1}'),
                                 linewidth=2, label=f'{g}G')
                          for g in range(2, max_gen + 1)]
            legend_model = [
                Line2D([0], [0], color='gray', linewidth=8, alpha=0.3,
                       label=label),
                Line2D([0], [0], color='gray', linewidth=3, label=compare_label),
            ]
            leg1 = hist_ax.legend(handles=legend_gen, loc='upper right',
                                  fontsize=12, frameon=False)
            hist_ax.add_artist(leg1)
            hist_ax.legend(handles=legend_model, loc='lower right',
                           fontsize=12, frameon=False)

            hist_ax.set_xlabel(r'$M_{\rm BH}^{\rm retained}$ $[M_{\odot}]$')
            hist_ax.set_ylabel('Counts')
            hist_ax.set_xlim(left=0)

        else:
            n_panels = 2
            if figsize is None:
                figsize = (8, 8)
            fig, axes = plt.subplots(n_panels, 1, figsize=figsize,
                                     gridspec_kw={'hspace': 0.3,
                                                  'height_ratios': [1, 1]})
            _scatter_panel(axes[0], data, label)
            axes[0].set_xlabel(r'$M_{\rm BH}^{\rm retained}$ $[M_{\odot}]$')

            # Mass histogram
            for g in range(1, max_gen + 1):
                if g not in data or len(data[g]['m']) == 0:
                    continue
                color = _COLORS.get(g, f'C{g-1}')
                axes[1].hist(data[g]['m'], bins=30, alpha=0.5, color=color,
                             label=f'{g}G ({len(data[g]["m"])})', density=True)
            axes[1].set_xlabel(r'$M_{\rm BH}^{\rm retained}$ $[M_{\odot}]$')
            axes[1].set_ylabel('Density')
            axes[1].legend(fontsize=10)

        # Shared scatter legend at top
        scatter_handles = [
            Line2D([0], [0], marker=_MARKERS.get(g, 'o'), color='w',
                   markerfacecolor=_COLORS.get(g, f'C{g-1}'),
                   markeredgecolor=_COLORS.get(g, f'C{g-1}'),
                   markersize=8, linestyle='None', label=f'{g}G')
            for g in range(1, max_gen + 1)
        ]
        fig.legend(handles=scatter_handles, ncol=max_gen,
                   bbox_to_anchor=(0.92, 0.99), frameon=False, fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig, axes

    def __repr__(self):
        v_esc_str = f"v_esc={self.v_esc_dist}"
        if self.evolve_v_esc:
            v_esc_str += f"*g^({self.v_esc_decay_index})"
        return (f"HierarchicalMergersInClusters("
                f"n={self.n_samples}, m=[{self.m_min},{self.m_max}], "
                f"IMF={self.imf}, chi_max={self.chi_max}, "
                f"{v_esc_str}, pairing={self.pairing}, kick={self.kick_model})")
