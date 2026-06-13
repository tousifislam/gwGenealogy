#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: hierarchical.py
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
#    Two classes:
#      HierarchicalMergersInCluster
#        — single cluster with fixed (Mcl, rh) or v_esc; 1G BHs from
#          Kroupa IMF + stellar evolution at metallicity Z; optional
#          time-varying v_esc as cluster loses mass from ejections.
#
#      HierarchicalMergersInClusterPopulation
#        — population-averaged: v_esc drawn from a distribution each
#          merger; 1G BH masses from a generic IMF (uniform/powerlaw).
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

from ..binaries.bbh_remnant import BBHRemnant, preload_kick_model
from ..binaries.bbh_spins import sample_spin_angles
from ..stellar import sample_1g_bh_masses_from_stellar_collapse
from ..hosts.star_clusters import Mcl_rh_to_vescape
from ..utils.distributions import (sample_uniform_1d, sample_powerlaw_1d,
                                   sample_gaussian_1d, sample_beta_1d,
                                   seed_legacy_rng)


# ======================================================================
# Shared helpers
# ======================================================================

_MARKERS = {1: '.', 2: 's', 3: 'v', 4: '*', 5: '+'}
_MSIZES = {1: 6, 2: 6, 3: 6, 4: 8, 5: 8}
_ALPHAS = {1: 0.4, 2: 0.6, 3: 0.8, 4: 1.0, 5: 1.0}
_COLORS = {1: 'C0', 2: 'C1', 3: 'C2', 4: 'C3', 5: 'C4'}


def _get_pair(m1_arr, m_pool, spin_pool, rng, pairing, pairing_beta):
    """Select partners using the chosen pairing model.

    Uses memory-efficient algorithms to avoid O(N x pool_N) matrices.
    """
    n = len(m1_arr)
    pool_n = len(m_pool)

    if pairing == 'random':
        idx = rng.integers(0, pool_n, size=n)
        return m_pool[idx], spin_pool[idx]

    elif pairing == 'secondary_mass_power_law':
        # Sorted inverse-CDF: O(N log pool_N) time, O(pool_N) memory.
        sort_idx = np.argsort(m_pool)
        m_sorted = m_pool[sort_idx]
        cumw = np.cumsum(m_sorted ** pairing_beta)

        cutoffs = np.searchsorted(m_sorted, m1_arr, side='left')
        bad = cutoffs == 0
        if bad.any():
            cutoffs[bad] = pool_n
        total_w = cumw[cutoffs - 1]
        u = rng.random(n) * total_w
        sampled = np.searchsorted(cumw, u)
        sampled = np.clip(sampled, 0, pool_n - 1)
        idx = sort_idx[sampled]
        return m_pool[idx], spin_pool[idx]

    elif pairing == 'total_mass_power_law':
        # Rejection sampling: O(N) memory, ~3-5 rounds to converge.
        m_max = m_pool.max()
        idx = np.empty(n, dtype=np.intp)
        remaining = np.arange(n)
        while len(remaining) > 0:
            proposals = rng.integers(0, pool_n, size=len(remaining))
            w = (m1_arr[remaining] + m_pool[proposals]) ** pairing_beta
            w_max = (m1_arr[remaining] + m_max) ** pairing_beta
            accepted = rng.random(len(remaining)) * w_max < w
            idx[remaining[accepted]] = proposals[accepted]
            remaining = remaining[~accepted]
        return m_pool[idx], spin_pool[idx]

    else:
        raise ValueError(f"Unknown pairing: {pairing}. "
                         "Choose 'random', 'secondary_mass_power_law', "
                         "or 'total_mass_power_law'.")


def _compute_remnants(m1, m2, spin1, spin2, rng, precessing,
                      mass_spin_model, kick_model):
    """Compute remnant properties for paired BH arrays via BBHRemnant."""
    m1_ord = np.maximum(m1, m2)
    m2_ord = np.minimum(m1, m2)
    a1_ord = np.where(m1 >= m2, spin1, spin2)
    a2_ord = np.where(m1 >= m2, spin2, spin1)

    n = len(m1)
    theta1, theta2, phi1, phi2 = sample_spin_angles(
        n, 'isotropic', seed=int(rng.integers(0, 2**31)))

    # Seed legacy RNG backends so that kick models with internal
    # randomness (CLZM2007 Theta angle, gwmodel flow sampling) are
    # reproducible from our controlled RNG chain. Preload first so a
    # one-time lazy model load can't perturb the seeded state.
    preload_kick_model(kick_model)
    seed_legacy_rng(rng)

    rem = BBHRemnant(m1=m1_ord, m2=m2_ord, a1=a1_ord, a2=a2_ord,
                     theta1=theta1, theta2=theta2, phi1=phi1, phi2=phi2,
                     precessing=precessing,
                     mass_spin_model=mass_spin_model,
                     kick_model=kick_model)

    q = m1_ord / m2_ord
    return rem.Mf, np.abs(rem.af), q, rem.vkick


def _scatter_panel(ax, d, panel_label, max_gen):
    """Mass-spin scatter panel used by both classes."""
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


def _plot_generations_single(data, kick_label, max_gen, figsize=None):
    """2-panel plot: scatter + histogram for a single dataset."""
    label = kick_label
    if figsize is None:
        figsize = (8, 8)
    fig, axes = plt.subplots(2, 1, figsize=figsize,
                             gridspec_kw={'hspace': 0.3,
                                          'height_ratios': [1, 1]})
    _scatter_panel(axes[0], data, label, max_gen)
    axes[0].set_xlabel(r'$M_{\rm BH}^{\rm retained}$ $[M_{\odot}]$')

    for g in range(1, max_gen + 1):
        if g not in data or len(data[g]['m']) == 0:
            continue
        color = _COLORS.get(g, f'C{g-1}')
        axes[1].hist(data[g]['m'], bins=30, alpha=0.5, color=color,
                     label=f'{g}G ({len(data[g]["m"])})', density=True)
    axes[1].set_xlabel(r'$M_{\rm BH}^{\rm retained}$ $[M_{\odot}]$')
    axes[1].set_ylabel('Density')
    axes[1].legend(fontsize=10)

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


def _plot_generations_compare(data, compare_data, label, compare_label,
                              max_gen, figsize=None):
    """3-panel plot: two scatters + overlaid histogram for comparison."""
    if figsize is None:
        figsize = (8, 9)
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'hspace': 0,
                                          'height_ratios': [3, 3, 4]})
    _scatter_panel(axes[0], data, label, max_gen)
    _scatter_panel(axes[1], compare_data, compare_label, max_gen)
    hist_ax = axes[2]

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


# ======================================================================
# HierarchicalMergersInCluster — single cluster
# ======================================================================

class HierarchicalMergersInCluster:
    """Monte Carlo simulation of hierarchical BH mergers in a single cluster.

    Supports two modes for the escape velocity and two modes for the
    1G BH mass pool:

    **Escape velocity** (provide exactly one):
      - Physical: ``Mcl`` + ``rh`` → v_esc computed via virial theorem.
      - Direct: ``v_esc`` → used as-is.
      If both ``Mcl``/``rh`` and ``v_esc`` are given, ``Mcl``/``rh`` takes
      precedence and a warning is printed.

    **1G BH masses** (priority order):
      1. ``m_pool`` (pre-built array) — used directly.
      2. ``m_min``/``m_max``/``imf`` — simple IMF sampling (no stellar
         evolution). If ``Z`` is also provided it is ignored with a warning.
      3. ``Z`` (+ optional ``stellar_model``) — Kroupa IMF → stellar
         evolution → BH remnant masses. If ``stellar_model`` is not
         specified it defaults to ``'SEVN_delayed'`` and a note is printed.

    Generation convention: remnant_gen = max(parent_gen1, parent_gen2) + 1.

    Parameters
    ----------
    Mcl : float or None
        Total cluster mass [Msun].
    rh : float or None
        Half-mass radius [pc].
    v_esc : float or None
        Cluster escape velocity [km/s]. Ignored when Mcl and rh are given.
    Z : float or None
        Metallicity for stellar evolution. Ignored when m_min/m_max/imf
        or m_pool are provided.
    stellar_model : str or None
        Stellar evolution model: 'Fryer12_delayed' or 'SEVN_delayed'.
        Defaults to 'SEVN_delayed' when Z is used.
    n_pool : int
        Number of ZAMS stars to sample for the 1G BH pool (default: 5000).
    m_pool : array or None
        Pre-built 1G BH mass pool. If provided, skips all mass generation.
    m_min, m_max : float or None
        1G BH mass range [Msun] for simple IMF sampling.
    imf : str or None
        Initial mass function when using m_min/m_max:
        'uniform' or 'powerlaw' (p(m) ~ m^imf_gamma).
    imf_gamma : float
        Power-law index for imf='powerlaw' (default: -2.5).
    chi_max : float
        Maximum natal spin magnitude for 1G BHs (default: 0.2).
    spin_dist : str
        1G spin magnitude distribution: 'uniform' (default) or 'beta'.
    beta_a, beta_b : float
        Beta distribution parameters for spin_dist='beta'
        (default: 1.4, 3.6).
    evolve_v_esc : bool
        If True, escape velocity decreases as BHs are ejected:
        v_esc(g) = v_esc_0 * (1 - cumulative_ejected_mass / Mcl)^(1/3).
        Default: False. Requires Mcl to be set.
    pairing : str
        Binary pairing model: 'random' (default),
        'secondary_mass_power_law', or 'total_mass_power_law'.
    pairing_beta : float or None
        Power-law index for pairing. None uses defaults.
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

    def __init__(self, Mcl=None, rh=None, v_esc=None,
                 Z=None, stellar_model=None, n_pool=5000,
                 m_pool=None, m_min=None, m_max=None,
                 imf=None, imf_gamma=-2.5,
                 chi_max=0.2, spin_dist='uniform', beta_a=1.4, beta_b=3.6,
                 evolve_v_esc=False,
                 pairing='random', pairing_beta=None,
                 max_gen=5, n_samples=None,
                 precessing=True, mass_spin_model=None, kick_model=None,
                 seed=None):

        # --- Escape velocity resolution ---
        if Mcl is not None and rh is not None:
            self.Mcl = float(Mcl)
            self.rh = float(rh)
            self.v_esc_0 = float(Mcl_rh_to_vescape(Mcl, rh))
            if v_esc is not None:
                print(f"[HierarchicalMergersInCluster] Mcl and rh provided — "
                      f"ignoring v_esc={v_esc}. Using v_esc={self.v_esc_0:.1f} km/s "
                      f"from (Mcl={Mcl:.1e}, rh={rh:.1f}).")
        elif v_esc is not None:
            self.Mcl = None
            self.rh = None
            self.v_esc_0 = float(v_esc)
        else:
            raise ValueError(
                "Must provide either (Mcl, rh) or v_esc for the cluster "
                "escape velocity.")

        if evolve_v_esc and self.Mcl is None:
            raise ValueError(
                "evolve_v_esc=True requires (Mcl, rh) so that cluster mass "
                "loss can be tracked. Use Mcl and rh instead of v_esc.")

        # --- 1G BH mass pool resolution ---
        self._mass_mode = None
        rng = np.random.default_rng(seed)

        if m_pool is not None:
            self.m_pool = np.asarray(m_pool, dtype=float)
            self._mass_mode = 'pool'
            self.Z = None
            self.stellar_model = None
            self.imf = None
            self.m_min = None
            self.m_max = None
            self.imf_gamma = float(imf_gamma)

        elif m_min is not None or m_max is not None or imf is not None:
            if m_min is None or m_max is None:
                raise ValueError(
                    "When using simple IMF mode, both m_min and m_max "
                    "must be provided.")
            self.m_min = float(m_min)
            self.m_max = float(m_max)
            self.imf = imf or 'uniform'
            self.imf_gamma = float(imf_gamma)
            self._mass_mode = 'imf'
            self.Z = None
            self.stellar_model = None
            if Z is not None:
                print(f"[HierarchicalMergersInCluster] m_min/m_max/imf provided "
                      f"— ignoring Z={Z}. Using {self.imf} IMF over "
                      f"[{self.m_min}, {self.m_max}] Msun.")
            seed_m = int(rng.integers(0, 2**31))
            if self.imf == 'uniform':
                self.m_pool = sample_uniform_1d(
                    n_pool, low=self.m_min, high=self.m_max, seed=seed_m)
            elif self.imf == 'powerlaw':
                self.m_pool = sample_powerlaw_1d(
                    n_pool, beta=self.imf_gamma, xmin=self.m_min,
                    xmax=self.m_max, seed=seed_m)
            else:
                raise ValueError(
                    f"Unknown imf='{self.imf}'. "
                    f"Choose 'uniform' or 'powerlaw'.")

        elif Z is not None:
            self.Z = float(Z)
            self.m_min = None
            self.m_max = None
            self.imf = None
            self.imf_gamma = float(imf_gamma)
            self._mass_mode = 'stellar'
            if stellar_model is None:
                stellar_model = 'SEVN_delayed'
                print(f"[HierarchicalMergersInCluster] No stellar_model specified "
                      f"— defaulting to '{stellar_model}'.")
            self.stellar_model = stellar_model
            self.m_pool = sample_1g_bh_masses_from_stellar_collapse(
                n_pool, Z=self.Z, model=self.stellar_model, imf='kroupa',
                m_zams_min=10.0, m_zams_max=150.0,
                seed=int(rng.integers(0, 2**31)))

        else:
            raise ValueError(
                "Must provide one of: m_pool, (m_min + m_max), or Z "
                "for the 1G BH mass pool.")

        # --- Common parameters ---
        self.chi_max = float(chi_max)
        self.spin_dist = spin_dist
        self.beta_a = float(beta_a)
        self.beta_b = float(beta_b)
        self.evolve_v_esc = evolve_v_esc
        self.pairing = pairing
        self.max_gen = max_gen
        self.precessing = precessing
        self.mass_spin_model = mass_spin_model
        self.kick_model = kick_model or ('gwmodel' if precessing else 'gwmodel_kick_q200')
        self.seed = seed

        if pairing_beta is None:
            if pairing == 'secondary_mass_power_law':
                self.pairing_beta = 6.7
            elif pairing == 'total_mass_power_law':
                self.pairing_beta = 4.0
            else:
                self.pairing_beta = 0.0
        else:
            self.pairing_beta = float(pairing_beta)

        self.n_samples = n_samples if n_samples is not None else len(self.m_pool)

        self._init_rng_state = int(rng.integers(0, 2**63))

    def _sample_1g_spins(self, n, rng):
        """Sample 1G BH spin magnitudes."""
        seed_s = int(rng.integers(0, 2**31))
        if self.spin_dist == 'uniform':
            return sample_uniform_1d(n, low=0, high=self.chi_max, seed=seed_s)
        elif self.spin_dist == 'beta':
            return sample_beta_1d(n, a=self.beta_a, b=self.beta_b, seed=seed_s)
        else:
            raise ValueError(f"Unknown spin_dist: {self.spin_dist}.")

    def simulate(self, verbose=False):
        """Run the hierarchical merger simulation across generations.

        Parameters
        ----------
        verbose : bool
            Print generation counts (default: False).

        Returns
        -------
        dict : {generation: {'m': array, 'spin': array, 'q': array}}
        """
        rng = np.random.default_rng(self._init_rng_state)

        data = {g: {'m': np.array([]), 'spin': np.array([]),
                     'q': np.array([])}
                for g in range(1, self.max_gen + 1)}

        masses = rng.choice(self.m_pool, self.n_samples, replace=True)
        spins = self._sample_1g_spins(self.n_samples, rng)
        data[1]['m'] = masses
        data[1]['spin'] = spins

        v_esc = self.v_esc_0
        cumulative_ejected_mass = 0.0

        if verbose:
            parts = [f"1g: {len(data[1]['m'])} BHs ("]
            if self.Mcl is not None:
                parts.append(f"Mcl={self.Mcl:.1e} Msun, rh={self.rh:.1f} pc, ")
            parts.append(f"v_esc={v_esc:.1f} km/s")
            if self._mass_mode == 'stellar':
                parts.append(f", Z={self.Z}, model={self.stellar_model}")
            elif self._mass_mode == 'imf':
                parts.append(f", {self.imf} IMF [{self.m_min:.0f},{self.m_max:.0f}] Msun")
            parts.append(")")
            print("".join(parts))

        def _merge_channel(gen1, gen2, n_chan, target_gen, append=False):
            nonlocal v_esc, cumulative_ejected_mass

            n1 = len(data[gen1]['m'])
            n2 = len(data[gen2]['m'])
            if n1 == 0 or n2 == 0 or n_chan == 0:
                return

            idx1 = rng.choice(n1, n_chan, replace=True)
            m1 = data[gen1]['m'][idx1]
            spin1 = data[gen1]['spin'][idx1]
            m2, spin2 = _get_pair(
                m1, data[gen2]['m'], data[gen2]['spin'], rng,
                self.pairing, self.pairing_beta)

            mf, chif, q, vkick = _compute_remnants(
                m1, m2, spin1, spin2, rng, self.precessing,
                self.mass_spin_model, self.kick_model)

            retained = vkick < v_esc

            if self.evolve_v_esc and self.Mcl is not None:
                ejected_mass = mf[~retained].sum()
                cumulative_ejected_mass += ejected_mass
                frac_remaining = max(1.0 - cumulative_ejected_mass / self.Mcl, 0.01)
                v_esc = self.v_esc_0 * frac_remaining**(1.0 / 3.0)

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

        # Merger budget logic:
        # - Each BH in the highest-generation parent pool merges at most
        #   once across all channels, so the total number of mergers is
        #   capped at the size of that pool (n_hi).
        # - For 2g (1g+1g), both parents come from the same pool, so
        #   each merger consumes two BHs and the cap is n_hi // 2.
        # - The budget is split across channels proportional to the
        #   secondary pool sizes (a BH is more likely to encounter a
        #   partner from a larger pool).
        for target in range(2, self.max_gen + 1):
            gen_hi = target - 1
            n_hi = len(data[gen_hi]['m'])
            channels = list(range(1, target))
            pool_sizes = np.array([len(data[g2]['m']) for g2 in channels],
                                  dtype=float)
            if pool_sizes.sum() == 0 or n_hi == 0:
                if verbose:
                    v_info = f", v_esc={v_esc:.1f} km/s" if self.evolve_v_esc else ""
                    print(f"{target}g: 0 BHs{v_info}")
                continue
            only_self = (len(channels) == 1 and channels[0] == gen_hi)
            n_total = min(n_hi // 2 if only_self else n_hi,
                          self.n_samples)
            weights = pool_sizes / pool_sizes.sum()
            n_per_chan = np.floor(weights * n_total).astype(int)
            n_per_chan[-1] = max(0, n_total - n_per_chan[:-1].sum())
            for i, g2 in enumerate(channels):
                _merge_channel(gen_hi, g2, n_per_chan[i], target,
                               append=(i > 0))
            if verbose:
                n_ret = len(data[target]['m'])
                v_info = f", v_esc={v_esc:.1f} km/s" if self.evolve_v_esc else ""
                if n_ret > 0:
                    m_med = np.median(data[target]['m'])
                    chi_med = np.median(data[target]['spin'])
                    print(f"{target}g: {n_ret} retained{v_info}, "
                          f"m_med={m_med:.1f} Msun, chi_med={chi_med:.2f}")
                else:
                    print(f"{target}g: 0 BHs{v_info}")

        return data

    def plot_generations(self, data, compare_data=None,
                         label=None, compare_label=None, figsize=None):
        """Paper-style figure: mass-spin scatter(s) + mass histogram.

        Parameters
        ----------
        data : dict
            Output from simulate().
        compare_data : dict or None
            Second dataset for comparison.
        label : str or None
            Label for the primary dataset (default: self.kick_model).
        compare_label : str or None
            Label for the comparison dataset.
        figsize : tuple or None
            Figure size.

        Returns
        -------
        fig, axes
        """
        label = label or self.kick_model
        max_gen = max(data.keys())
        if compare_data is not None:
            compare_label = compare_label or 'compare'
            return _plot_generations_compare(
                data, compare_data, label, compare_label, max_gen, figsize)
        else:
            return _plot_generations_single(data, label, max_gen, figsize)

    def __repr__(self):
        v_esc_str = f"v_esc={self.v_esc_0:.1f} km/s"
        if self.evolve_v_esc:
            v_esc_str += " (evolving)"
        parts = ["HierarchicalMergersInCluster("]
        if self.Mcl is not None:
            parts.append(f"Mcl={self.Mcl:.1e}, rh={self.rh:.1f}, ")
        parts.append(f"{v_esc_str}, ")
        if self._mass_mode == 'stellar':
            parts.append(f"Z={self.Z}, model={self.stellar_model}, ")
        elif self._mass_mode == 'imf':
            parts.append(f"{self.imf} IMF [{self.m_min:.0f},{self.m_max:.0f}], ")
        parts.append(f"kick={self.kick_model})")
        return "".join(parts)


# ======================================================================
# HierarchicalMergersInClusterPopulation — population-averaged
# ======================================================================

class HierarchicalMergersInClusterPopulation:
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

        def _merge_channel(gen1, gen2, n_chan, target_gen, append=False):
            n1 = len(data[gen1]['m'])
            n2 = len(data[gen2]['m'])
            if n1 == 0 or n2 == 0 or n_chan == 0:
                return

            idx1 = rng.choice(n1, n_chan, replace=True)
            m1 = data[gen1]['m'][idx1]
            spin1 = data[gen1]['spin'][idx1]
            m2, spin2 = _get_pair(
                m1, data[gen2]['m'], data[gen2]['spin'], rng,
                self.pairing, self.pairing_beta)
            v_esc = self._sample_v_esc(n_chan, rng, generation=target_gen)

            mf, chif, q, vkick = _compute_remnants(
                m1, m2, spin1, spin2, rng, self.precessing,
                self.mass_spin_model, self.kick_model)
            retained = vkick < v_esc

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

        # Merger budget logic:
        # - Each BH in the highest-generation parent pool merges at most
        #   once across all channels, so the total number of mergers is
        #   capped at the size of that pool (n_hi).
        # - For 2g (1g+1g), both parents come from the same pool, so
        #   each merger consumes two BHs and the cap is n_hi // 2.
        # - The budget is split across channels proportional to the
        #   secondary pool sizes (a BH is more likely to encounter a
        #   partner from a larger pool).
        for target in range(2, self.max_gen + 1):
            gen_hi = target - 1
            n_hi = len(data[gen_hi]['m'])
            channels = list(range(1, target))
            pool_sizes = np.array([len(data[g2]['m']) for g2 in channels],
                                  dtype=float)
            if pool_sizes.sum() == 0 or n_hi == 0:
                if verbose:
                    print(f"{target}g: 0 BHs")
                continue
            only_self = (len(channels) == 1 and channels[0] == gen_hi)
            n_total = min(n_hi // 2 if only_self else n_hi,
                          self.n_samples)
            weights = pool_sizes / pool_sizes.sum()
            n_per_chan = np.floor(weights * n_total).astype(int)
            n_per_chan[-1] = max(0, n_total - n_per_chan[:-1].sum())
            for i, g2 in enumerate(channels):
                _merge_channel(gen_hi, g2, n_per_chan[i], target,
                               append=(i > 0))
            if verbose:
                n_ret = len(data[target]['m'])
                if n_ret > 0:
                    m_med = np.median(data[target]['m'])
                    chi_med = np.median(data[target]['spin'])
                    print(f"{target}g: {n_ret} retained, "
                          f"m_med={m_med:.1f} Msun, chi_med={chi_med:.2f}")
                else:
                    print(f"{target}g: 0 BHs")

        return data

    def plot_generations(self, data, compare_data=None,
                         label=None, compare_label=None, figsize=None):
        """Paper-style figure: mass-spin scatter(s) + mass histogram.

        Parameters
        ----------
        data : dict
            Output from simulate().
        compare_data : dict or None
            Second dataset for comparison.
        label : str or None
            Label for the primary dataset (default: self.kick_model).
        compare_label : str or None
            Label for the comparison dataset.
        figsize : tuple or None
            Figure size.

        Returns
        -------
        fig, axes
        """
        label = label or self.kick_model
        max_gen = max(data.keys())
        if compare_data is not None:
            compare_label = compare_label or 'compare'
            return _plot_generations_compare(
                data, compare_data, label, compare_label, max_gen, figsize)
        else:
            return _plot_generations_single(data, label, max_gen, figsize)

    def __repr__(self):
        v_esc_str = f"v_esc={self.v_esc_dist}"
        if self.evolve_v_esc:
            v_esc_str += f"*g^({self.v_esc_decay_index})"
        return (f"HierarchicalMergersInClusterPopulation("
                f"n={self.n_samples}, m=[{self.m_min},{self.m_max}], "
                f"IMF={self.imf}, chi_max={self.chi_max}, "
                f"{v_esc_str}, pairing={self.pairing}, kick={self.kick_model})")
