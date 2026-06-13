#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: retention.py
#
#    Compute retention probability P_ret for BBH mergers on a
#    (q, chi_max, v_esc) grid. For each (q, chi_max) grid point, samples
#    spin magnitudes from a chosen distribution and computes kicks using
#    one or more kick models. Isotropic spin angles are pre-computed once
#    and shared across the grid for a clean model-to-model comparison.
#
#    Implements the retention probability computation from Section III.A /
#    Figure 2 of Islam, Wadekar & Kritos (2026, arXiv:2603.10170).
#
#    AUTHOR: Tousif Islam
#    CREATED: 06-08-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import matplotlib.pyplot as plt

from ..binaries.bbh_remnant import BBHRemnant, preload_kick_model
from ..binaries.bbh_spins import sample_spin_magnitudes, sample_spin_angles
from ..utils.distributions import (sample_uniform_1d, sample_beta_1d,
                                   sample_powerlaw_1d, seed_legacy_rng)


class BBHRetentionProbabilityOverChiq:
    """Retention probability grid for BBH mergers over the (chi, q) plane.

    Pre-computes shared isotropic spin angles, then sweeps over a
    (q, chi_max) grid computing P_ret = fraction with kick < v_esc
    for one or more kick models and v_esc values simultaneously.

    Parameters
    ----------
    q_values : array-like
        Mass ratio values (q >= 1).
    chi_max_values : array-like
        Maximum spin magnitude values. For spin_dist='uniform', spins
        are drawn from U(0, chi_max). For spin_dist='beta', chi_max
        scales the Beta distribution so spins lie in [0, chi_max].
    v_esc_values : array-like
        Escape velocity values [km/s].
    kick_models : list of str
        Kick model names supported by BBHRemnant:
        'hlz', 'gwmodel', 'gwmodel_kick_q200', 'sur7dq4remnant', etc.
        Default: ['hlz', 'gwmodel'].
    n_samples : int
        Number of spin samples per (q, chi_max) grid point (default: 10000).
    spin_dist : str
        Spin magnitude distribution: 'uniform' (default) or 'beta'.
    beta_a, beta_b : float
        Beta distribution shape parameters (default: 1.4, 3.6).
        Only used when spin_dist='beta'.
    precessing : bool
        If True (default), use precessing kick models.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, q_values, chi_max_values, v_esc_values,
                 kick_models=None, n_samples=10000,
                 spin_dist='uniform', beta_a=1.4, beta_b=3.6,
                 precessing=True, seed=None):
        self.q_values = np.atleast_1d(np.asarray(q_values, dtype=float))
        self.chi_max_values = np.atleast_1d(np.asarray(chi_max_values, dtype=float))
        self.v_esc_values = np.atleast_1d(np.asarray(v_esc_values, dtype=float))
        self.kick_models = kick_models or (['hlz', 'gwmodel'] if precessing
                                           else ['hlz', 'gwmodel_kick_q200'])
        self.n_samples = n_samples
        self.spin_dist = spin_dist
        self.beta_a = float(beta_a)
        self.beta_b = float(beta_b)
        self.precessing = precessing
        self.seed = seed

        self._kick_fns = {}
        self._setup_kick_functions()

    def _sample_spins(self, n, chi_max, seed):
        """Sample spin magnitudes from the chosen distribution."""
        a1, a2 = sample_spin_magnitudes(
            n, chi_min=0, chi_max=chi_max,
            spin_magnitude=self.spin_dist,
            beta_a=self.beta_a, beta_b=self.beta_b, seed=seed)
        return a1, a2

    def _setup_kick_functions(self):
        """Build callable kick functions for each requested model."""
        for model in self.kick_models:
            def _make_kick_fn(kick_model):
                def kick_fn(q, a1, a2, theta1, theta2, phi1, phi2):
                    rem = BBHRemnant(
                        m1=q * np.ones_like(a1), m2=np.ones_like(a1),
                        a1=a1, a2=a2,
                        theta1=theta1, theta2=theta2, phi1=phi1, phi2=phi2,
                        precessing=self.precessing,
                        kick_model=kick_model)
                    return rem.vkick
                return kick_fn
            self._kick_fns[model] = _make_kick_fn(model)

    def compute(self, verbose=False):
        """Run the retention probability grid computation.

        Returns
        -------
        dict with keys:
            'q_values' : array
            'chi_max_values' : array
            'v_esc_values' : array
            'p_ret' : dict of {model_name: 3D array (n_q, n_chi, n_vesc)}
        """
        rng = np.random.default_rng(self.seed)
        n = self.n_samples
        nq = len(self.q_values)
        nchi = len(self.chi_max_values)
        nvesc = len(self.v_esc_values)

        # Shared angles across all grid points for clean model comparison
        theta1, theta2, phi1, phi2 = sample_spin_angles(
            n, 'isotropic', seed=int(rng.integers(0, 2**31)))

        p_ret = {model: np.zeros((nq, nchi, nvesc))
                 for model in self.kick_models}

        total = nq * nchi
        count = 0

        for i, q in enumerate(self.q_values):
            for j, chi_max in enumerate(self.chi_max_values):
                seed_s = int(rng.integers(0, 2**31))
                a1, a2 = self._sample_spins(n, chi_max, seed=seed_s)

                # One legacy seed per grid point, re-applied before each
                # model so kick models with internal randomness (CLZM2007
                # Theta, gwmodel flow) see the identical realization — a
                # clean model-to-model comparison on the same spins+angles.
                legacy_seed = int(rng.integers(0, 2**31))

                # Same spins + angles evaluated by each kick model
                for model in self.kick_models:
                    preload_kick_model(model)
                    seed_legacy_rng(legacy_seed)
                    vkick = self._kick_fns[model](
                        q, a1, a2, theta1, theta2, phi1, phi2)
                    for k, v_esc in enumerate(self.v_esc_values):
                        p_ret[model][i, j, k] = (vkick < v_esc).mean()

                count += 1
                if verbose and count % max(1, total // 10) == 0:
                    print(f"  {count}/{total} grid points done")

        self.results = {
            'q_values': self.q_values,
            'chi_max_values': self.chi_max_values,
            'v_esc_values': self.v_esc_values,
            'p_ret': p_ret,
        }
        return self.results

    def plot_heatmap(self, model, v_esc, ax=None, cmap='plasma', vmin=0, vmax=1):
        """Plot a single P_ret heatmap for one model and v_esc value.

        Parameters
        ----------
        model : str
            Kick model name (must be in self.kick_models).
        v_esc : float
            Escape velocity value (must be in self.v_esc_values).
        ax : matplotlib.axes.Axes or None
            Axes to plot on. If None, creates a new figure.
        cmap : str
            Colormap (default: 'plasma').
        vmin, vmax : float
            Colorbar limits (default: 0-1).

        Returns
        -------
        fig, ax, im : Figure, Axes, and ScalarMappable
        """
        if not hasattr(self, 'results'):
            raise RuntimeError("Call .compute() before plotting.")

        k = np.argmin(np.abs(self.v_esc_values - v_esc))
        p_ret_2d = self.results['p_ret'][model][:, :, k]

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        else:
            fig = ax.figure

        im = ax.pcolormesh(self.q_values, self.chi_max_values, p_ret_2d.T,
                           cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        ax.set_xlabel(r'$q$')
        ax.set_ylabel(r'$\chi_{\rm max}$')
        ax.set_title(f'{model}, $v_{{\\rm esc}}$={self.v_esc_values[k]:.0f} km/s')

        return fig, ax, im

    def plot_heatmap_all_vesc(self, cmap='plasma', vmin=0, vmax=1, figsize=None):
        """Multi-panel P_ret heatmap: v_esc rows x model columns.

        Reproduces Figure 2 layout from Islam, Wadekar & Kritos (2026).

        Parameters
        ----------
        cmap : str
            Colormap (default: 'plasma').
        vmin, vmax : float
            Colorbar limits (default: 0-1).
        figsize : tuple or None
            Figure size. Default scales with number of panels.

        Returns
        -------
        fig, axes : Figure and 2D array of Axes
        """
        if not hasattr(self, 'results'):
            raise RuntimeError("Call .compute() before plotting.")

        nrows = len(self.v_esc_values)
        ncols = len(self.kick_models)
        if figsize is None:
            figsize = (5 * ncols + 1, 3 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)

        for i, v_esc in enumerate(self.v_esc_values):
            for j, model in enumerate(self.kick_models):
                ax = axes[i, j]
                p_ret_2d = self.results['p_ret'][model][:, :, i]
                im = ax.pcolormesh(self.q_values, self.chi_max_values,
                                   p_ret_2d.T, cmap=cmap, vmin=vmin,
                                   vmax=vmax, shading='auto')

                ax.text(0.05, 0.95,
                        f'$v_{{\\rm esc}}$={v_esc:.0f} km/s',
                        transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', color='w')
                ax.text(0.75, 0.95, model,
                        transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', color='w')

                if i == nrows - 1:
                    ax.set_xlabel(r'$q$')
                if j == 0:
                    ax.set_ylabel(r'$\chi_{\rm max}$')

        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(r'$p_{\rm ret}$')

        plt.tight_layout(rect=[0, 0, 0.88, 1])
        return fig, axes

    def __repr__(self):
        spin_str = self.spin_dist
        if self.spin_dist == 'beta':
            spin_str = f"beta({self.beta_a},{self.beta_b})"
        return (f"BBHRetentionProbabilityOverChiq("
                f"q=[{self.q_values[0]:.1f},{self.q_values[-1]:.1f}]x{len(self.q_values)}, "
                f"chi=[{self.chi_max_values[0]:.2f},{self.chi_max_values[-1]:.2f}]x{len(self.chi_max_values)}, "
                f"v_esc={list(self.v_esc_values)}, "
                f"models={self.kick_models}, spin={spin_str}, n={self.n_samples})")


class BBHRetentionProbabilities:
    """Population-level BBH retention probabilities.

    Samples a BBH population from specified distributions of mass ratio,
    spin magnitudes, and spin orientations, then computes kick velocities
    and retention probabilities as a function of escape velocity.

    Unlike :class:`BBHRetentionProbabilityOverChiq` which sweeps a (q, chi)
    grid, this class draws from distributions and returns P_ret(v_esc)
    curves plus the raw kick velocity samples.

    Parameters
    ----------
    v_esc_values : array-like
        Escape velocity values [km/s] at which to evaluate P_ret.
    kick_models : list of str or None
        Kick model names supported by BBHRemnant.
        Default: ['hlz', 'gwmodel'].
    n_samples : int
        Number of BBH samples to draw (default: 5000).
    q_min, q_max : float
        Mass ratio range (default: 1.0-10.0).
    q_dist : str
        Mass ratio distribution: 'uniform' (default), 'powerlaw',
        or 'beta'.
    q_power : float
        Power-law index for q_dist='powerlaw' (default: -1.0).
    q_beta_a, q_beta_b : float
        Beta distribution parameters for q_dist='beta'
        (default: 2.0, 5.0). The raw Beta sample in [0,1] is rescaled
        to [q_min, q_max].
    a_min, a_max : float
        Spin magnitude range (default: 0.0-1.0).
    spin_dist : str
        Spin magnitude distribution: 'uniform' (default) or 'beta'.
    spin_beta_a, spin_beta_b : float
        Beta distribution parameters for spin_dist='beta'
        (default: 1.4, 3.6). The raw Beta sample in [0,1] is rescaled
        to [a_min, a_max].
    spin_angles : str
        Spin orientation distribution: 'isotropic' (default),
        'uniform', or 'beta'.
    tilt_beta_a, tilt_beta_b : float or None
        Beta parameters for spin_angles='beta' (preferentially aligned).
    precessing : bool
        If True (default), use precessing kick models.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, v_esc_values, kick_models=None, n_samples=5000,
                 q_min=1.0, q_max=10.0, q_dist='uniform',
                 q_power=-1.0, q_beta_a=2.0, q_beta_b=5.0,
                 a_min=0.0, a_max=1.0,
                 spin_dist='uniform', spin_beta_a=1.4, spin_beta_b=3.6,
                 spin_angles='isotropic',
                 tilt_beta_a=None, tilt_beta_b=None,
                 precessing=True, seed=None):
        self.v_esc_values = np.atleast_1d(np.asarray(v_esc_values, dtype=float))
        self.kick_models = kick_models or (['hlz', 'gwmodel'] if precessing
                                           else ['hlz', 'gwmodel_kick_q200'])
        self.n_samples = n_samples
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.q_dist = q_dist
        self.q_power = float(q_power)
        self.q_beta_a = float(q_beta_a)
        self.q_beta_b = float(q_beta_b)
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.spin_dist = spin_dist
        self.spin_beta_a = float(spin_beta_a)
        self.spin_beta_b = float(spin_beta_b)
        self.spin_angles = spin_angles
        self.tilt_beta_a = tilt_beta_a
        self.tilt_beta_b = tilt_beta_b
        self.precessing = precessing
        self.seed = seed

    def _sample_q(self, n, rng):
        """Sample mass ratios from the chosen distribution."""
        seed = int(rng.integers(0, 2**31))
        if self.q_dist == 'uniform':
            return sample_uniform_1d(n, low=self.q_min, high=self.q_max,
                                     seed=seed)
        elif self.q_dist == 'powerlaw':
            return sample_powerlaw_1d(n, beta=self.q_power,
                                      xmin=self.q_min, xmax=self.q_max,
                                      seed=seed)
        elif self.q_dist == 'beta':
            raw = sample_beta_1d(n, a=self.q_beta_a, b=self.q_beta_b,
                                 seed=seed)
            return self.q_min + raw * (self.q_max - self.q_min)
        else:
            raise ValueError(
                f"Unknown q_dist='{self.q_dist}'. "
                f"Choose 'uniform', 'powerlaw', or 'beta'.")

    def _sample_spins(self, n, rng):
        """Sample spin magnitudes from the chosen distribution."""
        seed = int(rng.integers(0, 2**31))
        return sample_spin_magnitudes(
            n, chi_min=self.a_min, chi_max=self.a_max,
            spin_magnitude=self.spin_dist,
            beta_a=self.spin_beta_a, beta_b=self.spin_beta_b, seed=seed)

    def compute(self, verbose=False):
        """Sample the BBH population and compute kick velocities.

        Returns
        -------
        dict with keys:
            'q' : array of sampled mass ratios
            'a1', 'a2' : arrays of sampled spin magnitudes
            'v_esc_values' : array
            'kicks' : dict of {model: array of kick velocities}
            'p_ret' : dict of {model: array of P_ret per v_esc}
        """
        rng = np.random.default_rng(self.seed)
        n = self.n_samples

        q = self._sample_q(n, rng)
        a1, a2 = self._sample_spins(n, rng)

        theta1, theta2, phi1, phi2 = sample_spin_angles(
            n, self.spin_angles,
            tilt_beta_a=self.tilt_beta_a, tilt_beta_b=self.tilt_beta_b,
            seed=int(rng.integers(0, 2**31)))

        kicks = {}
        p_ret = {}

        for model in self.kick_models:
            if verbose:
                print(f"  Computing kicks for {model}...")
            # Seed legacy backends so kick models with internal randomness
            # (CLZM2007 Theta, gwmodel flow) are reproducible from `seed`.
            # Preload first so a one-time lazy load can't perturb the seed.
            preload_kick_model(model)
            seed_legacy_rng(rng)
            rem = BBHRemnant(
                m1=q * np.ones(n), m2=np.ones(n),
                a1=a1, a2=a2,
                theta1=theta1, theta2=theta2, phi1=phi1, phi2=phi2,
                precessing=self.precessing,
                kick_model=model)
            vk = rem.vkick
            kicks[model] = vk
            p_ret[model] = np.array([(vk < v).mean()
                                     for v in self.v_esc_values])

        self.results = {
            'q': q,
            'a1': a1, 'a2': a2,
            'v_esc_values': self.v_esc_values,
            'kicks': kicks,
            'p_ret': p_ret,
        }
        return self.results

    def plot_kicks(self, bins=50, ax=None, log=True):
        """Plot kick velocity distributions for all models.

        Parameters
        ----------
        bins : int
            Number of histogram bins (default: 50).
        ax : matplotlib.axes.Axes or None
            Axes to plot on. If None, creates a new figure.
        log : bool
            Log scale on x-axis (default: True).

        Returns
        -------
        fig, ax
        """
        if not hasattr(self, 'results'):
            raise RuntimeError("Call .compute() before plotting.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        for model in self.kick_models:
            vk = self.results['kicks'][model]
            if log:
                vk_pos = vk[vk > 0]
                ax.hist(np.log10(vk_pos), bins=bins, alpha=0.5,
                        density=True, label=model)
                ax.set_xlabel(r'$\log_{10}(v_{\rm kick}$ [km/s]$)$')
            else:
                ax.hist(vk, bins=bins, alpha=0.5, density=True,
                        label=model)
                ax.set_xlabel(r'$v_{\rm kick}$ [km/s]')

        ax.set_ylabel('Density')
        ax.legend()
        return fig, ax

    def plot_retention(self, ax=None):
        """Plot P_ret(v_esc) curves for all models.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Axes to plot on. If None, creates a new figure.

        Returns
        -------
        fig, ax
        """
        if not hasattr(self, 'results'):
            raise RuntimeError("Call .compute() before plotting.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        for model in self.kick_models:
            ax.plot(self.v_esc_values, self.results['p_ret'][model],
                    'o-', label=model)

        ax.set_xlabel(r'$v_{\rm esc}$ [km/s]')
        ax.set_ylabel(r'$P_{\rm ret}$')
        ax.set_ylim(-0.02, 1.02)
        ax.legend()
        return fig, ax

    def __repr__(self):
        q_str = f"q~{self.q_dist}[{self.q_min:.0f},{self.q_max:.0f}]"
        spin_str = f"a~{self.spin_dist}[{self.a_min:.1f},{self.a_max:.1f}]"
        if self.spin_dist == 'beta':
            spin_str += f"(B={self.spin_beta_a},{self.spin_beta_b})"
        return (f"BBHRetentionProbabilities("
                f"{q_str}, {spin_str}, "
                f"angles={self.spin_angles}, "
                f"models={self.kick_models}, n={self.n_samples})")
