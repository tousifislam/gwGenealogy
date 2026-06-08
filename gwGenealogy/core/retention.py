#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: retention.py
#
#    Compute retention probability P_ret for 1G+1G BBH mergers on a
#    (q, chi_max, v_esc) grid. For each (q, chi_max) grid point, samples
#    spin magnitudes uniformly in [0, chi_max] and computes kicks using
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

from ..binaries.bbh_remnant import BBHRemnant
from ..binaries.bbh_spins import sample_spin_angles
from ..utils.distributions import sample_uniform_1d


class BBHRetentionProbability1G1G:
    """Retention probability grid for 1G+1G BBH mergers.

    Pre-computes shared isotropic spin angles, then sweeps over a
    (q, chi_max) grid computing P_ret = fraction with kick < v_esc
    for one or more kick models and v_esc values simultaneously.

    Parameters
    ----------
    q_values : array-like
        Mass ratio values (q >= 1).
    chi_max_values : array-like
        Maximum spin magnitude values for uniform spin sampling.
    v_esc_values : array-like
        Escape velocity values [km/s].
    kick_models : list of str
        Kick model names supported by BBHRemnant:
        'hlz', 'gwmodel', 'gwmodel_kick_q200', 'sur7dq4remnant', etc.
        Default: ['hlz', 'gwmodel'].
    n_samples : int
        Number of spin samples per (q, chi_max) grid point (default: 10000).
    precessing : bool
        If True (default), use precessing kick models.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, q_values, chi_max_values, v_esc_values,
                 kick_models=None, n_samples=10000, precessing=True,
                 seed=None):
        self.q_values = np.atleast_1d(np.asarray(q_values, dtype=float))
        self.chi_max_values = np.atleast_1d(np.asarray(chi_max_values, dtype=float))
        self.v_esc_values = np.atleast_1d(np.asarray(v_esc_values, dtype=float))
        self.kick_models = kick_models or (['hlz', 'gwmodel'] if precessing
                                           else ['hlz', 'gwmodel_kick_q200'])
        self.n_samples = n_samples
        self.precessing = precessing
        self.seed = seed

        self._kick_fns = {}
        self._setup_kick_functions()

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
                seed_s1 = int(rng.integers(0, 2**31))
                seed_s2 = int(rng.integers(0, 2**31))
                a1 = sample_uniform_1d(n, low=0, high=chi_max, seed=seed_s1)
                a2 = sample_uniform_1d(n, low=0, high=chi_max, seed=seed_s2)

                # Same spins + angles evaluated by each kick model
                for model in self.kick_models:
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
        return (f"BBHRetentionProbability1G1G("
                f"q=[{self.q_values[0]:.1f},{self.q_values[-1]:.1f}]x{len(self.q_values)}, "
                f"chi=[{self.chi_max_values[0]:.2f},{self.chi_max_values[-1]:.2f}]x{len(self.chi_max_values)}, "
                f"v_esc={list(self.v_esc_values)}, "
                f"models={self.kick_models}, n={self.n_samples})")
