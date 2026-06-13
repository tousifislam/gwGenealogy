#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: progenitor.py
#
#    Invert the IW2025 precessing-kick flow: infer the progenitor binary
#    (mass ratio q and spin magnitudes a1, a2) from one or more observed
#    recoil kick velocities.
#
#    The flow models the conditional density p(v_kick | q, a1, a2) (spin
#    angles marginalized internally), exposed via flow.log_prob. We invert it
#    with Bayes for each observed kick:
#
#        p(q, a1, a2 | d) ~ [ INT p(d|v) p(v|q,a1,a2) dv ] * prior(q,a1,a2)
#
#    using importance sampling over a prior pool, optionally marginalizing a
#    (possibly asymmetric) Gaussian measurement uncertainty on each kick.
#
#    AUTHOR: Tousif Islam
#    CREATED: 06-13-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

from ..binaries.bbh_remnant import _get_flow_model
from ..utils.distributions import (sample_uniform_1d, sample_loguniform_1d,
                                    sample_powerlaw_1d, sample_beta_1d)


class KickToProgenitor:
    """Infer progenitor (q, a1, a2) posteriors from observed recoil kicks.

    For each kick in ``v_kicks``, performs Bayesian importance sampling using
    the IW2025 precessing-kick flow density p(v_kick | q, a1, a2) as the
    likelihood, over a prior pool of progenitors. A (possibly asymmetric)
    Gaussian measurement uncertainty can be marginalized per kick.

    The prior for each parameter is specified either by a range + distribution
    (``*_min``, ``*_max``, ``*_distribution``) or by a pre-built sample array
    (``*_array``); an array, when given, takes precedence and sets the prior
    pool size.

    Parameters
    ----------
    v_kicks : float or array-like
        Observed recoil kick magnitude(s) [km/s]. Scalar is promoted to a
        length-1 array.
    q_min, q_max : float
        Mass-ratio prior range (q >= 1). Default 1, 20.
    q_distribution : str
        'uniform' (default), 'loguniform', 'powerlaw', or 'beta'.
    q_params : dict or None
        Shape parameters for the distribution: {'beta': index} for 'powerlaw',
        {'a': .., 'b': ..} for 'beta'. Ignored for uniform/loguniform.
    q_array : array-like or None
        Pre-built prior samples for q; overrides the range/distribution.
    a1_min, a1_max, a1_distribution, a1_params, a1_array : ...
        Same for the primary spin magnitude (default range 0-1).
    a2_min, a2_max, a2_distribution, a2_params, a2_array : ...
        Same for the secondary spin magnitude (default range 0-1).
    sigma : float, array-like, or None
        Symmetric Gaussian measurement uncertainty on each kick [km/s].
        Scalar applies to all kicks; array must match len(v_kicks).
        None (default) treats kicks as exact (delta-function likelihood).
    sigma_lo, sigma_hi : float, array-like, or None
        Asymmetric (split-normal) measurement uncertainty: width below / above
        the central value. If either is given it overrides ``sigma``; a missing
        side defaults to the other.
    n_prior : int
        Prior pool size when sampling from distributions (default: 200000).
    n_posterior : int
        Number of posterior samples drawn per kick (default: 20000).
    n_grid : int
        Quadrature points for marginalizing the measurement error (default 51).
        Unused when the kicks are treated as exact.
    seed : int or None
        Random seed for reproducibility.

    Notes
    -----
    The flow is calibrated for q up to ~14; wider priors extrapolate. Watch the
    effective sample size (``ess``) — kicks in the far tail of the model thin
    the prior pool and need a larger ``n_prior``.
    """

    def __init__(self, v_kicks,
                 q_min=1.0, q_max=20.0, q_distribution='uniform',
                 q_params=None, q_array=None,
                 a1_min=0.0, a1_max=1.0, a1_distribution='uniform',
                 a1_params=None, a1_array=None,
                 a2_min=0.0, a2_max=1.0, a2_distribution='uniform',
                 a2_params=None, a2_array=None,
                 sigma=None, sigma_lo=None, sigma_hi=None,
                 n_prior=200000, n_posterior=20000, n_grid=51, seed=None):

        self.v_kicks = np.atleast_1d(np.asarray(v_kicks, dtype=float))
        self.n_kicks = len(self.v_kicks)
        self.n_posterior = int(n_posterior)
        self.n_grid = int(n_grid)
        self.seed = seed

        self._q_spec = (q_min, q_max, q_distribution, q_params, q_array)
        self._a1_spec = (a1_min, a1_max, a1_distribution, a1_params, a1_array)
        self._a2_spec = (a2_min, a2_max, a2_distribution, a2_params, a2_array)

        # Resolve prior pool size: a supplied array fixes it (and all arrays
        # must agree); otherwise use n_prior.
        arr_lens = {len(np.asarray(a)) for a in (q_array, a1_array, a2_array)
                    if a is not None}
        if len(arr_lens) > 1:
            raise ValueError("q_array, a1_array, a2_array must have equal length.")
        self.n_prior = arr_lens.pop() if arr_lens else int(n_prior)

        # Resolve the per-kick measurement model -> sigma_lo/sigma_hi arrays,
        # or None for exact (delta) kicks.
        self.sigma_lo, self.sigma_hi = self._resolve_sigma(sigma, sigma_lo, sigma_hi)

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------
    def _resolve_sigma(self, sigma, sigma_lo, sigma_hi):
        if sigma_lo is None and sigma_hi is None and sigma is None:
            return None, None
        if sigma_lo is not None or sigma_hi is not None:
            lo = sigma_lo if sigma_lo is not None else sigma_hi
            hi = sigma_hi if sigma_hi is not None else sigma_lo
        else:
            lo = hi = sigma
        lo = np.broadcast_to(np.asarray(lo, dtype=float), (self.n_kicks,)).copy()
        hi = np.broadcast_to(np.asarray(hi, dtype=float), (self.n_kicks,)).copy()
        return lo, hi

    def _sample_prior(self, spec, rng):
        lo, hi, dist, params, array = spec
        if array is not None:
            return np.asarray(array, dtype=float)
        params = params or {}
        seed = int(rng.integers(0, 2**31))
        if dist == 'uniform':
            return sample_uniform_1d(self.n_prior, low=lo, high=hi, seed=seed)
        if dist == 'loguniform':
            return sample_loguniform_1d(self.n_prior, low=lo, high=hi, seed=seed)
        if dist == 'powerlaw':
            return sample_powerlaw_1d(self.n_prior, beta=params.get('beta', -1.0),
                                      xmin=lo, xmax=hi, seed=seed)
        if dist == 'beta':
            raw = sample_beta_1d(self.n_prior, a=params.get('a', 1.0),
                                 b=params.get('b', 1.0), seed=seed)
            return lo + raw * (hi - lo)
        raise ValueError(f"Unknown distribution '{dist}'. Choose 'uniform', "
                         "'loguniform', 'powerlaw', or 'beta'.")

    def _log_measurement(self, v, k):
        """Split-normal log p(d | v) for kick k (unnormalized; const cancels)."""
        sig = np.where(v >= self.v_kicks[k], self.sigma_hi[k], self.sigma_lo[k])
        return -0.5 * ((v - self.v_kicks[k]) / sig) ** 2

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def infer(self, verbose=False):
        """Run the inversion for every kick.

        Returns
        -------
        dict with keys:
            'v_kicks'  : (n_kicks,) the observed kicks
            'q','a1','a2' : (n_kicks, n_posterior) posterior samples
            'ess'      : (n_kicks,) effective sample size per kick
            'prior'    : {'q','a1','a2'} the shared prior pools
        """
        rng = np.random.default_rng(self.seed)
        flow = _get_flow_model()

        q_pool = self._sample_prior(self._q_spec, rng)
        a1_pool = self._sample_prior(self._a1_spec, rng)
        a2_pool = self._sample_prior(self._a2_spec, rng)
        np_ = self.n_prior

        M, npost = self.n_kicks, self.n_posterior
        q_post = np.empty((M, npost))
        a1_post = np.empty((M, npost))
        a2_post = np.empty((M, npost))
        ess = np.empty(M)

        exact = self.sigma_lo is None
        for k in range(M):
            v0 = self.v_kicks[k]
            if exact:
                logL = flow.log_prob(np.full(np_, v0), q_pool, a1_pool, a2_pool)
            else:
                vg = np.linspace(v0 - 5 * self.sigma_lo[k],
                                 v0 + 5 * self.sigma_hi[k], self.n_grid)
                dv = vg[1] - vg[0]
                G = len(vg)
                lpv = flow.log_prob(np.tile(vg, np_), np.repeat(q_pool, G),
                                    np.repeat(a1_pool, G),
                                    np.repeat(a2_pool, G)).reshape(np_, G)
                logL = logsumexp(lpv + self._log_measurement(vg, k)[None, :],
                                 axis=1) + np.log(dv)

            w = np.exp(logL - logL.max())
            w /= w.sum()
            ess[k] = 1.0 / np.sum(w ** 2)
            idx = rng.choice(np_, size=npost, p=w)
            q_post[k], a1_post[k], a2_post[k] = q_pool[idx], a1_pool[idx], a2_pool[idx]
            if verbose:
                print(f"  kick {k + 1}/{M}: v={v0:.0f} km/s  ESS={ess[k]:.0f}/{np_}"
                      f"  -> q={np.median(q_post[k]):.2f}, "
                      f"a1={np.median(a1_post[k]):.2f}, a2={np.median(a2_post[k]):.2f}")

        self.results = {
            'v_kicks': self.v_kicks,
            'q': q_post, 'a1': a1_post, 'a2': a2_post,
            'ess': ess,
            'prior': {'q': q_pool, 'a1': a1_pool, 'a2': a2_pool},
        }
        return self.results

    def summary(self, ci=90.0):
        """Per-kick posterior medians and central credible intervals.

        Returns a dict of (n_kicks,) arrays: ``*_median``, ``*_low``,
        ``*_high`` for each of q, a1, a2, plus ``ess``.
        """
        if not hasattr(self, 'results'):
            raise RuntimeError("Call .infer() before .summary().")
        lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
        out = {'v_kicks': self.results['v_kicks'], 'ess': self.results['ess']}
        for p in ('q', 'a1', 'a2'):
            s = self.results[p]
            out[f'{p}_median'] = np.median(s, axis=1)
            out[f'{p}_low'] = np.percentile(s, lo_q, axis=1)
            out[f'{p}_high'] = np.percentile(s, hi_q, axis=1)
        return out

    def posterior_predictive(self, index=0, n=None, seed=None):
        """Draw kicks from the inferred progenitor posterior for one kick.

        Each posterior progenitor is sampled once through the flow; because the
        flow marginalizes spin orientation, this distribution is intrinsically
        broad even for a sharply measured kick.
        """
        if not hasattr(self, 'results'):
            raise RuntimeError("Call .infer() before .posterior_predictive().")
        flow = _get_flow_model()
        q = self.results['q'][index]
        a1 = self.results['a1'][index]
        a2 = self.results['a2'][index]
        if n is not None and n < len(q):
            rng = np.random.default_rng(seed)
            sel = rng.choice(len(q), n, replace=False)
            q, a1, a2 = q[sel], a1[sel], a2[sel]
        return flow.sample(q, a1, a2, num_samples=1)

    def plot_posteriors(self, index=0, figsize=None):
        """3-panel prior-vs-posterior marginals for one kick."""
        if not hasattr(self, 'results'):
            raise RuntimeError("Call .infer() before plotting.")
        r = self.results
        fig, axes = plt.subplots(1, 3, figsize=figsize or (16, 4.5))
        panels = [('q', r'$q = m_1/m_2$'), ('a1', r'$a_1$'), ('a2', r'$a_2$')]
        for ax, (p, lbl) in zip(axes, panels):
            prior = r['prior'][p]
            post = r[p][index]
            xlim = (prior.min(), prior.max())
            ax.hist(prior, bins=40, range=xlim, density=True, color='0.85',
                    label='prior')
            ax.hist(post, bins=40, range=xlim, density=True, histtype='step',
                    lw=2.5, color='C0', label='posterior')
            ax.axvline(np.median(post), color='C0', ls='--', lw=1.5)
            ax.set_xlabel(lbl)
            ax.set_ylabel('Density')
            ax.set_xlim(xlim)
        axes[0].legend()
        fig.suptitle(rf"Progenitor posterior for $v_{{\rm kick}}="
                     rf"{r['v_kicks'][index]:.0f}$ km/s", y=1.02)
        plt.tight_layout()
        return fig, axes

    def __repr__(self):
        meas = "exact" if self.sigma_lo is None else "Gaussian errors"
        return (f"KickToProgenitor(n_kicks={self.n_kicks}, {meas}, "
                f"n_prior={self.n_prior}, n_posterior={self.n_posterior})")
