"""Tests for KickToProgenitor (kick -> progenitor inversion)."""
import numpy as np
import pytest

from gwGenealogy.core import KickToProgenitor
from gwGenealogy.binaries.bbh_remnant import _get_flow_model


def test_shapes_array_of_kicks():
    k = KickToProgenitor([200, 600, 1000], n_prior=20000, n_posterior=5000, seed=0)
    r = k.infer()
    assert r["q"].shape == (3, 5000)
    assert r["a1"].shape == (3, 5000)
    assert r["a2"].shape == (3, 5000)
    assert r["ess"].shape == (3,)
    assert np.all(r["ess"] > 0)


def test_spin_increases_with_kick():
    """Larger recoils should demand higher inferred primary spin."""
    k = KickToProgenitor([200, 600, 1000], q_min=1, q_max=20,
                         n_prior=50000, n_posterior=10000, seed=0)
    k.infer()
    a1_med = np.median(k.results["a1"], axis=1)
    assert a1_med[0] < a1_med[1] < a1_med[2]


def test_injection_recovery():
    """Invert a kick drawn from a known progenitor; truth lies in the 90% CI."""
    flow = _get_flow_model()
    true_q, true_a1, true_a2 = 4.0, 0.7, 0.5
    v_obs = float(np.median(flow.sample(true_q, true_a1, true_a2, num_samples=4000)))

    k = KickToProgenitor(v_obs, q_min=1, q_max=20,
                         n_prior=100000, n_posterior=20000, seed=0)
    k.infer()
    s = k.summary(ci=90)
    assert s["q_low"][0] <= true_q <= s["q_high"][0]
    assert s["a1_low"][0] <= true_a1 <= s["a1_high"][0]
    assert s["a2_low"][0] <= true_a2 <= s["a2_high"][0]


def test_array_prior_overrides_pool_size():
    rng = np.random.default_rng(0)
    qa = rng.uniform(1, 10, 5000)
    k = KickToProgenitor(500.0, q_array=qa,
                         a1_array=rng.uniform(0, 1, 5000),
                         a2_array=rng.uniform(0, 1, 5000),
                         n_posterior=2000, seed=0)
    assert k.n_prior == 5000
    r = k.infer()
    assert r["q"].shape == (1, 2000)


def test_mismatched_array_lengths_raise():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        KickToProgenitor(500.0, q_array=rng.uniform(1, 10, 100),
                         a1_array=rng.uniform(0, 1, 200))


def test_distribution_options_run():
    """loguniform q + beta spin priors produce a valid posterior."""
    k = KickToProgenitor(500.0, q_distribution="loguniform", q_min=1, q_max=15,
                         a1_distribution="beta", a1_params={"a": 2, "b": 2},
                         n_prior=40000, n_posterior=8000, seed=1)
    r = k.infer()
    assert r["q"].shape == (1, 8000)
    assert np.all(np.isfinite(r["q"])) and np.all(r["ess"] > 0)


def test_measurement_error_modes_run():
    """Exact, symmetric, and asymmetric measurement models all work."""
    base = dict(q_min=1, q_max=20, n_prior=60000, n_posterior=15000, seed=0)
    exact = KickToProgenitor(700.0, **base).infer()
    symm = KickToProgenitor(700.0, sigma=120, **base).infer()
    asym = KickToProgenitor(700.0, sigma_lo=150, sigma_hi=90, **base).infer()
    for r in (exact, symm, asym):
        assert np.all(np.isfinite(r["a1"])) and r["ess"][0] > 0
    # a finite measurement error should not sharpen the posterior
    assert np.std(symm["a1"][0]) >= 0.9 * np.std(exact["a1"][0])


def test_posterior_predictive_runs():
    k = KickToProgenitor(800.0, n_prior=40000, n_posterior=8000, seed=0)
    k.infer()
    pp = k.posterior_predictive(index=0, n=4000, seed=1)
    assert pp.shape == (4000,)
    assert np.all(np.isfinite(pp)) and np.all(pp > 0)
