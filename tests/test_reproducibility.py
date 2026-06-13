"""Same seed -> identical results for the gwmodel-using classes.

These guard the legacy-RNG seeding fix (seed_legacy_rng + preload_kick_model):
the gwmodel kick draws from global torch/numpy RNG, so without that fix these
would not be reproducible.
"""
import numpy as np

from gwGenealogy.core import (
    BBHRetentionProbabilities,
    BBHRetentionProbabilityOverChiq,
    HierarchicalMergersInClusterPopulation,
    MonteCarloBHSeedGrowth,
)


def test_retention_probabilities_reproducible():
    kw = dict(v_esc_values=[100, 200], kick_models=["gwmodel"], n_samples=500, seed=7)
    r1 = BBHRetentionProbabilities(**kw).compute()
    r2 = BBHRetentionProbabilities(**kw).compute()
    assert np.allclose(r1["kicks"]["gwmodel"], r2["kicks"]["gwmodel"])
    assert np.allclose(r1["p_ret"]["gwmodel"], r2["p_ret"]["gwmodel"])


def test_retention_probabilities_seed_changes_result():
    base = dict(v_esc_values=[100], kick_models=["gwmodel"], n_samples=500)
    a = BBHRetentionProbabilities(seed=1, **base).compute()
    b = BBHRetentionProbabilities(seed=2, **base).compute()
    assert not np.allclose(a["kicks"]["gwmodel"], b["kicks"]["gwmodel"])


def test_retention_over_chiq_reproducible():
    kw = dict(q_values=[2.0], chi_max_values=[0.5], v_esc_values=[100],
              kick_models=["gwmodel"], n_samples=500, seed=3)
    r1 = BBHRetentionProbabilityOverChiq(**kw).compute()
    r2 = BBHRetentionProbabilityOverChiq(**kw).compute()
    assert np.allclose(r1["p_ret"]["gwmodel"], r2["p_ret"]["gwmodel"])


def test_hierarchical_population_reproducible():
    kw = dict(n_samples=1000, max_gen=3, kick_model="gwmodel", seed=5)
    d1 = HierarchicalMergersInClusterPopulation(**kw).simulate()
    d2 = HierarchicalMergersInClusterPopulation(**kw).simulate()
    for g in d1:
        assert np.allclose(d1[g]["m"], d2[g]["m"])
        assert np.allclose(d1[g]["spin"], d2[g]["spin"])


def test_seed_growth_reproducible():
    kw = dict(v_esc=300, Z=0.005, m_seed=10, m_targets=[100],
              kick_model="gwmodel", n_pool=500, seed=9)
    r1 = MonteCarloBHSeedGrowth(**kw).simulate(n_experiments=200)
    r2 = MonteCarloBHSeedGrowth(**kw).simulate(n_experiments=200)
    assert np.allclose(r1["final_masses"], r2["final_masses"])
    assert r1["P_ret"] == r2["P_ret"]


def test_seed_growth_grid_reproducible():
    kw = dict(v_esc=300, Z=0.005, m_seed=10, m_targets=[100],
              kick_model="gwmodel", n_pool=500, seed=9)
    g1 = MonteCarloBHSeedGrowth(**kw).simulate_grid([100, 300], n_experiments=150)
    g2 = MonteCarloBHSeedGrowth(**kw).simulate_grid([100, 300], n_experiments=150)
    assert np.allclose(g1["P_ret"], g2["P_ret"])
