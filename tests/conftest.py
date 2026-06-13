"""Shared pytest fixtures for the gwGenealogy test suite."""
import pytest


@pytest.fixture(scope="session", autouse=True)
def _preload_flow():
    """Load the IW2025 gwmodel flow once before any test runs.

    Keeps the (one-time, RNG-perturbing) lazy load out of the seeded paths so
    reproducibility tests are stable regardless of test order.
    """
    from gwGenealogy.binaries.bbh_remnant import preload_kick_model
    preload_kick_model("gwmodel")
