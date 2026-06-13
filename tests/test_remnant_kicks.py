"""Tests for the batched gwmodel kick path in BBHRemnant."""
import numpy as np

from gwGenealogy.binaries import BBHRemnant


def ordered_masses(rng, n, lo=10.0, hi=50.0):
    """Draw component masses with m1 >= m2 (so q = m1/m2 >= 1)."""
    a = rng.uniform(lo, hi, n)
    b = rng.uniform(lo, hi, n)
    return np.maximum(a, b), np.minimum(a, b)


def test_gwmodel_scalar_shape():
    """Scalar construction still yields a length-1 vkick array."""
    r = BBHRemnant(m1=30, m2=20, a1=0.5, a2=0.4,
                   theta1=0.5, theta2=0.5, phi1=0.0, phi2=0.0,
                   precessing=True, kick_model="gwmodel")
    assert r.vkick.shape == (1,)
    assert np.isfinite(r.vkick).all() and (r.vkick > 0).all()


def test_gwmodel_array_shape():
    rng = np.random.default_rng(0)
    n = 500
    m1, m2 = ordered_masses(rng, n, 20, 50)
    z = rng.uniform(0, np.pi, n)
    r = BBHRemnant(m1=m1, m2=m2, a1=rng.uniform(0, 1, n), a2=rng.uniform(0, 1, n),
                   theta1=z, theta2=z, phi1=z, phi2=z,
                   precessing=True, kick_model="gwmodel")
    assert r.vkick.shape == (n,)
    assert np.isfinite(r.vkick).all() and (r.vkick > 0).all()


def test_gwmodel_and_hlz_same_order_of_magnitude():
    rng = np.random.default_rng(0)
    n = 2000
    m1, m2 = ordered_masses(rng, n, 20, 50)
    z = rng.uniform(0, np.pi, n)
    kw = dict(m1=m1, m2=m2, a1=rng.uniform(0, 1, n), a2=rng.uniform(0, 1, n),
              theta1=z, theta2=z, phi1=z, phi2=z, precessing=True)
    v_gw = BBHRemnant(kick_model="gwmodel", **kw).vkick
    v_hlz = BBHRemnant(kick_model="hlz", **kw).vkick
    assert 0.3 < v_gw.mean() / v_hlz.mean() < 3.0
