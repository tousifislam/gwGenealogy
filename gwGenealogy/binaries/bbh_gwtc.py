#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==================================================================================
#
#    FILE: bbh_gwtc.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 06-07-2026
#    DESCRIPTION: GWTC default-family BBH population sampling (GWTC-3/4/5)
#
#    Functional forms validated against released popsummary grids:
#      spin magnitude: L1 residual ~1e-16
#      spin tilt:      ~1e-8
#      redshift (PL):  ~1e-16
#      mass m1:        ~1e-3
#      mass ratio q:   median 5e-4, max 3e-3
#
#    References:
#      GWTC-5.0: arXiv:2605.27226, Zenodo 20292639
#      GWTC-4.0: Zenodo 16911563
#      GWTC-3.0: Zenodo 5655785
#
#    ---- GWTC-4.0/5.0 default family model ----
#
#    Primary mass p(m1):
#      3-component mixture (Dirichlet weights lam_0, lam_1, 1-lam_0-lam_1):
#        p(m1) = [lam_0 * BPL(m1) + lam_1 * G1(m1) + (1-lam_0-lam_1) * G2(m1)]
#                * S(m1 | mlow_1, delta_m_1)
#      where:
#        BPL(m1) = (m1/m_break)^{-alpha_1}  for m1 < m_break   (broken power law)
#                  (m1/m_break)^{-alpha_2}  for m1 >= m_break
#        G1(m1) = N(m1; mpp_1, sigpp_1)   (low-mass Gaussian peak)
#        G2(m1) = N(m1; mpp_2, sigpp_2)   (high-mass Gaussian peak)
#        S(m | mlow, dm) = Planck-taper smoothing window (Talbot & Thrane 2018)
#
#    Mass ratio p(small_q | m1), where small_q = m2/m1 in [0, 1]:
#      p(small_q | m1) = small_q^beta * S(small_q*m1 | mlow_2, delta_m_2) / Z(m1)
#      where Z(m1) normalises over small_q in [0, 1].
#      Marginal p(small_q) = integral over m1 of p(m1) * p(small_q|m1).
#
#    Spin magnitude (iid for chi1_mag, chi2_mag):
#      p(chi) = TruncatedNormal(chi; mu_chi, sigma_chi, [0, amax])
#
#    Spin tilt (iid for cos_theta_1, cos_theta_2):
#      p(cos_theta) = xi * TruncatedNormal(cos_theta; mu_spin, sigma_spin, [-1, 1])
#                     + (1 - xi) * 1/2
#      where mu_spin is FREE (not fixed at +1, unlike GWTC-3).
#
#    Redshift:
#      PowerLaw:         dR/dz = (1 + z)^lambda
#      Madau-Dickinson:  dR/dz = (1+z)^gamma / [1 + ((1+z)/(1+z_peak))^kappa]
#
#    ---- GWTC-3 default model (Power Law + Peak) ----
#
#    Primary mass p(m1):
#      2-component mixture (weight lam):
#        p(m1) = [(1 - lam) * PL(m1) + lam * G(m1)] * S(m1 | mmin, delta_m)
#      where:
#        PL(m1) = m1^{-alpha}              (single power law)
#        G(m1)  = N(m1; mpp, sigpp)        (single Gaussian peak)
#
#    Mass ratio p(small_q | m1), where small_q = m2/m1 in [0, 1]:
#      p(small_q | m1) = small_q^beta * S(small_q*m1 | mmin, delta_m) / Z(m1)
#
#    Spin magnitude (iid for chi1_mag, chi2_mag):
#      p(chi) = Beta(chi/amax; alpha_chi, beta_chi) / amax
#      where (alpha_chi, beta_chi) are derived from (mu_chi, sigma_chi=variance):
#        mu' = mu_chi / amax,  var' = sigma_chi / amax^2
#        alpha_chi = mu' * [mu'(1-mu')/var' - 1]
#        beta_chi  = (1-mu') * [mu'(1-mu')/var' - 1]
#
#    Spin tilt (iid for cos_theta_1, cos_theta_2):
#      p(cos_theta) = xi * TruncatedNormal(cos_theta; mu=+1, sigma_spin, [-1, 1])
#                     + (1 - xi) * 1/2
#      NOTE: mu is FIXED at +1 (not free like GWTC-4/5).
#
#    Redshift:
#      dR/dz = (1 + z)^lambda   (same PowerLaw as GWTC-4/5)
#
#==================================================================================
__author__ = "Tousif Islam"

import numpy as np
from pathlib import Path
from scipy.stats import truncnorm, norm
from scipy.integrate import cumulative_trapezoid

# ============================================================================
# Model registry: maps catalog keys to HDF5 filenames
# ============================================================================
_MODELS = {
    "gwtc5": {
        "file": "gwtc5_default_var1.h5",
        "redshift_model": "PowerLaw",
    },
    "gwtc5_var4": {
        "file": "gwtc5_default_var4.h5",
        "redshift_model": "PowerLaw",
    },
    "gwtc5_madau_dickinson": {
        "file": "gwtc5_default_madau_dickinson.h5",
        "redshift_model": "MadauDickinson",
    },
    "gwtc4": {
        "file": "gwtc4_default.h5",
        "redshift_model": "PowerLaw",
    },
    "gwtc3": {
        "file": None,
        "redshift_model": "PowerLaw",
    },
}

# ============================================================================
# Hyper-prior tables (GWTC-4.0/5.0 default family)
# Transcribed from GWTC-4.0/5.0 methods paper appendix tables.
# All priors are uniform on the stated range unless noted.
# ============================================================================
_PRIORS_COMMON = {
    # mass: broken power law
    "alpha_1":    (-4.0, 12.0),     # power-law slope below break
    "alpha_2":    (-4.0, 12.0),     # power-law slope above break
    "beta":       (-2.0, 7.0),      # mass-ratio power-law index: p(q|m1) ~ q^beta
    "break_mass": (20.0, 50.0),     # break mass in Msun
    "delta_m_1":  (0.0, 10.0),      # low-mass Planck-taper width on m1
    "delta_m_2":  (0.0, 10.0),      # low-mass Planck-taper width on m2
    # mass: Gaussian peaks
    "mpp_1":      (5.0, 20.0),      # low-mass peak location (Msun)
    "mpp_2":      (25.0, 60.0),     # high-mass peak location (Msun)
    "sigpp_1":    (0.0, 10.0),      # low-mass peak width (Msun)
    "sigpp_2":    (0.0, 10.0),      # high-mass peak width (Msun)
    # spin magnitude: truncated Gaussian on [0, amax]
    "mu_chi":     (0.0, 1.0),       # Gaussian mean
    "sigma_chi":  (0.005, 1.0),     # Gaussian width
    # spin tilt: mixture of isotropic + Gaussian
    "xi_spin":    (0.0, 1.0),       # fraction in Gaussian tilt component
    "mu_spin":    (-1.0, 1.0),      # Gaussian tilt peak location (FREE)
    "sigma_spin": (0.01, 4.0),      # Gaussian tilt width
}
_FIXED = {"mmax": 300.0, "amax": 1.0}
# mlow_1, mlow_2 are jointly uniform on the triangle 3 <= mlow_2 <= mlow_1 <= 10
_MLOW_MIN, _MLOW_MAX = 3.0, 10.0

_PRIORS_REDSHIFT = {
    "PowerLaw":       {"lamb":   (-10.0, 10.0)},
    "MadauDickinson": {"gamma":  (-10.0, 10.0),
                       "kappa":  (-10.0, 10.0),
                       "z_peak": (0.0, 2.5)},
}

GRID_KEYS = ["mass_1", "mass_ratio", "a_1", "a_2",
             "cos_tilt_1", "cos_tilt_2", "redshift"]

_PACKAGE_DATA_DIR = Path(__file__).parent / "data"


# ============================================================================
# Functional forms: GWTC-4.0/5.0 default family
# ============================================================================

def _smoothing(m, mlow, dm):
    """Planck-taper low-mass smoothing window (Talbot & Thrane 2018).

    S(m | mlow, dm) = 0                                   for m <= mlow
                    = 1 / [exp(dm/a + dm/(a-dm)) + 1]     for mlow < m < mlow+dm
                    = 1                                    for m >= mlow+dm
    where a = m - mlow.
    """
    s = np.zeros_like(m, dtype=float)
    s[m >= mlow + dm] = 1.0
    with np.errstate(over="ignore"):
        mid = (m > mlow) & (m < mlow + dm)
        a = m[mid] - mlow
        s[mid] = 1.0 / (np.exp(dm / a + dm / (a - dm)) + 1.0)
    return s


def _mass1_pdf(m, p):
    """GWTC-4/5 primary mass PDF: broken power law + two Gaussian peaks.

    p(m1) = [lam_0 * BPL(m1) + lam_1 * G(m1; mpp_1, sigpp_1)
             + (1-lam_0-lam_1) * G(m1; mpp_2, sigpp_2)]
            * S(m1 | mlow_1, delta_m_1)

    BPL(m1) = (m1/m_break)^{-alpha_1}  for m1 < m_break
              (m1/m_break)^{-alpha_2}  for m1 >= m_break
    """
    a1, a2, mb = p["alpha_1"], p["alpha_2"], p["break_mass"]
    mlow, mhigh, dm = p["mlow_1"], p["mmax"], p["delta_m_1"]
    l0, l1 = p["lam_0"], p["lam_1"]

    # Broken power law: (m/m_break)^{-alpha_1} below break, ^{-alpha_2} above
    win = ((m >= mlow) & (m < mhigh)).astype(float)
    bpl = np.where(m < mb, (m / mb) ** (-a1), (m / mb) ** (-a2)) * win
    area = np.trapezoid(bpl, m)
    if area > 0:
        bpl /= area

    # Two Gaussian peaks
    g1 = norm.pdf(m, p["mpp_1"], p["sigpp_1"])
    g2 = norm.pdf(m, p["mpp_2"], p["sigpp_2"])

    # 3-component Dirichlet mixture * Planck-taper smoothing
    pdf = (l0 * bpl + l1 * g1 + (1.0 - l0 - l1) * g2) * _smoothing(m, mlow, dm)
    return np.where((m >= mlow) & (m <= mhigh), pdf, 0.0)


def _massratio_pdf(q, m1grid, p):
    """GWTC-4/5 marginal mass ratio PDF.

    p(q | m1) = q^beta * S(q*m1 | mlow_2, delta_m_2) / Z(m1)
    p(q) = integral over m1 of p(m1) * p(q | m1)
    """
    # Normalise p(m1)
    pm1 = _mass1_pdf(m1grid, p)
    area = np.trapezoid(pm1, m1grid)
    if area > 0:
        pm1 = pm1 / area

    # Build 2D grid: p(q|m1) = q^beta * S(m2 | mlow_2, delta_m_2)
    M1, Q = np.meshgrid(m1grid, q, indexing="ij")
    m2 = (M1 * Q).ravel()
    Sm2 = _smoothing(m2, p["mlow_2"], p["delta_m_2"]).reshape(M1.shape)
    pqg = (Q ** p["beta"]) * Sm2

    # Normalise p(q|m1) for each m1
    nq = np.trapezoid(pqg, q, axis=1)
    nq[nq == 0] = 1.0
    pqg /= nq[:, None]

    # Marginalise: p(q) = int p(m1) p(q|m1) dm1
    return np.trapezoid(pm1[:, None] * pqg, m1grid, axis=0)


def _spinmag_pdf(a, p):
    """GWTC-4/5 spin magnitude PDF (iid for chi1_mag, chi2_mag).

    p(chi) = TruncatedNormal(chi; mu_chi, sigma_chi, [0, amax])
    """
    mu, sig = p["mu_chi"], p["sigma_chi"]
    lo, hi = 0.0, p["amax"]
    return truncnorm.pdf(a, (lo - mu) / sig, (hi - mu) / sig, loc=mu, scale=sig)


def _costilt_pdf(c, p):
    """GWTC-4/5 spin tilt PDF (iid for cos_theta_1, cos_theta_2).

    p(cos_theta) = xi * TruncatedNormal(cos_theta; mu_spin, sigma_spin, [-1, 1])
                   + (1 - xi) * 1/2

    mu_spin is FREE (not fixed at +1 like GWTC-3).
    """
    z, mu, st = p["xi_spin"], p["mu_spin"], p["sigma_spin"]
    g = truncnorm.pdf(c, (-1.0 - mu) / st, (1.0 - mu) / st, loc=mu, scale=st)
    return z * g + (1.0 - z) * 0.5


def _redshift_pdf(z, p, model="PowerLaw"):
    """Comoving merger-rate density (no dVc/dz factor).

    PowerLaw:        dR/dz = (1 + z)^lambda
    MadauDickinson:  dR/dz = (1+z)^gamma / [1 + ((1+z)/(1+z_peak))^kappa]
    """
    if model == "PowerLaw":
        return (1.0 + z) ** p["lamb"]
    elif model == "MadauDickinson":
        return ((1.0 + z) ** p["gamma"]
                / (1.0 + ((1.0 + z) / (1.0 + p["z_peak"])) ** p["kappa"]))
    raise ValueError(f"Unknown redshift model: {model}")


# ============================================================================
# Functional forms: GWTC-3 default (Power Law + Peak)
# ============================================================================

def _mu_var_to_alpha_beta(mu, var, amax=1.0):
    """Convert Beta distribution (mean, variance) to (alpha, beta) shape params.

    For Beta on [0, amax]:
      mu' = mu / amax,   var' = var / amax^2
      alpha = mu' * [mu'(1-mu')/var' - 1]
      beta  = (1-mu') * [mu'(1-mu')/var' - 1]

    NB: GWTC-3 stores sigma_chi as the Beta VARIANCE (not std dev).
    """
    mu_scaled = mu / amax
    var_scaled = var / amax**2
    alpha = mu_scaled * (mu_scaled * (1 - mu_scaled) / var_scaled - 1)
    beta = (1 - mu_scaled) * (mu_scaled * (1 - mu_scaled) / var_scaled - 1)
    return alpha, beta


def _spinmag_pdf_beta(a, alpha_chi, beta_chi, amax=1.0, eps=1e-6):
    """GWTC-3 spin magnitude PDF (iid for chi1_mag, chi2_mag).

    p(chi) = Beta(chi/amax; alpha_chi, beta_chi) / amax

    Grid endpoints are nudged by eps to avoid divergence when alpha<1 or beta<1.
    """
    from scipy.stats import beta as _beta
    x = np.clip(np.asarray(a, float) / amax, eps, 1.0 - eps)
    pdf = _beta.pdf(x, alpha_chi, beta_chi) / amax
    return np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)


def _costilt_pdf_gwtc3(c, xi_spin, sigma_spin):
    """GWTC-3 spin tilt PDF (iid for cos_theta_1, cos_theta_2).

    p(cos_theta) = xi * TruncatedNormal(cos_theta; mu=+1, sigma_spin, [-1, 1])
                   + (1 - xi) * 1/2

    Key difference from GWTC-4/5: mu is FIXED at +1 (aligned).
    """
    g = truncnorm.pdf(c, (-1.0 - 1.0) / sigma_spin, (1.0 - 1.0) / sigma_spin,
                       loc=1.0, scale=sigma_spin)
    return xi_spin * g + (1.0 - xi_spin) * 0.5


def _mass1_pdf_gwtc3(m, p):
    """GWTC-3 primary mass PDF: single power law + single Gaussian peak.

    p(m1) = [(1 - lam) * PL(m1) + lam * G(m1; mpp, sigpp)]
            * S(m1 | mmin, delta_m)

    PL(m1) = m1^{-alpha}   (single power law, normalised over [mmin, mmax])
    G(m1)  = N(m1; mpp, sigpp)  (single Gaussian peak, normalised over [mmin, mmax])
    """
    alpha = p["alpha"]
    mmin, mmax, dm = p["mmin"], p["mmax"], p["delta_m"]
    lam = p["lam"]

    # Single power law: m1^{-alpha}
    win = ((m >= mmin) & (m <= mmax)).astype(float)
    pl = m ** (-alpha) * win
    area = np.trapezoid(pl, m)
    if area > 0:
        pl /= area

    # Single Gaussian peak
    g = norm.pdf(m, p["mpp"], p["sigpp"])
    g_area = np.trapezoid(g * win, m)
    if g_area > 0:
        g = g * win / g_area

    # 2-component mixture * Planck-taper smoothing
    pdf = ((1.0 - lam) * pl + lam * g) * _smoothing(m, mmin, dm)
    return np.where((m >= mmin) & (m <= mmax), pdf, 0.0)


def _massratio_pdf_gwtc3(q, m1grid, p):
    """GWTC-3 marginal mass ratio PDF.

    p(q | m1) = q^beta * S(q*m1 | mmin, delta_m) / Z(m1)
    p(q) = integral over m1 of p(m1) * p(q | m1)

    Same structure as GWTC-4/5 but uses mmin/delta_m for both m1 and m2 taper.
    """
    # Normalise p(m1)
    pm1 = _mass1_pdf_gwtc3(m1grid, p)
    area = np.trapezoid(pm1, m1grid)
    if area > 0:
        pm1 = pm1 / area

    # Build 2D grid: p(q|m1) = q^beta * S(m2=q*m1 | mmin, delta_m)
    M1, Q = np.meshgrid(m1grid, q, indexing="ij")
    m2 = (M1 * Q).ravel()
    Sm2 = _smoothing(m2, p["mmin"], p["delta_m"]).reshape(M1.shape)
    pqg = (Q ** p["beta"]) * Sm2

    # Normalise p(q|m1) for each m1
    nq = np.trapezoid(pqg, q, axis=1)
    nq[nq == 0] = 1.0
    pqg /= nq[:, None]

    # Marginalise: p(q) = int p(m1) p(q|m1) dm1
    return np.trapezoid(pm1[:, None] * pqg, m1grid, axis=0)


# ============================================================================
# Hyper-prior sampling (GWTC-4/5 default family)
# ============================================================================
def _sample_hyperprior(n_draws, redshift="PowerLaw", seed=None):
    """Draw hyper-parameter sets from the population hyper-prior.

    All common params are U(lo, hi). Additional structure:
      - Dirichlet(1,1,1) for mixture weights (lam_0, lam_1, 1-lam_0-lam_1)
      - Triangular prior for (mlow_1, mlow_2): joint U on 3 <= mlow_2 <= mlow_1 <= 10
    """
    rng = np.random.default_rng(seed)
    h = {}
    for name, (lo, hi) in _PRIORS_COMMON.items():
        h[name] = rng.uniform(lo, hi, n_draws)

    # Enforce strictly positive widths
    h["sigpp_1"] = np.clip(h["sigpp_1"], 1e-3, None)
    h["sigpp_2"] = np.clip(h["sigpp_2"], 1e-3, None)
    h["delta_m_1"] = np.clip(h["delta_m_1"], 1e-3, None)
    h["delta_m_2"] = np.clip(h["delta_m_2"], 1e-3, None)

    # 3-component flat Dirichlet: (lam_0, lam_1, lam_2=1-lam_0-lam_1)
    g = rng.gamma(1.0, 1.0, size=(n_draws, 3))
    g /= g.sum(axis=1, keepdims=True)
    h["lam_0"], h["lam_1"] = g[:, 0], g[:, 1]

    # Triangular low-mass edges: mlow_1 >= mlow_2, both in [3, 10]
    e = rng.uniform(_MLOW_MIN, _MLOW_MAX, size=(n_draws, 2))
    h["mlow_1"], h["mlow_2"] = e.max(axis=1), e.min(axis=1)

    # Fixed constants
    for name, val in _FIXED.items():
        h[name] = np.full(n_draws, val)

    # Redshift-model-specific params
    for name, (lo, hi) in _PRIORS_REDSHIFT[redshift].items():
        h[name] = rng.uniform(lo, hi, n_draws)
    return h


# ============================================================================
# Grid builders
# ============================================================================

def _build_prior_grids(positions, n_draws=1000, redshift="PowerLaw", seed=0):
    """Build prior-predictive dR/dx grids for the GWTC-4/5 default family.

    For each of n_draws hyper-prior samples, evaluate all PDFs on the given
    position grids. The output structure matches the released posterior grids
    so the same inverse-CDF sampler works for both.
    """
    H = _sample_hyperprior(n_draws, redshift=redshift, seed=seed)
    keys = list(positions)
    grids = {k: {"positions": np.asarray(positions[k]),
                 "rates": np.empty((n_draws, len(positions[k])))} for k in keys}
    m1grid = np.asarray(positions["mass_1"])

    for d in range(n_draws):
        p = {k: H[k][d] for k in H}
        if "mass_1" in grids:
            grids["mass_1"]["rates"][d] = _mass1_pdf(m1grid, p)
        if "mass_ratio" in grids:
            grids["mass_ratio"]["rates"][d] = _massratio_pdf(
                grids["mass_ratio"]["positions"], m1grid, p)
        for ak in ("a_1", "a_2"):
            if ak in grids:
                grids[ak]["rates"][d] = _spinmag_pdf(grids[ak]["positions"], p)
        for ck in ("cos_tilt_1", "cos_tilt_2"):
            if ck in grids:
                grids[ck]["rates"][d] = _costilt_pdf(grids[ck]["positions"], p)
        if "redshift" in grids:
            grids["redshift"]["rates"][d] = _redshift_pdf(
                grids["redshift"]["positions"], p, model=redshift)
    return grids


def _build_gwtc3_grids(positions, hyper_samples, n_draws=None, seed=0):
    """Build per-hyper-draw dR/dx grids for the GWTC-3 default model.

    Evaluates the GWTC-3 functional forms (Power Law + Peak mass, Beta spin,
    fixed-peak tilt) for each hyper-posterior draw, producing the same grid
    structure as _load_population_grids.
    """
    rng = np.random.default_rng(seed)
    H = {k: np.asarray(v).reshape(-1) for k, v in hyper_samples.items()}

    # Convert spin-magnitude (mu, variance) -> (alpha_chi, beta_chi)
    if "alpha_chi" not in H or "beta_chi" not in H:
        if "mu_chi" in H and "sigma_chi" in H:
            amax = float(H["amax"][0]) if "amax" in H else 1.0
            a_chi, b_chi = _mu_var_to_alpha_beta(H["mu_chi"], H["sigma_chi"], amax)
            H["alpha_chi"] = np.clip(a_chi, 1e-2, None)
            H["beta_chi"] = np.clip(b_chi, 1e-2, None)
        else:
            raise KeyError(
                "need (alpha_chi, beta_chi) or (mu_chi, sigma_chi) in hyper_samples")

    # Optionally subsample hyper-draws (mass eval is the cost driver)
    ntot = len(H["lamb"])
    idx = np.arange(ntot)
    if n_draws is not None and n_draws < ntot:
        idx = rng.choice(ntot, size=n_draws, replace=False)
    nd = len(idx)

    keys = list(positions)
    pos = {k: np.asarray(positions[k], float) for k in keys}
    grids = {k: {"positions": pos[k],
                 "rates": np.empty((nd, pos[k].size))} for k in keys}

    for d, j in enumerate(idx):
        p = {k: float(H[k][j]) for k in H if H[k].ndim == 1}

        # Mass: Power Law + Peak (native, no gwpopulation)
        if "mass_1" in grids or "mass_ratio" in grids:
            p_m1 = _mass1_pdf_gwtc3(pos["mass_1"], p)
            p_q = _massratio_pdf_gwtc3(pos["mass_ratio"], pos["mass_1"], p)
            if "mass_1" in grids:
                grids["mass_1"]["rates"][d] = p_m1
            if "mass_ratio" in grids:
                grids["mass_ratio"]["rates"][d] = p_q

        # Spin magnitude: Beta(alpha_chi, beta_chi) on [0, amax]
        for ak in ("a_1", "a_2"):
            if ak in grids:
                grids[ak]["rates"][d] = _spinmag_pdf_beta(
                    pos[ak], p["alpha_chi"], p["beta_chi"])

        # Spin tilt: xi * N(cos_t; +1, sigma) + (1-xi)/2
        for ck in ("cos_tilt_1", "cos_tilt_2"):
            if ck in grids:
                grids[ck]["rates"][d] = _costilt_pdf_gwtc3(
                    pos[ck], p["xi_spin"], p["sigma_spin"])

        # Redshift: (1+z)^lambda
        if "redshift" in grids:
            grids["redshift"]["rates"][d] = (1.0 + pos["redshift"]) ** p["lamb"]

    return grids


# ============================================================================
# Grid loader (from popsummary HDF5 files)
# ============================================================================

def _resolve_data_file(filename, data_dir=None):
    """Find a data file, checking data_dir first then the package data/ dir."""
    if data_dir is not None:
        path = Path(data_dir) / filename
        if path.exists():
            return path
    path = _PACKAGE_DATA_DIR / filename
    if path.exists():
        return path
    raise FileNotFoundError(
        f"Population file not found: '{filename}'\n"
        f"Searched: {data_dir or '(not specified)'} and {_PACKAGE_DATA_DIR}\n"
        f"Download from the GWTC data release (see docstring).")


def _load_population_grids(fname, keys=None):
    """Load per-hyper-draw dR/dx grids from a popsummary HDF5 file.

    Returns {key: {'positions': (Npos,), 'rates': (Ndraws, Npos)}}.
    """
    import h5py
    if keys is None:
        keys = GRID_KEYS
    grids = {}
    with h5py.File(fname, "r") as f:
        rog = f["posterior/rates_on_grids"]
        available = list(rog.keys())
        for k in keys:
            if k not in available:
                raise KeyError(
                    f"grid '{k}' not in file; available: {available}")
            grids[k] = {
                "positions": rog[k]["positions"][:].reshape(-1),
                "rates":     rog[k]["rates"][:],
            }
    return grids


# ============================================================================
# Inverse-CDF sampler
# ============================================================================

def _inverse_cdf_sample(positions, rates, hyper_idx, n, rng):
    """Draw samples via inverse CDF from per-hyper-draw rate grids.

    hyper_idx=None  -> sample from the draw-averaged PDF (mode='mean')
    hyper_idx=array -> sample i uses the CDF of hyper draw hyper_idx[i] (mode='ppd')
    """
    if hyper_idx is None:
        # Mean mode: average over all hyper-draws
        pdf = rates.mean(axis=0)
        cdf = cumulative_trapezoid(pdf, positions, initial=0.0)
        cdf /= cdf[-1]
        return np.interp(rng.random(n), cdf, positions)

    # PPD mode: vectorised per unique draw
    out = np.empty(hyper_idx.shape[0])
    for j in np.unique(hyper_idx):
        cdf = cumulative_trapezoid(rates[j], positions, initial=0.0)
        cdf /= cdf[-1]
        mask = hyper_idx == j
        out[mask] = np.interp(rng.random(int(mask.sum())), cdf, positions)
    return out


def _sample_from_grids(grids, n, mode="ppd", seed=None):
    """Draw n BBH parameter sets from a dict of dR/dx grids.

    Derived quantities computed after sampling:
      m2 = small_q * m1
      chi_eff = (chi1_mag*cos_theta_1 + small_q*chi2_mag*cos_theta_2) / (1 + small_q)
      chi_p = max(chi1_mag*sin_theta_1,
                  [(3+4*small_q)/(4+3*small_q)] * small_q * chi2_mag * sin_theta_2)

    Convention: small_q = m2/m1 in [0, 1].
    """
    rng = np.random.default_rng(seed)
    ndraws = grids["mass_1"]["rates"].shape[0]

    if mode == "ppd":
        hyper_idx = rng.integers(0, ndraws, n)
    elif mode == "mean":
        hyper_idx = None
    elif isinstance(mode, (int, np.integer)):
        if not (0 <= int(mode) < ndraws):
            raise ValueError(f"draw index must be in [0, {ndraws})")
        hyper_idx = np.full(n, int(mode))
    else:
        raise ValueError("mode must be 'ppd', 'mean', or an integer draw index")

    def draw(key):
        g = grids[key]
        return _inverse_cdf_sample(g["positions"], g["rates"],
                                   hyper_idx, n, rng)

    m1 = draw("mass_1")
    small_q = draw("mass_ratio")
    chi1_mag, chi2_mag = draw("a_1"), draw("a_2")
    cos_theta_1, cos_theta_2 = draw("cos_tilt_1"), draw("cos_tilt_2")
    z = draw("redshift")

    # Derived quantities
    m2 = small_q * m1
    chi_eff = (chi1_mag * cos_theta_1 + small_q * chi2_mag * cos_theta_2) / (1.0 + small_q)
    sin_theta_1 = np.sqrt(1 - cos_theta_1**2)
    sin_theta_2 = np.sqrt(1 - cos_theta_2**2)
    chi_p = np.maximum(chi1_mag * sin_theta_1,
                       ((3 + 4*small_q) / (4 + 3*small_q)) * small_q * chi2_mag * sin_theta_2)

    out = {
        "mass_1": m1, "mass_2": m2, "q": 1.0 / small_q, "small_q": small_q,
        "chi1_mag": chi1_mag, "chi2_mag": chi2_mag,
        "cos_theta_1": cos_theta_1, "cos_theta_2": cos_theta_2,
        "theta_1": np.arccos(cos_theta_1), "theta_2": np.arccos(cos_theta_2),
        "redshift": z,
        "chi_eff": chi_eff, "chi_p": chi_p,
    }
    if hyper_idx is not None:
        out["hyper_draw_index"] = hyper_idx
    return out


# ============================================================================
# Default grid positions (used for prior-predictive sampling)
# ============================================================================
_DEFAULT_POSITIONS = {
    "mass_1":     np.linspace(2.0, 100.0, 1000),
    "mass_ratio": np.linspace(0.05, 1.0, 500),
    "a_1":        np.linspace(0.0, 1.0, 1000),
    "a_2":        np.linspace(0.0, 1.0, 1000),
    "cos_tilt_1": np.linspace(-1.0, 1.0, 1000),
    "cos_tilt_2": np.linspace(-1.0, 1.0, 1000),
    "redshift":   np.linspace(1e-6, 1.9, 2500),
}


# ============================================================================
# Public API
# ============================================================================
def sample_gwtc_population(n_samples, catalog="gwtc5", source="posterior",
                           mode="ppd", data_dir=None, n_hyper_draws=1000,
                           gwtc3_hyper_samples=None, seed=None):
    """
    Sample BBH parameters from GWTC default-family population models.

    Parameters
    ----------
    n_samples : int
        Number of BBH systems to draw
    catalog : str
        Population catalog (default: 'gwtc5'):
        - 'gwtc3': Power Law + Peak, Beta spin, tilt peak fixed at +1
        - 'gwtc4': Broken PowerLaw + Two Peaks, Gaussian spin, free tilt peak
        - 'gwtc5': same model family as gwtc4, O3+O4 selection
        - 'gwtc5_var4': gwtc5 with looser selection-variance cut
        - 'gwtc5_madau_dickinson': gwtc5 mass/spin + Madau-Dickinson redshift
    source : str
        'prior' or 'posterior' (default: 'posterior')
    mode : str or int
        Sampling mode (default: 'ppd'):
        - 'ppd': posterior predictive (random hyper-draw per binary)
        - 'mean': hyper-posterior/prior-averaged distribution
        - int: single hyper-draw index
    data_dir : str or Path or None
        Directory containing popsummary HDF5 files. If None, searches the
        package data/ directory. Required for source='posterior' if files
        are not in the package data/ directory.
    n_hyper_draws : int
        Number of hyper-prior draws for source='prior', or number of
        hyper-posterior draws to subsample for gwtc3 (default: 1000)
    gwtc3_hyper_samples : dict or None
        Pre-loaded GWTC-3 hyper-posterior samples. If None and catalog='gwtc3'
        with source='posterior', loads from gwtc3_default.json.
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary with keys: mass_1, mass_2, q (=m1/m2 >= 1), small_q (=m2/m1),
        chi1_mag, chi2_mag, cos_theta_1, cos_theta_2, theta_1, theta_2,
        redshift, chi_eff, chi_p (and hyper_draw_index if mode != 'mean')
    """
    catalog = catalog.lower()
    if catalog not in _MODELS:
        raise ValueError(
            f"Unknown catalog: '{catalog}'. "
            f"Choose from {list(_MODELS.keys())}")

    rng = np.random.default_rng(seed)
    seed_grid = int(rng.integers(0, 2**31))
    seed_sample = int(rng.integers(0, 2**31))

    model_info = _MODELS[catalog]
    redshift_model = model_info["redshift_model"]

    if source == "prior":
        if catalog == "gwtc3":
            raise ValueError(
                "GWTC-3 uses a different model family (Power Law + Peak, "
                "Beta spin). Hyper-prior sampling is only implemented for "
                "the GWTC-4/5 default family. Use source='posterior' for gwtc3.")
        grids = _build_prior_grids(
            _DEFAULT_POSITIONS, n_draws=n_hyper_draws,
            redshift=redshift_model, seed=seed_grid)

    elif source == "posterior":
        if catalog == "gwtc3":
            grids = _load_gwtc3_posterior_grids(
                data_dir, gwtc3_hyper_samples, n_hyper_draws, seed_grid)
        else:
            fname = _resolve_data_file(model_info["file"], data_dir)
            grids = _load_population_grids(str(fname))
    else:
        raise ValueError(f"Unknown source: '{source}'. Choose 'prior' or 'posterior'.")

    return _sample_from_grids(grids, n_samples, mode=mode, seed=seed_sample)


def _load_gwtc3_posterior_grids(data_dir, hyper_samples, n_draws, seed):
    """Load GWTC-3 hyper-posterior samples and build dR/dx grids."""
    if hyper_samples is None:
        import json
        candidates = []
        if data_dir is not None:
            candidates.append(
                Path(data_dir) / "GWTC-3-population-data" / "analyses" /
                "PowerLawPeak" /
                "o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json")
            candidates.append(Path(data_dir) / "gwtc3_default.json")
        candidates.append(_PACKAGE_DATA_DIR / "gwtc3_default.json")

        result_path = None
        for c in candidates:
            if c.exists():
                result_path = c
                break
        if result_path is None:
            raise FileNotFoundError(
                "GWTC-3 result file not found. Searched:\n"
                + "\n".join(f"  {c}" for c in candidates)
                + "\nDownload from Zenodo 5655785.")

        with open(result_path) as f:
            content = json.load(f)["posterior"]["content"]
        hyper_samples = {k: np.asarray(v, float) for k, v in content.items()}

    return _build_gwtc3_grids(
        _DEFAULT_POSITIONS, hyper_samples, n_draws=n_draws, seed=seed)


def available_catalogs():
    """Return list of available catalog keys."""
    return list(_MODELS.keys())
