#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: bbh_remnant.py
#
#    BBHRemnant class: compute remnant mass, spin, and kick velocity
#    for binary black hole mergers. Takes a BBHs object or raw arrays
#    and dispatches to the chosen model backend.
#
#    Precessing models:
#      mass/spin: 'hbr' (default), 'sur7dq4remnant', 'sur7dq4emri'
#      kick:      'gwmodel' (default), 'hlz' (-> CLZM2007), 'sur7dq4remnant'
#
#    Nonprecessing models:
#      mass/spin: 'uib' (default), 'hbr', 'sur7dq4remnant', 'sur7dq4emri'
#      kick:      'gwmodel_kick_q200' (default), 'hlz', 'sur3dq8remnant'
#
#    Formula-based models come from the gwModels package.
#    Surrogate models use surfinBH directly (optional dependency).
#
#    AUTHOR: Tousif Islam
#    CREATED: 06-07-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import os

from gwModels.remnants import (
    bbh_final_mass_precessing_BMR2012,
    bbh_final_spin_precessing_HBR2016,
    bbh_final_mass_non_precessing_UIB2016,
    bbh_final_spin_non_precessing_UIB2016,
    bbh_final_kick_precessing_CLZM2007,
    bbh_final_kick_nonprecessing_HLZ2014,
    gwModel_kick_q200,
    gwModel_kick_prec_flow,
)
import gwModels.remnants.IW2025_kick_precessing as _iw2025_mod

from ..utils.coordinates import spins_polar_to_cartesian_vectors

try:
    import surfinBH
    _HAS_SURFIN = True
except ImportError:
    _HAS_SURFIN = False

C_KMS = 299792.458

_FLOW_MODEL = None
_SURFIN_FITS = {}


def _get_flow_model():
    """Lazy-load and cache the gwModel precessing kick flow model."""
    global _FLOW_MODEL
    if _FLOW_MODEL is None:
        datadir = os.path.join(os.path.dirname(os.path.dirname(_iw2025_mod.__file__)), 'data')
        _FLOW_MODEL = gwModel_kick_prec_flow(datadir)
    return _FLOW_MODEL


def _get_surfin_fit(name):
    """Lazy-load and cache a surfinBH fit by name."""
    if not _HAS_SURFIN:
        raise ImportError(f"surfinBH required for {name}. Install with: pip install surfinBH")
    if name not in _SURFIN_FITS:
        _SURFIN_FITS[name] = surfinBH.LoadFits(name)
    return _SURFIN_FITS[name]


class BBHRemnant:
    """Compute remnant properties for BBH mergers.

    Parameters
    ----------
    bbh : BBHs instance, optional
        A gwGenealogy.binaries.BBHs object. If provided, all binary
        parameters are taken from it.
    m1, m2 : array, optional
        Component masses [Msun]. Required if bbh is not given.
    a1, a2 : array, optional
        Spin magnitudes [0, 1].
    theta1, theta2 : array, optional
        Spin tilt angles [rad].
    phi1, phi2 : array, optional
        Spin azimuthal angles [rad].
    precessing : bool
        True (default) for precessing models, False for nonprecessing.
    mass_spin_model : str or None
        Model for remnant mass and spin. None uses the default.
        Precessing: 'hbr' (default), 'sur7dq4remnant', 'sur7dq4emri'
        Nonprecessing: 'uib' (default), 'hbr', 'sur7dq4remnant', 'sur7dq4emri'
    kick_model : str or None
        Model for kick velocity. None uses the default.
        Precessing: 'gwmodel' (default), 'hlz' (-> CLZM2007), 'sur7dq4remnant'
        Nonprecessing: 'gwmodel_kick_q200' (default), 'hlz' (-> HLZ2014), 'sur3dq8remnant'

    Attributes (after construction)
    ----------
    Mf : array
        Remnant mass [Msun]
    af : array
        Remnant spin magnitude
    vkick : array
        Kick velocity [km/s]
    """

    _PREC_MASS_SPIN = ('hbr', 'sur7dq4remnant', 'sur7dq4emri')
    _PREC_KICK = ('gwmodel', 'hlz', 'sur7dq4remnant')
    _NONPREC_MASS_SPIN = ('uib', 'hbr', 'sur7dq4remnant', 'sur7dq4emri', 'sur3dq8remnant')
    _NONPREC_KICK = ('gwmodel_kick_q200', 'hlz', 'sur3dq8remnant')

    def __init__(self, bbh=None, m1=None, m2=None,
                 a1=None, a2=None, theta1=None, theta2=None,
                 phi1=None, phi2=None,
                 precessing=True,
                 mass_spin_model=None, kick_model=None):

        if bbh is not None:
            self.m1 = np.atleast_1d(np.asarray(bbh.m1, dtype=float))
            self.m2 = np.atleast_1d(np.asarray(bbh.m2, dtype=float))
            self.q = np.atleast_1d(np.asarray(bbh.q, dtype=float))
            self.M = np.atleast_1d(np.asarray(bbh.M, dtype=float))
            self.a1 = np.atleast_1d(np.asarray(bbh.a1, dtype=float))
            self.a2 = np.atleast_1d(np.asarray(bbh.a2, dtype=float))
            self.theta1 = np.atleast_1d(np.asarray(bbh.theta1, dtype=float))
            self.theta2 = np.atleast_1d(np.asarray(bbh.theta2, dtype=float))
            self.delta_phi = np.atleast_1d(np.asarray(bbh.delta_phi, dtype=float))
            self.chi1z = np.atleast_1d(np.asarray(bbh.chi1z, dtype=float))
            self.chi2z = np.atleast_1d(np.asarray(bbh.chi2z, dtype=float))
            self.chi1_vec = np.atleast_2d(np.asarray(bbh.chi1_vec, dtype=float))
            self.chi2_vec = np.atleast_2d(np.asarray(bbh.chi2_vec, dtype=float))
        else:
            self.m1 = np.atleast_1d(np.asarray(m1, dtype=float))
            self.m2 = np.atleast_1d(np.asarray(m2, dtype=float))
            self.a1 = np.atleast_1d(np.asarray(a1, dtype=float))
            self.a2 = np.atleast_1d(np.asarray(a2, dtype=float))
            self.theta1 = np.atleast_1d(np.asarray(theta1, dtype=float))
            self.theta2 = np.atleast_1d(np.asarray(theta2, dtype=float))
            self.q = self.m1 / self.m2
            self.M = self.m1 + self.m2
            self.delta_phi = (np.atleast_1d(np.asarray(phi1, dtype=float))
                              - np.atleast_1d(np.asarray(phi2, dtype=float)))
            self.chi1z = self.a1 * np.cos(self.theta1)
            self.chi2z = self.a2 * np.cos(self.theta2)
            self.chi1_vec, self.chi2_vec = spins_polar_to_cartesian_vectors(
                self.a1, self.a2, self.theta1, self.theta2,
                np.atleast_1d(np.asarray(phi1, dtype=float)),
                np.atleast_1d(np.asarray(phi2, dtype=float)))

        self.precessing = precessing

        if precessing:
            ms_model = mass_spin_model or 'hbr'
            k_model = kick_model or 'gwmodel'
            if ms_model not in self._PREC_MASS_SPIN:
                raise ValueError(f"Precessing mass_spin_model must be one of {self._PREC_MASS_SPIN}")
            if k_model not in self._PREC_KICK:
                raise ValueError(f"Precessing kick_model must be one of {self._PREC_KICK}")
        else:
            ms_model = mass_spin_model or 'uib'
            k_model = kick_model or 'gwmodel_kick_q200'
            if ms_model not in self._NONPREC_MASS_SPIN:
                raise ValueError(f"Nonprecessing mass_spin_model must be one of {self._NONPREC_MASS_SPIN}")
            if k_model not in self._NONPREC_KICK:
                raise ValueError(f"Nonprecessing kick_model must be one of {self._NONPREC_KICK}")

        self.mass_spin_model = ms_model
        self.kick_model = k_model

        self._compute_mass_spin(ms_model)
        self._compute_kick(k_model)

    def _compute_mass_spin(self, model):
        """Dispatch remnant mass and spin computation to the chosen model."""
        if model == 'hbr':
            self._mass_spin_hbr()
        elif model == 'uib':
            self._mass_spin_uib()
        elif model == 'sur7dq4remnant':
            self._mass_spin_kick_sur7dq4()
        elif model == 'sur7dq4emri':
            self._mass_spin_kick_sur7dq4emri()
        elif model == 'sur3dq8remnant':
            self._mass_spin_kick_sur3dq8()

    def _compute_kick(self, model):
        """Dispatch kick velocity computation; skips if surrogate already set it."""
        if model in ('sur7dq4remnant', 'sur3dq8remnant'):
            return
        if model == 'gwmodel':
            self._kick_gwmodel_flow()
        elif model == 'gwmodel_kick_q200':
            self._kick_gwmodel_q200()
        elif model == 'hlz':
            self._kick_hlz()

    def _mass_spin_hbr(self):
        """Compute Mf and af using HBR2016 (precessing, from gwModels)."""
        Mf_frac = bbh_final_mass_precessing_BMR2012(
            self.q, self.a1, self.a2, self.theta1, self.theta2)
        self.Mf = Mf_frac * self.M
        self.af = bbh_final_spin_precessing_HBR2016(
            self.q, self.a1, self.a2, self.theta1, self.theta2, self.delta_phi)

    def _mass_spin_uib(self):
        """Compute Mf and af using UIB2016 (nonprecessing, from gwModels)."""
        Mf_frac = bbh_final_mass_non_precessing_UIB2016(
            self.q, self.chi1z, self.chi2z)
        self.Mf = Mf_frac * self.M
        self.af = bbh_final_spin_non_precessing_UIB2016(
            self.q, self.chi1z, self.chi2z)

    def _surfinbh_spin_vecs(self):
        """Return spin vectors for surfinBH: aligned-spin zeros out x,y."""
        if self.precessing:
            return self.chi1_vec, self.chi2_vec
        s1 = np.column_stack([np.zeros(len(self.m1)),
                              np.zeros(len(self.m1)), self.chi1z])
        s2 = np.column_stack([np.zeros(len(self.m1)),
                              np.zeros(len(self.m1)), self.chi2z])
        return s1, s2

    def _mass_spin_kick_sur7dq4(self):
        """Compute Mf, af, and vkick using NRSur7dq4Remnant surrogate."""
        fit = _get_surfin_fit('NRSur7dq4Remnant')
        s1, s2 = self._surfinbh_spin_vecs()
        N = len(self.m1)
        mf_frac = np.empty(N)
        chif = np.empty((N, 3))
        vkick = np.empty(N)
        for i in range(N):
            mf_i, chif_i, vf_i = fit.all(
                self.q[i], s1[i], s2[i], allow_extrap=True)[:3]
            mf_frac[i] = mf_i
            chif[i] = chif_i
            vkick[i] = np.linalg.norm(vf_i) * C_KMS
        self.Mf = mf_frac * self.M
        self.af = np.linalg.norm(chif, axis=1)
        self.vkick = vkick

    def _mass_spin_kick_sur7dq4emri(self):
        """Compute Mf and af using NRSur7dq4EmriRemnant surrogate (no kick)."""
        fit = _get_surfin_fit('NRSur7dq4EmriRemnant')
        s1, s2 = self._surfinbh_spin_vecs()
        N = len(self.m1)
        mf_frac = np.empty(N)
        chif = np.empty((N, 3))
        for i in range(N):
            mf_i, chif_i = fit.all(self.q[i], s1[i], s2[i])[:2]
            mf_frac[i] = mf_i
            chif[i] = chif_i
        self.Mf = mf_frac * self.M
        self.af = np.linalg.norm(chif, axis=1)

    def _mass_spin_kick_sur3dq8(self):
        """Compute Mf, af, and vkick using NRSur3dq8Remnant surrogate."""
        fit = _get_surfin_fit('NRSur3dq8Remnant')
        s1 = np.column_stack([np.zeros(len(self.m1)),
                              np.zeros(len(self.m1)), self.chi1z])
        s2 = np.column_stack([np.zeros(len(self.m1)),
                              np.zeros(len(self.m1)), self.chi2z])
        N = len(self.m1)
        mf_frac = np.empty(N)
        chif = np.empty((N, 3))
        vkick = np.empty(N)
        for i in range(N):
            mf_i, chif_i, vf_i = fit.all(
                self.q[i], s1[i], s2[i], allow_extrap=True)[:3]
            mf_frac[i] = mf_i
            chif[i] = chif_i
            vkick[i] = np.linalg.norm(vf_i) * C_KMS
        self.Mf = mf_frac * self.M
        self.af = np.linalg.norm(chif, axis=1)
        self.vkick = vkick

    def _kick_gwmodel_flow(self):
        """Compute vkick using the IW2025 normalizing-flow precessing kick model."""
        flow = _get_flow_model()
        vkick = np.zeros(len(self.m1))
        for i in range(len(self.m1)):
            samples = flow.sample(self.q[i], self.a1[i], self.a2[i], num_samples=1)
            vkick[i] = samples[0]
        self.vkick = vkick

    def _kick_gwmodel_q200(self):
        """Compute vkick using the gwModel aligned-spin kick fit (q <= 200)."""
        self.vkick = gwModel_kick_q200(self.q, self.chi1z, self.chi2z)

    def _kick_hlz(self):
        """Compute vkick using the HLZ kick formula.

        Dispatches to the precessing CLZM2007 formula when precessing=True,
        or the nonprecessing HLZ2014 formula when precessing=False.
        """
        if self.precessing:
            self.vkick = bbh_final_kick_precessing_CLZM2007(
                self.q, self.a1, self.a2,
                self.theta1, self.theta2, self.delta_phi)
        else:
            self.vkick = bbh_final_kick_nonprecessing_HLZ2014(
                self.q, self.chi1z, self.chi2z)

    def __repr__(self):
        s = (f"BBHRemnant(n={len(self.m1)}, precessing={self.precessing}, "
             f"mass_spin={self.mass_spin_model}, kick={self.kick_model}")
        if hasattr(self, 'Mf'):
            s += f", Mf=[{self.Mf.min():.1f}, {self.Mf.max():.1f}]"
        if hasattr(self, 'af'):
            s += f", af=[{self.af.min():.3f}, {self.af.max():.3f}]"
        if hasattr(self, 'vkick'):
            s += f", vkick=[{self.vkick.min():.1f}, {self.vkick.max():.1f}] km/s"
        return s + ")"
