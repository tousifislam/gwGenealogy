#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: bbh.py
#
#    Container class for binary black hole parameters.
#    Accepts either (m1, m2) or (q/small_q, M) as mass input, plus
#    spin magnitudes and angles, and computes all derived quantities:
#    component masses, mass ratios, chirp mass, eta, spin vectors,
#    chi_eff, chi_p, delta, chi_tilde, etc.
#
#    Works with both scalar and array inputs.
#
#    AUTHOR: Tousif Islam
#    CREATED: 06-07-2026
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from ..core.conversions import (m1_m2_to_mchirp, m1_m2_to_q, m1_m2_to_eta,
                                chi_eff, chi_p,
                                delta_parallel, delta_perp,
                                chi_tilde_parallel, chi_tilde_perp)
from ..core.coordinates import (spins_polar_to_cartesian_vectors,
                                polar_to_cartesian)


class BBHs:
    """Container for binary black hole parameters.

    Parameters
    ----------
    m1 : float or array, optional
        Primary mass (m1 >= m2). Provide with m2, OR provide M with q/small_q.
    m2 : float or array, optional
        Secondary mass.
    M : float or array, optional
        Total mass. Provide with q or small_q.
    q : float or array, optional
        Mass ratio m1/m2 >= 1.
    small_q : float or array, optional
        Mass ratio m2/m1 in (0, 1].
    a1 : float or array
        Primary dimensionless spin magnitude.
    a2 : float or array
        Secondary dimensionless spin magnitude.
    theta1 : float or array
        Primary spin tilt angle [rad].
    theta2 : float or array
        Secondary spin tilt angle [rad].
    phi1 : float or array
        Primary azimuthal spin angle [rad].
    phi2 : float or array
        Secondary azimuthal spin angle [rad].
    z : float or array, optional
        Redshift.
    """

    def __init__(self, m1=None, m2=None, M=None, q=None, small_q=None,
                 a1=0.0, a2=0.0,
                 theta1=0.0, theta2=0.0,
                 phi1=0.0, phi2=0.0,
                 z=None):

        self._resolve_masses(m1, m2, M, q, small_q)
        self._set_spins(a1, a2, theta1, theta2, phi1, phi2)
        self._compute_derived_spins()

        self.z = np.asarray(z, dtype=float) if z is not None else None

    def _resolve_masses(self, m1, m2, M, q, small_q):
        """Compute all mass parameters from whichever pair is provided."""
        if m1 is not None and m2 is not None:
            self.m1 = np.asarray(m1, dtype=float)
            self.m2 = np.asarray(m2, dtype=float)
        elif M is not None and (q is not None or small_q is not None):
            M = np.asarray(M, dtype=float)
            if small_q is not None:
                sq = np.asarray(small_q, dtype=float)
            else:
                sq = 1.0 / np.asarray(q, dtype=float)
            self.m1 = M / (1.0 + sq)
            self.m2 = M * sq / (1.0 + sq)
        else:
            raise ValueError("Provide either (m1, m2) or (M with q or small_q)")

        self.M = self.m1 + self.m2
        self.q = m1_m2_to_q(self.m1, self.m2)
        self.small_q = 1.0 / self.q
        self.eta = m1_m2_to_eta(self.m1, self.m2)
        self.mchirp = m1_m2_to_mchirp(self.m1, self.m2)

    def _set_spins(self, a1, a2, theta1, theta2, phi1, phi2):
        """Store spin magnitudes and angles, build Cartesian vectors."""
        self.a1 = np.asarray(a1, dtype=float)
        self.a2 = np.asarray(a2, dtype=float)
        self.theta1 = np.asarray(theta1, dtype=float)
        self.theta2 = np.asarray(theta2, dtype=float)
        self.phi1 = np.asarray(phi1, dtype=float)
        self.phi2 = np.asarray(phi2, dtype=float)

        self.cos_theta1 = np.cos(self.theta1)
        self.cos_theta2 = np.cos(self.theta2)

        chi1x, chi1y, chi1z = polar_to_cartesian(self.a1, self.theta1, self.phi1)
        chi2x, chi2y, chi2z = polar_to_cartesian(self.a2, self.theta2, self.phi2)

        self.chi1z = chi1z
        self.chi2z = chi2z
        self.a1_perp = np.sqrt(chi1x**2 + chi1y**2)
        self.a2_perp = np.sqrt(chi2x**2 + chi2y**2)
        self.chi1_vec = np.stack([chi1x, chi1y, chi1z], axis=-1)
        self.chi2_vec = np.stack([chi2x, chi2y, chi2z], axis=-1)

    def _compute_derived_spins(self):
        """Compute chi_eff, chi_p, delta, chi_tilde from stored parameters."""
        self.chi_eff = chi_eff(self.q, self.chi1z, self.chi2z)
        self.chi_p = chi_p(self.q, self.a1_perp, self.a2_perp)

        self.delta_phi = self.phi1 - self.phi2

        self.delta_parallel = delta_parallel(
            self.q, self.a1, self.a2, self.theta1, self.theta2)
        self.delta_perp = delta_perp(
            self.q, self.a1, self.a2, self.theta1, self.theta2, self.delta_phi)

        self.chi_tilde_parallel = chi_tilde_parallel(
            self.q, self.a1, self.a2, self.theta1, self.theta2)
        self.chi_tilde_perp = chi_tilde_perp(
            self.q, self.a1, self.a2, self.theta1, self.theta2, self.delta_phi)
