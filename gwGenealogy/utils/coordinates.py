#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: coordinates.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np


# ---- Generic coordinate transforms ----

def polar_to_cartesian(r, theta, phi):
    """
    Convert spherical polar coordinates to Cartesian.

    Parameters
    ----------
    r : float or array
        Radial coordinate (e.g. spin magnitude)
    theta : float or array
        Polar angle in radians (0 = +z axis)
    phi : float or array
        Azimuthal angle in radians

    Returns
    -------
    x, y, z : float or array
    """
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    sin_theta = np.sin(theta)
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def cartesian_to_polar(x, y, z):
    """
    Convert Cartesian coordinates to spherical polar.

    Parameters
    ----------
    x, y, z : float or array
        Cartesian coordinates

    Returns
    -------
    r : float or array
        Radial coordinate (>= 0)
    theta : float or array
        Polar angle in radians [0, pi]
    phi : float or array
        Azimuthal angle in radians [0, 2*pi)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    r = np.sqrt(x**2 + y**2 + z**2)
    # guard against r=0 (zero vector) and floating-point overshoot outside [-1,1]
    theta = np.arccos(np.clip(z / np.maximum(r, 1e-30), -1.0, 1.0))
    phi = np.arctan2(y, x) % (2 * np.pi)
    return r, theta, phi


# ---- Spin-specific coordinate transforms ----

def spins_polar_to_cartesian_vectors(a1, a2, theta1, theta2, phi1, phi2):
    """
    Convert spin magnitudes and angles to 3D Cartesian spin vectors.

    Parameters
    ----------
    a1, a2 : array
        Dimensionless spin magnitudes
    theta1, theta2 : array
        Polar (tilt) angles in radians
    phi1, phi2 : array
        Azimuthal angles in radians

    Returns
    -------
    chi1 : array of shape (n_samples, 3)
        3D spin vectors for the primary black hole
    chi2 : array of shape (n_samples, 3)
        3D spin vectors for the secondary black hole
    """
    x1, y1, z1 = polar_to_cartesian(a1, theta1, phi1)
    x2, y2, z2 = polar_to_cartesian(a2, theta2, phi2)

    chi1 = np.column_stack([x1, y1, z1])
    chi2 = np.column_stack([x2, y2, z2])
    return chi1, chi2


def spins_cartesian_vectors_to_polar(chi1_vec, chi2_vec):
    """
    Convert 3D Cartesian spin vectors to magnitudes and angles.

    Parameters
    ----------
    chi1_vec : array of shape (n_samples, 3) or (3,)
        3D spin vectors for the primary black hole
    chi2_vec : array of shape (n_samples, 3) or (3,)
        3D spin vectors for the secondary black hole

    Returns
    -------
    a1, a2 : array
        Dimensionless spin magnitudes
    theta1, theta2 : array
        Polar (tilt) angles in radians
    phi1, phi2 : array
        Azimuthal angles in radians
    """
    chi1_vec = np.atleast_2d(chi1_vec)
    chi2_vec = np.atleast_2d(chi2_vec)

    a1, theta1, phi1 = cartesian_to_polar(chi1_vec[:, 0], chi1_vec[:, 1], chi1_vec[:, 2])
    a2, theta2, phi2 = cartesian_to_polar(chi2_vec[:, 0], chi2_vec[:, 1], chi2_vec[:, 2])

    return a1, a2, theta1, theta2, phi1, phi2
