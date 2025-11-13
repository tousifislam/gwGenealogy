#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: globular_clusters_vesc.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED: 
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np

# SAURON early type galaxy data
etg_vesc = np.array([ 190.85,  191.28,  215.97,  243.48,  272.9,  299.92,  319.93,
                      336.68,  376.32,  386.16,  440.95,  454.34,  470.84,  531.87,
                      545.49,  597.14,  612.91,  674.96,  700.52,  777.03,  841.25,
                      857.23,  878.64,  899.32,  909.  ,  988.32, 1016.92, 1056.45,
                      1085.11, 1162.36, 1184.03, 1246.02, 1261.4 ])

def sample_andromeda_galaxy_vesc_distribution_Kafle2018(n_samples=1000, seed=42):
    """
    Sample from Andromeda galaxy escape velocity distribution based on Kafle et al. 2018
    (https://academic.oup.com/mnras/article/475/3/4043/4797184)
    
    Based on Kafle et al. 2018 measurements of M31 escape velocity:
    Mean ~ 470 km/s, Standard deviation ~ 40 km/s
    
    Parameters:
    n_samples: int, number of samples to generate
    seed: int, random seed for reproducible results
    
    Returns:
    samples: array of escape velocity samples from M31 distribution (km/s)
    """
    # Create random number generator with specified seed
    rng = np.random.default_rng(seed)
    
    # Generate samples from normal distribution (mean=470, std=40)
    samples = rng.normal(470, 40, n_samples)
    
    # Ensure positive values (escape velocities can't be negative)
    samples = np.abs(samples)
    
    return samples

def sample_milky_way_vesc_distribution_KH2021(n_samples=1000, seed=42):
    """
    Sample from Milky Way galaxy escape velocity distribution based on Helmer Koppelman and Helmi 2021
    (https://www.aanda.org/articles/aa/full_html/2021/05/aa38777-20/aa38777-20.html)
    
    Based on Helmer H. Koppelman and Amina Helmi measurements of Milky Way escape velocity:
    Mean ~ 497 km/s, Standard deviation ~ 8 km/s
    
    Parameters:
    n_samples: int, number of samples to generate
    seed: int, random seed for reproducible results
    
    Returns:
    samples: array of escape velocity samples from Milky Way distribution (km/s)
    """
    # Create random number generator with specified seed
    rng = np.random.default_rng(seed)
    
    # Generate samples from normal distribution (mean=497, std=8)
    samples = rng.normal(497, 8, n_samples)
    
    # Ensure positive values (escape velocities can't be negative)
    samples = np.abs(samples)
    
    return samples

def sample_early_type_galaxies_vesc_distribution(n_samples=1000):
    """
    Sample from early type galaxies escape velocity distribution using Monte Carlo with CDF
    """
    # Sort the data and create CDF
    vesc_sorted = np.sort(etg_vesc)
    cdf = np.linspace(0, 1, len(vesc_sorted))
    
    # Generate random numbers and interpolate
    random_probs = np.random.uniform(0, 1, n_samples)
    samples = np.interp(random_probs, cdf, vesc_sorted)
    
    return samples

def sample_uniform_early_type_galaxies_vesc_distribution(n_samples=1000, min_vesc=None, max_vesc=None):
    """
    Sample from uniform escape velocity distribution for early type galaxies
    
    Parameters:
    n_samples: int, number of samples to generate
    min_vesc: float, minimum escape velocity (km/s). If None, uses min from gc_vesc
    max_vesc: float, maximum escape velocity (km/s). If None, uses max from gc_vesc
    
    Returns:
    samples: array of escape velocity samples uniformly distributed between min and max
    """
    # Use gc_vesc array limits if not provided
    if min_vesc is None:
        min_vesc = np.min(etg_vesc)
    if max_vesc is None:
        max_vesc = np.max(etg_vesc)
    
    # Generate uniform samples
    samples = np.random.uniform(min_vesc, max_vesc, n_samples)
    
    return samples

def sample_gaussian_early_type_galaxies_vesc_distribution(n_samples=1000, median_vesc=None, std_vesc=None):
    """
    Sample from Gaussian escape velocity distribution for early type galaxies
    
    Parameters:
    n_samples: int, number of samples to generate
    median_vesc: float, median escape velocity (km/s). If None, uses median from gc_vesc
    std_vesc: float, standard deviation (km/s). If None, uses std from gc_vesc
    
    Returns:
    samples: array of escape velocity samples from Gaussian distribution
    """
    # Use gc_vesc statistics if not provided
    if median_vesc is None:
        median_vesc = np.median(etg_vesc)
    if std_vesc is None:
        std_vesc = np.std(etg_vesc)
    
    # Generate Gaussian samples using median as mean
    samples = np.random.normal(median_vesc, std_vesc, n_samples)
    
    # Ensure positive values (escape velocities can't be negative)
    samples = np.abs(samples)
    
    return samples

def sample_uniform_dwarf_elliptical_galaxies_vesc_distribution(n_samples=1000, min_vesc=20, max_vesc=150):
    """
    Sample from uniform escape velocity distribution for dwarf elliptical galaxies
    Based on Fig 4 of https://arxiv.org/pdf/astro-ph/0402057
    
    Parameters:
    n_samples: int, number of samples to generate
    min_vesc: float, minimum escape velocity (km/s). If None, uses min from gc_vesc
    max_vesc: float, maximum escape velocity (km/s). If None, uses max from gc_vesc
    
    Returns:
    samples: array of escape velocity samples uniformly distributed between min and max
    """
    # Generate uniform samples
    samples = np.random.uniform(min_vesc, max_vesc, n_samples)
    
    return samples

def sample_gaussian_dwarf_elliptical_galaxies_vesc_distribution(n_samples=1000, median_vesc=80, std_vesc=60):
    """
    Sample from Gaussian escape velocity distribution for dwarf elliptical galaxies
    Based on Fig 4 of https://arxiv.org/pdf/astro-ph/0402057
    
    Parameters:
    n_samples: int, number of samples to generate
    median_vesc: float, median escape velocity (km/s). If None, uses median from gc_vesc
    std_vesc: float, standard deviation (km/s). If None, uses std from gc_vesc
    
    Returns:
    samples: array of escape velocity samples from Gaussian distribution
    """
    # Generate Gaussian samples using median as mean
    samples = np.random.normal(median_vesc, std_vesc, n_samples)
    
    # Ensure positive values (escape velocities can't be negative)
    samples = np.abs(samples)
    
    return samples

def sample_uniform_dwarf_spheroidal_galaxies_vesc_distribution(n_samples=1000, min_vesc=2, max_vesc=20):
    """
    Sample from uniform escape velocity distribution for dwarf spherical galaxies
    Based on Fig 4 of https://arxiv.org/pdf/astro-ph/0402057
    
    Parameters:
    n_samples: int, number of samples to generate
    min_vesc: float, minimum escape velocity (km/s). If None, uses min from gc_vesc
    max_vesc: float, maximum escape velocity (km/s). If None, uses max from gc_vesc
    
    Returns:
    samples: array of escape velocity samples uniformly distributed between min and max
    """
    # Generate uniform samples
    samples = np.random.uniform(min_vesc, max_vesc, n_samples)
    
    return samples

def sample_gaussian_dwarf_spheroidal_galaxies_vesc_distribution(n_samples=1000, median_vesc=11, std_vesc=8):
    """
    Sample from Gaussian escape velocity distribution for dwarf spheroidal galaxies
    Based on Fig 4 of https://arxiv.org/pdf/astro-ph/0402057
    
    Parameters:
    n_samples: int, number of samples to generate
    median_vesc: float, median escape velocity (km/s). If None, uses median from gc_vesc
    std_vesc: float, standard deviation (km/s). If None, uses std from gc_vesc
    
    Returns:
    samples: array of escape velocity samples from Gaussian distribution
    """
    # Generate Gaussian samples using median as mean
    samples = np.random.normal(median_vesc, std_vesc, n_samples)
    
    # Ensure positive values (escape velocities can't be negative)
    samples = np.abs(samples)
    
    return samples