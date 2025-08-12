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

# Your gc_vesc data
gc_vesc = np.array([ 47.4       ,  10.9       ,  34.        ,   1.1       ,
        21.4       ,   1.9       ,   3.3       ,   2.2       ,
        19.2       ,  41.1       ,  27.6       ,  15.3       ,
        18.2       ,   3.6       ,  53.1       ,   2.4       ,
         2.4       ,  17.3       ,   2.6       ,   2.        ,
        14.2       ,  15.5       ,   5.6       ,  14.8       ,
         8.1       ,  22.2       ,  25.9       ,   6.        ,
        62.2       ,  32.        ,  41.8       ,   0.9       ,
         6.5       ,  24.2       ,  33.6       ,  10.5       ,
        44.8       ,   2.1       ,  12.5       ,  30.4       ,
        25.2       ,  24.3       ,  30.6       ,  11.1       ,
         2.3       ,  12.8       ,  44.6       ,  17.        ,
        11.5       ,  18.8       ,  46.4       ,  15.        ,
         7.6       ,  15.7       ,   5.5       ,  32.4       ,
        17.6       ,  25.3       ,  22.2       ,  15.5       ,
        23.9       ,  19.8       ,   4.3       ,  59.3       ,
        44.8       ,  27.6       ,   6.1       ,   7.7       ,
        26.8       ,  27.4       ,   6.3       ,  22.8       ,
        30.3       ,  20.6       ,  32.4       ,  36.4       ,
        18.4       ,  13.2       ,  21.6       ,  31.1       ,
         6.7       ,  17.4       ,  23.1       ,  20.3       ,
        16.1       ,  13.1       ,   9.1       ,  87.9       ,
        31.8       ,  37.9       ,  11.6       ,  60.        ,
        21.5       ,  23.6       ,  35.6       ,  19.7       ,
        10.3       ,  14.8       ,  57.5       ,  55.6       ,
        11.8       ,  69.4       ,  26.4       ,  27.9       ,
        15.7       ,  25.9       ,   6.8       ,  10.8       ,
        29.1       ,  15.4       ,  37.3       ,  35.2       ,
        32.4       ,  10.2       ,  19.        ,  22.5       ,
        15.1       ,  12.8       ,  37.3       ,  25.2       ,
        11.2       ,   6.8       ,  32.8       ,   8.4       ,
        16.7       ,  14.3       ,  13.1       ,  28.4       ,
         9.1       ,  16.7       ,  28.        ,  41.1       ,
        24.3       ,  26.7       ,  20.        ,  19.1       ,
        35.5       ,  11.7       ,  25.3       ,  30.4       ,
        19.2       ,  68.9       ,  11.6       ,  21.5       ,
        20.5       ,  31.3       ,  25.6       ,  21.9       ,
         4.6       ,  14.6       ,   4.5       ,  18.1       ,
         6.        ,   4.        ,   2.7       ,  10.2       ,
        48.4       ,  21.4       ,  12.4       ,  15.7       ,
         1.9       ,  48.9       ,  43.6       ,  21.        ,
         2.5       ,   1.8       ,   4.4       ,  21.44      ,
        10.84      ,  13.26      ,  13.91      ,  87.41      ,
         6.22      ,   9.81      ,   6.89      ,  11.46      ,
         6.35      ,  22.82      ,  25.25      ,  22.79      ,
        17.05      ,  42.57      ,  66.14      ,  18.2       ,
        20.33      ,   8.03      ,  34.52      ,   8.58      ,
        17.96      ,  18.64      ,   6.9       ,  23.12      ,
         7.9       ,   8.67      ,  18.18      ,   4.19      ,
         5.96      ,   7.42      ,   8.31      ,  10.29      ,
        76.28      ,  17.49      ,  33.79      ,   9.87      ,
         9.41      ,  12.07      ,   8.9       ,   8.71      ,
        11.31      ,  66.56      ,  20.34      ,  12.01      ,
        31.29      ,   8.78      ,   9.73      ,   9.54      ,
        18.19      ,  39.62      ,  11.22      ,   5.79      ,
         7.83      ,  13.64      ,  13.15      ,   9.62      ,
        51.08      ,  10.47      ,  77.17      ,  35.98      ,
        23.29      ,  21.37      ,  14.14      ,   7.9       ,
         9.93      ,   5.74      ,   8.3       ,  17.25      ,
        32.77      ,  76.73      ,  27.37      ,  11.55      ,
        60.35      ,   8.36      ,  20.96      ,  13.12      ,
        19.62      ,   8.37      ,  15.18      ,  53.65      ,
        28.86      ,  33.92      ,  24.22      ,  22.75      ,
         7.84      ,   9.3       ,   7.62      ,  16.1       ,
        17.61      ,  11.79      ,  16.51      ,  56.16      ,
        21.68      ,  51.43      ,   5.65      ,  22.38      ,
        83.41      ,   5.59      ,  13.92      ,  30.54      ,
        26.07      , 212.15      ,  15.29      ,  54.59      ,
        14.09      ,  27.13      ,  21.23      ,  28.89      ,
        79.45      ,  17.17      ,  26.34      ,  17.54      ,
        12.97      ,  11.9       ,  60.43      ,   8.51      ,
        14.74      ,  27.55      ,  33.77      ,  17.54      ,
        18.29      ,  10.71      ,  10.81      ,  12.35      ,
        77.79      ,  25.45      ,  29.52      ,  20.76      ,
        17.99      ,  16.17      ,  10.73      ,   5.93      ,
        13.92      ,   6.45      ,   9.92      ,  24.71      ,
         9.72      ,  19.82      ,   7.08      ,   6.26      ,
        14.45      ,  21.47      ,  10.66      ,   8.42      ,
        16.42      ,   7.23      ,  30.55      ,  33.41      ,
        16.04      ,  30.14      ,  72.34      ,  12.34      ,
        14.36      ,  17.82      ,   9.91      ,  18.46      ,
         6.32      ,   8.13      ,  12.18      ,  13.1       ,
       241.71      ,  11.31      ,  28.64      ,  26.34      ,
        12.07      ,  35.61      ,  10.29      ,  53.85      ,
        29.76      ,  35.51      ,  22.1       ,  33.23      ,
        22.25      ,  26.3       ,  11.92214112,  13.11435523,
        13.45498783,  13.62530414,  13.79562044,  15.66909976,
        16.02018121,  16.35036496,  16.61023468,  16.86131387,
        16.86131387,  16.86131387,  18.05352798,  18.39416058,
        18.63941606,  19.9814139 ,  20.0973236 ,  21.28953771,
        21.80048662,  22.14111922,  22.65206813,  24.69586375,
        24.71996511,  25.63260341,  27.42092457,  27.59124088,
        27.59124088,  27.66853301,  27.76155718,  28.23516604,
        28.61313869,  29.12408759,  30.1459854 ,  31.16726612,
        31.50851582,  33.21167883,  33.21167883,  33.89294404,
        34.06326034,  34.06326034,  34.34610706,  36.29327729,
        36.61800487,  37.29927007,  37.46958637,  38.15085158,
        38.15085158,  38.20205314,  39.00243309,  40.1946472 ,
        41.08669643,  42.06812652,  42.40875912,  45.30413625,
        45.47445255,  47.51824818,  47.91027801,  48.02919708,
        48.19951338,  50.51824818,  51.35036496,  52.45742092,
        58.41849148,  58.67396594,  58.91970803,  61.99513382,
        63.69829684,  68.80778589,  72.89537713,  82.94403893,
        84.64720195,  85.32846715,  93.67396594,  97.93187348,
       102.18978102, 105.89550519, 105.98481281, 106.00973236,
       106.2238192 , 106.27471107, 106.39971214, 106.54665615,
       106.78275528, 107.10390726, 111.86818999, 111.88969992,
       111.89125959, 111.94571168, 112.34817682, 112.4541768 ,
       112.80230553, 112.89537713, 112.91615977, 112.94676763,
       113.10264035, 113.12810748, 113.74381917, 113.90289759,
       113.91240876, 119.44657512, 119.50747644, 119.5856374 ,
       119.70308698, 119.8702352 , 120.02686537, 120.23404009,
       120.52119349, 124.33090024])

def sample_globular_cluster_vesc_distribution(n_samples=1000):
    """Sample from globular cluster escape velocity distribution using Monte Carlo with CDF"""
    # Sort the data and create CDF
    vesc_sorted = np.sort(gc_vesc)
    cdf = np.linspace(0, 1, len(vesc_sorted))
    
    # Generate random numbers and interpolate
    random_probs = np.random.uniform(0, 1, n_samples)
    samples = np.interp(random_probs, cdf, vesc_sorted)
    
    return samples

def sample_uniform_globular_cluster_vesc_distribution(n_samples=1000, min_vesc=None, max_vesc=None):
    """Sample from uniform escape velocity distribution for globular clusters
    
    Parameters:
    n_samples: int, number of samples to generate
    min_vesc: float, minimum escape velocity (km/s). If None, uses min from gc_vesc
    max_vesc: float, maximum escape velocity (km/s). If None, uses max from gc_vesc
    
    Returns:
    samples: array of escape velocity samples uniformly distributed between min and max
    """
    # Use gc_vesc array limits if not provided
    if min_vesc is None:
        min_vesc = np.min(gc_vesc)
    if max_vesc is None:
        max_vesc = np.max(gc_vesc)
    
    # Generate uniform samples
    samples = np.random.uniform(min_vesc, max_vesc, n_samples)
    
    return samples

def sample_gaussian_globular_cluster_vesc_distribution(n_samples=1000, median_vesc=None, std_vesc=None):
    """Sample from Gaussian escape velocity distribution for globular clusters
    
    Parameters:
    n_samples: int, number of samples to generate
    median_vesc: float, median escape velocity (km/s). If None, uses median from gc_vesc
    std_vesc: float, standard deviation (km/s). If None, uses std from gc_vesc
    
    Returns:
    samples: array of escape velocity samples from Gaussian distribution
    """
    # Use gc_vesc statistics if not provided
    if median_vesc is None:
        median_vesc = np.median(gc_vesc)
    if std_vesc is None:
        std_vesc = np.std(gc_vesc)
    
    # Generate Gaussian samples using median as mean
    samples = np.random.normal(median_vesc, std_vesc, n_samples)
    
    # Ensure positive values (escape velocities can't be negative)
    samples = np.abs(samples)
    
    return samples