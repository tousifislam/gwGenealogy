#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: plotting.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED: 02-28-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import matplotlib.pyplot as plt


def plot_mass_spin_generations(data_dict, ax=None):
    """
    Scatter plot of remnant mass vs spin colored by generation.

    Parameters:
    -----------
    data_dict : dict
        Output from run_population_mergers: {gen: {'m': array, 'spin': array}}
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, creates a new figure.

    Returns:
    --------
    fig, ax : matplotlib Figure and Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.figure

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(data_dict)))

    for i, (gen, vals) in enumerate(sorted(data_dict.items())):
        if len(vals['m']) == 0:
            continue
        ax.scatter(vals['m'], vals['spin'], s=10, alpha=0.4,
                   color=colors[i], label=f'{gen}g ({len(vals["m"])})')

    ax.set_xlabel(r'Mass [$M_\odot$]')
    ax.set_ylabel(r'Spin $\chi$')
    ax.legend(fontsize=10, markerscale=2)
    ax.set_title('Mass vs Spin by Generation')

    return fig, ax


def plot_retention_heatmap(p_ret, q_values, chi_max_values, v_esc=None, ax=None):
    """
    2D heatmap of retention probability as a function of q and chi_max.

    Parameters:
    -----------
    p_ret : 2D array
        Retention probability array of shape (len(q_values), len(chi_max_values))
    q_values : array
        Mass ratio values
    chi_max_values : array
        Maximum spin magnitude values
    v_esc : float or None
        Escape velocity for title annotation
    ax : matplotlib.axes.Axes or None
        Axes to plot on

    Returns:
    --------
    fig, ax : matplotlib Figure and Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.figure

    im = ax.pcolormesh(q_values, chi_max_values, p_ret.T,
                       cmap='viridis', shading='auto')
    plt.colorbar(im, ax=ax, label=r'$p_{\rm ret}$')
    ax.set_xlabel(r'Mass ratio $q$')
    ax.set_ylabel(r'$\chi_{\rm max}$')

    title = 'Retention Probability'
    if v_esc is not None:
        title += f' ($v_{{\\rm esc}}={v_esc}$ km/s)'
    ax.set_title(title)

    return fig, ax


def plot_imbh_probability_heatmap(p_imbh, x_values, y_values, ax=None,
                                  xlabel=None, ylabel=None):
    """
    2D heatmap of IMBH formation probability.

    Parameters:
    -----------
    p_imbh : 2D array
        IMBH probability array
    x_values, y_values : array
        Grid values for x and y axes
    ax : matplotlib.axes.Axes or None
        Axes to plot on
    xlabel, ylabel : str or None
        Custom axis labels

    Returns:
    --------
    fig, ax : matplotlib Figure and Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.figure

    im = ax.pcolormesh(x_values, y_values, p_imbh.T,
                       cmap='inferno', shading='auto')
    plt.colorbar(im, ax=ax, label=r'$p_{\rm IMBH}$')
    ax.set_xlabel(xlabel or 'x')
    ax.set_ylabel(ylabel or 'y')
    ax.set_title('IMBH Formation Probability')

    return fig, ax


def plot_kick_distribution(kick_samples, v_esc=None, ax=None, bins=50, label=None):
    """
    Histogram of kick velocity distribution with optional v_esc line.

    Parameters:
    -----------
    kick_samples : array
        Array of kick velocities in km/s
    v_esc : float or None
        Escape velocity to mark with vertical line
    ax : matplotlib.axes.Axes or None
        Axes to plot on
    bins : int
        Number of histogram bins (default: 50)
    label : str or None
        Label for the histogram

    Returns:
    --------
    fig, ax : matplotlib Figure and Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.figure

    ax.hist(kick_samples, bins=bins, density=True, alpha=0.7, label=label)

    if v_esc is not None:
        ax.axvline(v_esc, color='red', linestyle='--', linewidth=2,
                   label=f'$v_{{\\rm esc}}={v_esc}$ km/s')

    ax.set_xlabel('Kick velocity [km/s]')
    ax.set_ylabel('Probability density')
    ax.set_title('Kick Velocity Distribution')
    if label is not None or v_esc is not None:
        ax.legend()

    return fig, ax


def plot_generation_histogram(result_dict, ax=None):
    """
    Bar chart showing the number of BHs at each generation.

    Parameters:
    -----------
    result_dict : dict
        Output from run_population_mergers: {gen: {'m': array, ...}}
        OR output from run_hierarchical_merger_mc with 'final_generations' key
    ax : matplotlib.axes.Axes or None
        Axes to plot on

    Returns:
    --------
    fig, ax : matplotlib Figure and Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.figure

    if 'final_generations' in result_dict:
        # MC output format
        gens = result_dict['final_generations']
        unique, counts = np.unique(gens, return_counts=True)
        ax.bar(unique, counts, color='steelblue', alpha=0.8)
    else:
        # Population merger output format
        gens = sorted(result_dict.keys())
        counts = [len(result_dict[g]['m']) for g in gens]
        ax.bar(gens, counts, color='steelblue', alpha=0.8)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Number of BHs')
    ax.set_title('Generation Distribution')

    return fig, ax
