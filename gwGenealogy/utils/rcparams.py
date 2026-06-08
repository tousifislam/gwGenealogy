#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#     FILE: rcparams.py
#     Matplotlib plotting configuration for gwGenealogy
#
#     AUTHOR: Tousif Islam
#     CREATED: 07-02-2024
#     REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"


def set_rcparams():
    """
    Apply gwGenealogy matplotlib style settings.

    Call this explicitly to configure matplotlib for publication-quality plots.
    Requires a LaTeX installation for text.usetex.
    """
    import matplotlib.pyplot as plt

    plt.rc('figure', figsize=(8, 5))

    try:
        plt.rcParams.update({
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath}',
        })
        import matplotlib
        matplotlib.texmanager.TexManager().get_grey("$x$")
    except Exception:
        import logging
        logging.getLogger(__name__).info("LaTeX not available, falling back to mathtext rendering")
        plt.rcParams.update({
            'text.usetex': False,
        })

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Georgia'],
        'mathtext.fontset': 'cm',
        'lines.linewidth': 1.8,
        'font.size': 16,
        'xtick.labelsize': 'medium',
        'ytick.labelsize': 'medium',
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.major.size': 4.,
        'ytick.major.size': 4.,
        'ytick.right': True,
        'axes.labelsize': 'medium',
        'axes.titlesize': 'medium',
        'axes.grid': True,
        'grid.alpha': 0.5,
        'lines.markersize': 12,
        'legend.borderpad': 0.2,
        'legend.fancybox': True,
        'legend.fontsize': 14,
        'legend.framealpha': 0.7,
        'legend.handletextpad': 0.5,
        'legend.labelspacing': 0.2,
        'legend.loc': 'best',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.dpi': 80,
        'pdf.compression': 9,
    })
