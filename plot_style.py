#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modern Plot Styling for Calcium Transient Analysis

This module provides publication-quality plot styling using Seaborn
and matplotlib, replacing the legacy pp_style module.

Author: hjoca
Created: 2026-01-13
"""
import matplotlib.pyplot as plt
import seaborn as sns


def setup_plot_style(context='notebook', style='ticks', palette='husl', font_scale=1.2):
    """
    Configure modern plot styling for scientific figures.
    
    Args:
        context: Seaborn context ('paper', 'notebook', 'talk', 'poster')
        style: Seaborn style ('white', 'dark', 'whitegrid', 'darkgrid', 'ticks')
        palette: Color palette ('husl', 'Set2', 'deep', 'muted', etc.)
        font_scale: Scaling factor for fonts
    """
    # Set seaborn style
    sns.set_context(context, font_scale=font_scale)
    sns.set_style(style)
    sns.set_palette(palette)
    
    # Additional matplotlib customizations for scientific plots
    plt.rcParams.update({
        # Figure
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        
        # Axes
        'axes.linewidth': 1.5,
        'axes.edgecolor': '#333333',
        'axes.labelweight': 'bold',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        
        # Lines
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        
        # Ticks
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.minor.width': 1.0,
        'ytick.minor.width': 1.0,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Grid
        'grid.linewidth': 0.8,
        'grid.alpha': 0.3,
        
        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fontsize': 10,
        
        # Save
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
    })


def apply_calcium_style():
    """
    Apply optimized styling specifically for calcium imaging plots.
    Uses clean, publication-ready aesthetics.
    """
    setup_plot_style(
        context='notebook',
        style='ticks',
        palette='husl',
        font_scale=1.2
    )
    
    # Remove top and right spines for cleaner look
    sns.despine()


def apply_publication_style():
    """
    Apply Nature/Science journal-style formatting.
    More conservative styling for publication figures.
    """
    setup_plot_style(
        context='paper',
        style='white',
        palette='deep',
        font_scale=1.0
    )
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.5,
    })
    
    sns.despine()


