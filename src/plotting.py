"""Phase 6 — shared plotting utilities for the manuscript figure set.

This module is intentionally small: it owns the *house style* (one
``rcParams`` block) and the two micro-helpers ``panel_letter`` /
``shared_correlation_colorbar`` so every Phase-6 panel ends up
visually consistent.

Use::

    from src.plotting import consistent_style, panel_letter
    consistent_style()
    fig, axes = plt.subplots(...)
    panel_letter(axes[0, 0], "a")
"""
from __future__ import annotations
from typing import Iterable
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ---------------------------------------------------------------------------
# House style
# ---------------------------------------------------------------------------

def consistent_style() -> None:
    """Apply the manuscript figure style globally (rcParams patch).

    Targets a single-column-friendly look at 300 dpi: sans-serif body
    text, ~8 pt labels, thin axes, no top/right spines.
    """
    plt.rcParams.update({
        # Geometry
        "figure.dpi":          110,
        "savefig.dpi":         300,
        "savefig.bbox":        "tight",
        # Typography
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":           8.5,
        "axes.labelsize":      9,
        "axes.titlesize":      9.5,
        "axes.titleweight":    "bold",
        "legend.fontsize":     7,
        "xtick.labelsize":     7,
        "ytick.labelsize":     7,
        # Lines
        "axes.linewidth":      0.8,
        "lines.linewidth":     1.5,
        "grid.linewidth":      0.4,
        "grid.alpha":          0.30,
        # Spines
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        # Math
        "mathtext.fontset":    "dejavusans",
    })


# ---------------------------------------------------------------------------
# Panel labelling
# ---------------------------------------------------------------------------

def panel_letter(ax, letter: str,
                  x: float = -0.16, y: float = 1.08,
                  fontsize: int = 12) -> None:
    """Bold panel letter in the top-left exterior of an axes."""
    ax.text(x, y, f"({letter})",
             transform=ax.transAxes,
             fontsize=fontsize, fontweight="bold",
             va="top", ha="left")


# ---------------------------------------------------------------------------
# Shared correlation colorbar
# ---------------------------------------------------------------------------

def shared_correlation_colorbar(fig, axes_iter, cmap: str = "RdBu_r",
                                  vmin: float = -1.0, vmax: float = 1.0,
                                  label: str = "Pearson r",
                                  shrink: float = 0.85):
    """Attach a single colorbar shared by all correlation-heatmap axes
    listed in ``axes_iter``. Returns the colorbar handle.

    Both (d) and (f) of Figure 9 plot a 40x40 Pearson-r heatmap on the
    same [-1, 1] scale; the shared bar avoids duplicate legends.
    """
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax),
                          cmap=plt.get_cmap(cmap))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes_iter),
                          location="right", shrink=shrink, pad=0.02,
                          fraction=0.02)
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    return cbar
