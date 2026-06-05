"""
Visualization for the four required plots:
  1. Optimizer replication variance (loss vs. iteration, mean ± std band)
  2. GP surrogate predictive variance over loss values
  3. Parameter posterior given the data
  4. Physics-informed improvement (physics-on vs physics-off)

All functions accept a list of `benchmark.RunRecord` objects so they can be
called either standalone or via `BenchmarkSuite.plot_all`.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(d: str | None) -> Optional[str]:
    if d is None:
        return None
    os.makedirs(d, exist_ok=True)
    return d


def _save(fig, name: str, save_dir: str | None):
    if save_dir is None:
        return
    fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# 1. Optimizer replication variance
# ---------------------------------------------------------------------------

def plot_optimizer_replication_variance(records,
                                        save_dir: str | None = None):
    """Mean ± std of running-min loss across replicates per optimizer."""
    by_opt: Dict[str, List[List[float]]] = defaultdict(list)
    for r in records:
        if not r.physics_informed:
            continue  # use physics-on series for the headline plot
        running = np.minimum.accumulate(np.asarray(r.history, dtype=float))
        by_opt[r.optimizer].append(running.tolist())

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, runs in by_opt.items():
        m = max(len(r) for r in runs)
        padded = np.full((len(runs), m), np.nan)
        for i, run in enumerate(runs):
            padded[i, :len(run)] = run
            padded[i, len(run):] = run[-1]
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        x = np.arange(len(mean))
        ax.plot(x, mean, label=name, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.20)

    ax.set_xlabel("Function evaluation")
    ax.set_ylabel("Best-so-far loss")
    ax.set_title("Optimizer replication variance (mean ± 1σ)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, "01_replication_variance", _ensure_dir(save_dir))
    return fig


# ---------------------------------------------------------------------------
# 2. GP surrogate predictive variance over loss values
# ---------------------------------------------------------------------------

def plot_gp_predictive_variance(records, save_dir: str | None = None):
    """Scatter of GP predictive variance vs observed loss for PIBO runs."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    found = False
    for r in records:
        if r.optimizer != "pibo":
            continue
        gp_var = r.extras.get("gp_predictive_var")
        if not gp_var:
            continue
        # Align variance log to the *acquisition* iterations (post-warm-start).
        h = np.asarray(r.history[-len(gp_var):], dtype=float)
        ax.scatter(h, gp_var, alpha=0.6,
                   label=f"rep {r.replicate} ({'phys' if r.physics_informed else 'plain'})")
        found = True

    if not found:
        ax.text(0.5, 0.5, "no GP variance log (run PIBO first)",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Observed loss at iteration")
    ax.set_ylabel("Mean GP predictive variance over 64 candidates")
    ax.set_yscale("log")
    ax.set_title("GP surrogate predictive variance vs loss")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    _save(fig, "02_gp_predictive_variance", _ensure_dir(save_dir))
    return fig


# ---------------------------------------------------------------------------
# 3. Parameter posterior given the data
# ---------------------------------------------------------------------------

def plot_parameter_posterior(records, specs, save_dir: str | None = None):
    """Marginal histograms of best-x across replicates per optimizer + phys-on.

    Interpreted as samples from the *approximate* posterior of optimal
    parameters: each replicate is one draw from the optimizer's posterior over
    minimizers.
    """
    n = len(specs)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.2 * nrow))
    axes = np.atleast_1d(axes).ravel()

    by_opt: Dict[str, List[np.ndarray]] = defaultdict(list)
    for r in records:
        if not r.physics_informed:
            continue
        if r.best_x is None or len(r.best_x) == 0:
            continue
        by_opt[r.optimizer].append(np.asarray(r.best_x, dtype=float))

    for i, spec in enumerate(specs):
        ax = axes[i]
        for opt_name, xs in by_opt.items():
            arr = np.array(xs)[:, i]
            ax.hist(arr, bins=8, alpha=0.5, label=opt_name)
        ax.axvline(spec.lo, color="gray", ls="--", lw=0.8)
        ax.axvline(spec.hi, color="gray", ls="--", lw=0.8)
        ax.set_title(spec.name, fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Approximate parameter posterior across replicates", y=1.02)
    fig.tight_layout()
    _save(fig, "03_parameter_posterior", _ensure_dir(save_dir))
    return fig


# ---------------------------------------------------------------------------
# 4. Physics-informed improvement plot
# ---------------------------------------------------------------------------

def plot_physics_improvement(records, save_dir: str | None = None):
    """Compare physics-on vs physics-off convergence per optimizer."""
    grouped: Dict[str, Dict[bool, List[List[float]]]] = defaultdict(
        lambda: {True: [], False: []})
    for r in records:
        running = np.minimum.accumulate(np.asarray(r.history, dtype=float))
        grouped[r.optimizer][r.physics_informed].append(running.tolist())

    optimizers = list(grouped.keys())
    fig, axes = plt.subplots(1, len(optimizers),
                             figsize=(3.8 * len(optimizers), 3.6), sharey=True)
    if len(optimizers) == 1:
        axes = [axes]

    for ax, name in zip(axes, optimizers):
        for phys, runs in grouped[name].items():
            if not runs:
                continue
            m = max(len(r) for r in runs)
            padded = np.full((len(runs), m), np.nan)
            for i, run in enumerate(runs):
                padded[i, :len(run)] = run
                padded[i, len(run):] = run[-1]
            mean = np.nanmean(padded, axis=0)
            std = np.nanstd(padded, axis=0)
            x = np.arange(len(mean))
            label = "physics-informed" if phys else "vanilla"
            ax.plot(x, mean, label=label, lw=2)
            ax.fill_between(x, mean - std, mean + std, alpha=0.18)
        ax.set_title(name)
        ax.set_xlabel("Function eval")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Best-so-far loss")
    fig.suptitle("Physics-informed improvement: convergence comparison", y=1.02)
    fig.tight_layout()
    _save(fig, "04_physics_improvement", _ensure_dir(save_dir))
    return fig


# ---------------------------------------------------------------------------
# Aggregate convenience function
# ---------------------------------------------------------------------------

def render_all(records, specs, save_dir: str | None = "figures"):
    return {
        "replication_variance": plot_optimizer_replication_variance(
            records, save_dir=save_dir),
        "gp_predictive_variance": plot_gp_predictive_variance(
            records, save_dir=save_dir),
        "parameter_posterior": plot_parameter_posterior(
            records, specs, save_dir=save_dir),
        "physics_improvement": plot_physics_improvement(
            records, save_dir=save_dir),
    }
