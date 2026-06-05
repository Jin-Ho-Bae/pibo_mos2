"""
Posterior-parameter reporting for the BO benchmark.

Bayesian-OPT-style posterior summary
------------------------------------
Each optimizer in this project stashes its full evaluation history in
``RunRecord.extras['all_X']`` (shape ``(n_evals, d)``) and
``RunRecord.extras['all_y']`` (loss per eval). We compute a posterior
moment for each parameter via Boltzmann-importance weights:

    w_j  = exp( -β · (loss_j - loss_min) )
    μ_i  = Σ_j w_j · X_{j,i}  /  Σ_j w_j
    σ_i  = sqrt( Σ_j w_j · (X_{j,i} - μ_i)^2  /  Σ_j w_j )

This is the canonical loss-weighted posterior approximation used in
e.g. SMAC / ParamILS reports. β is set so the effective sample size
``ESS = (Σ w_j)^2 / Σ w_j^2`` lands in the 10–30 % of n_evals range,
which keeps the posterior from collapsing to the best point (high β)
or smearing to the prior (low β). The chosen β is reported with the
CSV.

Public API
----------
``compute_posterior(record, specs)``
    Returns a list of dicts, one per parameter, with prior bounds,
    best value, posterior mean/std, manuscript Table-2 anchor (when
    available), and posterior z-score relative to the anchor.

``save_posterior_csv(rows, path)``
    Writes the per-parameter table.

``plot_posterior(rows, path, title)``
    Renders a one-axis figure: x = parameter index, y = normalised
    range with prior bounds, posterior mean ± std, best value, and
    manuscript anchor (when present). matplotlib is loaded lazily so
    the report module doesn't pull it in just by being imported.

``write_per_optimizer_reports(suite, results_dir, only_physics=True)``
    Convenience driver: for every (optimizer, replicate) in
    ``suite.results``, write
    ``posterior_parameters_<opt>_rep<k>.csv`` and a matching figure.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Core posterior computation
# ---------------------------------------------------------------------------

def _choose_beta(losses: np.ndarray,
                 target_ess_frac: float = 0.20,
                 max_iter: int = 64) -> float:
    """Bisect on β so that ESS / n is close to ``target_ess_frac``.

    ESS = (Σw)² / Σw², which sweeps from n (β=0) down to 1 (β→∞).
    """
    finite = np.isfinite(losses)
    if not np.any(finite):
        return 1.0
    l = losses[finite]
    l_min = float(np.min(l))
    centered = l - l_min
    # Anchor the range from a tight (high β) and loose (low β) estimate.
    lo, hi = 1e-3, 1e3
    n = len(l)
    target = target_ess_frac * n
    for _ in range(max_iter):
        beta = 0.5 * (lo + hi)
        w = np.exp(-beta * centered)
        s = float(np.sum(w))
        if s <= 0 or not np.isfinite(s):
            hi = beta
            continue
        ess = s * s / float(np.sum(w * w))
        if ess > target:
            lo = beta
        else:
            hi = beta
    return float(0.5 * (lo + hi))


def compute_posterior(record, specs) -> List[Dict[str, Any]]:
    """Per-parameter posterior summary from a single ``RunRecord``.

    Parameters
    ----------
    record : BenchmarkSuite.RunRecord
        Must carry ``best_x`` and ``extras['all_X']`` / ``extras['all_y']``.
        If ``extras`` is missing those keys, we fall back to a "best point
        only" report (posterior_mean = best_x, posterior_std = nan).
    specs : list of ParameterSpec
        Same order as ``best_x``. Provides prior bounds and manuscript
        anchors.
    """
    names = [s.name for s in specs]
    lo = np.array([s.lo for s in specs], dtype=float)
    hi = np.array([s.hi for s in specs], dtype=float)
    best_x = np.asarray(record.best_x, dtype=float)

    extras = record.extras or {}
    X = extras.get("all_X")
    y = extras.get("all_y")
    if X is None or y is None or len(X) == 0:
        # No history — degenerate posterior at the best point.
        post_mean = best_x.copy()
        post_std = np.full_like(best_x, fill_value=np.nan)
        beta_used = float("nan")
        ess = float("nan")
    else:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # Filter out non-finite losses; they'd pin β to extremes.
        m = np.isfinite(y)
        if not np.any(m):
            post_mean = best_x.copy()
            post_std = np.full_like(best_x, np.nan)
            beta_used = float("nan")
            ess = float("nan")
        else:
            X_f, y_f = X[m], y[m]
            beta_used = _choose_beta(y_f)
            centered = y_f - float(np.min(y_f))
            w = np.exp(-beta_used * centered)
            ws = float(np.sum(w))
            ess = (ws * ws) / float(np.sum(w * w)) if ws > 0 else float("nan")
            if ws > 0:
                post_mean = (w[:, None] * X_f).sum(axis=0) / ws
                var = (w[:, None] * (X_f - post_mean[None, :]) ** 2).sum(axis=0) / ws
                post_std = np.sqrt(var)
            else:
                post_mean = best_x.copy()
                post_std = np.full_like(best_x, np.nan)

    rows: List[Dict[str, Any]] = []
    for i, s in enumerate(specs):
        manuscript_mean = (float(s.manuscript_mean)
                           if s.manuscript_mean is not None else None)
        manuscript_std = (float(s.manuscript_std)
                          if s.manuscript_std is not None else None)
        # z = (posterior_mean - manuscript_mean) / manuscript_std, if both.
        z = None
        if (manuscript_mean is not None
                and manuscript_std is not None
                and manuscript_std > 0
                and np.isfinite(post_mean[i])):
            z = float((post_mean[i] - manuscript_mean) / manuscript_std)
        rows.append({
            "name":              s.name,
            "block":             s.block,
            "prior_lo":          float(lo[i]),
            "prior_hi":          float(hi[i]),
            "best_value":        float(best_x[i]),
            "posterior_mean":    float(post_mean[i]),
            "posterior_std":     float(post_std[i]),
            "manuscript_mean":   manuscript_mean,
            "manuscript_std":    manuscript_std,
            "z_vs_manuscript":   z,
            "beta_used":         float(beta_used),
            "effective_sample":  float(ess),
        })
    return rows


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_posterior_csv(rows: List[Dict[str, Any]], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = ["name", "block", "prior_lo", "prior_hi", "best_value",
              "posterior_mean", "posterior_std",
              "manuscript_mean", "manuscript_std", "z_vs_manuscript",
              "beta_used", "effective_sample"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def plot_posterior(rows: List[Dict[str, Any]], path: str,
                   title: str = "Posterior parameters") -> Optional[str]:
    """One axis: every parameter in [0,1] normalized prior coordinates,
    with the posterior 1-σ interval as a vertical bar, best value as a
    dot, and the manuscript anchor (if present) as a triangle.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if not rows:
        return None

    def _norm(v, lo, hi):
        if not np.isfinite(v) or hi <= lo:
            return float("nan")
        return float((v - lo) / (hi - lo))

    n = len(rows)
    xs = np.arange(n)
    best_n   = np.array([_norm(r["best_value"], r["prior_lo"], r["prior_hi"])
                         for r in rows])
    mean_n   = np.array([_norm(r["posterior_mean"], r["prior_lo"], r["prior_hi"])
                         for r in rows])
    # Convert posterior std to normalized prior units for the error bar.
    err_n = np.array([
        (r["posterior_std"] / (r["prior_hi"] - r["prior_lo"]))
        if (r["prior_hi"] > r["prior_lo"]
            and np.isfinite(r["posterior_std"])) else 0.0
        for r in rows])
    anchor_n = np.array([
        _norm(r["manuscript_mean"], r["prior_lo"], r["prior_hi"])
        if r["manuscript_mean"] is not None else float("nan")
        for r in rows])

    fig, ax = plt.subplots(figsize=(max(8, 0.22 * n), 4.5))
    ax.fill_between(xs, 0.0, 1.0, color="#e8e8e8", step=None,
                    label="prior bound")
    ax.errorbar(xs, mean_n, yerr=err_n, fmt="o", ms=4, lw=0.8,
                color="#1f77b4", ecolor="#1f77b4", capsize=2,
                label="posterior mean ± σ")
    ax.scatter(xs, best_n, marker="x", s=30, color="#d62728",
               label="best value")
    if np.any(np.isfinite(anchor_n)):
        ax.scatter(xs, anchor_n, marker="^", s=28, color="#2ca02c",
                   label="manuscript Table 2 anchor")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["name"] for r in rows], rotation=90, fontsize=7)
    ax.set_ylabel("normalised prior position")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", color="#cccccc", lw=0.4)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Per-optimizer driver
# ---------------------------------------------------------------------------

def write_per_optimizer_reports(suite,
                                results_dir: str,
                                only_physics: bool = True
                                ) -> Dict[str, List[str]]:
    """Drive the full posterior-export pass for a BenchmarkSuite.

    Returns ``{"csv": [...], "png": [...]}`` with the paths written.
    """
    written: Dict[str, List[str]] = {"csv": [], "png": []}
    specs = getattr(suite, "specs", None) or []
    if not specs:
        return written
    for rec in getattr(suite, "results", []):
        if only_physics and not rec.physics_informed:
            continue
        try:
            rows = compute_posterior(rec, specs)
        except Exception as exc:  # noqa: BLE001
            print(f"[posterior] {rec.optimizer} rep={rec.replicate}: "
                  f"compute failed: {exc}")
            continue
        tag = f"{rec.optimizer}_rep{rec.replicate}"
        if rec.physics_informed is False:
            tag += "_physoff"
        csv_path = os.path.join(results_dir,
                                f"posterior_parameters_{tag}.csv")
        png_path = os.path.join(results_dir, "figures",
                                f"posterior_parameters_{tag}.png")
        save_posterior_csv(rows, csv_path)
        written["csv"].append(csv_path)
        out = plot_posterior(rows, png_path,
                             title=f"Posterior parameters — {tag}  "
                                   f"(loss={rec.best_loss:.4f})")
        if out:
            written["png"].append(out)
    return written


def write_optimizer_benchmark_figure(suite, path: str) -> Optional[str]:
    """Side-by-side benchmark figure: per-optimizer best_loss + val_rmse.

    Bar chart with error bars (mean ± std across replicates), grouped by
    physics_informed (True / False). Returns the saved path or None if
    matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception:
        return None
    df = suite.report() if hasattr(suite, "report") else None
    if df is None or df.empty:
        return None

    summary = (df.groupby(["optimizer", "physics_informed"])
                 .agg(loss_mean=("Loss(RMSE)", "mean"),
                      loss_std=("Loss(RMSE)", "std"),
                      val_mean=("val_RMSE", "mean"),
                      val_std=("val_RMSE", "std"),
                      n=("Loss(RMSE)", "count"))
                 .reset_index())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    width = 0.35
    optimizers = list(summary["optimizer"].unique())
    xs = np.arange(len(optimizers))
    for k, phys in enumerate([True, False]):
        sub = summary[summary["physics_informed"] == phys]
        if sub.empty:
            continue
        idx = [optimizers.index(o) for o in sub["optimizer"]]
        offset = (k - 0.5) * width
        ax1.bar(np.array(idx) + offset,
                sub["loss_mean"].fillna(0.0),
                yerr=sub["loss_std"].fillna(0.0),
                width=width, capsize=3,
                label=f"phys={'on' if phys else 'off'}")
        ax2.bar(np.array(idx) + offset,
                sub["val_mean"].fillna(0.0),
                yerr=sub["val_std"].fillna(0.0),
                width=width, capsize=3,
                label=f"phys={'on' if phys else 'off'}")
    ax1.set_xticks(xs); ax1.set_xticklabels(optimizers)
    ax2.set_xticks(xs); ax2.set_xticklabels(optimizers)
    ax1.set_ylabel("best loss (weighted RMSE, eV)")
    ax2.set_ylabel("validation RMSE (eV)")
    ax1.set_title("Per-optimizer best-loss")
    ax2.set_title("Per-optimizer held-out validation RMSE")
    for ax in (ax1, ax2):
        ax.legend(fontsize=8); ax.grid(axis="y", color="#cccccc", lw=0.4)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
