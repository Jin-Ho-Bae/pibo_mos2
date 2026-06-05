"""
Parametric study for the staged-acquisition schedule (UCB → EI → PI).

The manuscript fixes the stage boundaries at 30% / 70% of the BO budget
(`stage_ucb_until=0.30`, `stage_ei_until=0.70`). This module lets you
sweep those (and optionally the ancillary acquisition knobs) over a
user-supplied grid, runs PIBO for ``replicates`` independent seeds per
grid point, and reports loss + DFT-vs-ReaxFF energy comparison metrics
so you can pick the schedule empirically.

Typical use
-----------
    from pibo_reaxff.benchmark import BenchmarkSuite
    from pibo_reaxff.parametric_study import (
        AcquisitionGridPoint, run_acquisition_grid,
        summarize_grid, plot_loss_heatmap, plot_parity)

    suite = BenchmarkSuite(profile="demo", lammps_mode="auto")
    grid = [
        AcquisitionGridPoint(0.20, 0.60),
        AcquisitionGridPoint(0.30, 0.70),     # manuscript default
        AcquisitionGridPoint(0.40, 0.80),
    ]
    records = run_acquisition_grid(suite, grid, replicates=3, budget=120)
    df, agg = summarize_grid(records)
    plot_loss_heatmap(agg, save_path="results/parametric_acq/heatmap.png")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .loss import (ReaxFFLoss, _shift_to_match_mean,
                   energy_mae as _energy_mae,
                   energy_rmse as _energy_rmse)
from .optimizers.pibo import PIBOOptimizer
from .physics_constraints import PhysicsPenalty


# ---------------------------------------------------------------------------
# Grid point spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AcquisitionGridPoint:
    """One (UCB%, EI%, ...) combination to sweep.

    The two stage boundaries (``ucb_until``, ``ei_until``) are the headline
    knobs the user will edit by hand. The ancillary knobs default to the
    manuscript values so a one-axis sweep works out of the box.
    """
    ucb_until: float                  # 0–30% (manuscript: 0.30)
    ei_until:  float                  # 30–70% (manuscript: 0.70)
    ucb_beta_init:  float = 2.0
    ucb_beta_decay: float = 0.995
    ei_xi:          float = 0.01
    thompson_prob:  float = 0.15

    def __post_init__(self):
        if not (0.0 <= self.ucb_until <= self.ei_until <= 1.0):
            raise ValueError(
                f"Invalid grid point: need 0 ≤ ucb_until ≤ ei_until ≤ 1; "
                f"got ucb_until={self.ucb_until}, ei_until={self.ei_until}.")

    @property
    def label(self) -> str:
        return f"ucb={self.ucb_until:.2f}_ei={self.ei_until:.2f}"

    def to_acq_config(self) -> Dict[str, float]:
        return {
            "stage_ucb_until": self.ucb_until,
            "stage_ei_until":  self.ei_until,
            "ucb_beta_init":   self.ucb_beta_init,
            "ucb_beta_decay":  self.ucb_beta_decay,
            "ei_xi":           self.ei_xi,
            "thompson_prob":   self.thompson_prob,
        }


# ---------------------------------------------------------------------------
# Per-run record
# ---------------------------------------------------------------------------

@dataclass
class GridRunRecord:
    """One (grid_point × replicate) result with loss + energy comparison."""
    ucb_until: float
    ei_until:  float
    replicate: int
    seed:      int

    best_loss:      float
    best_loss_history: List[float]            # per-iteration best-so-far
    val_rmse:       float

    # DFT-vs-ReaxFF energy comparison at best_x (mean-shifted, eV)
    energy_mae:     float
    energy_rmse:    float
    energy_max_err: float
    predictions_eV: List[float]
    references_eV:  List[float]
    frame_categories: List[str]

    n_evals:        int
    wall_clock_s:   float
    stage_counts:   Dict[str, int]            # init / ucb / ei / pi / thompson(?)
    best_x:         List[float]


# ---------------------------------------------------------------------------
# Core grid runner
# ---------------------------------------------------------------------------

def _running_min(arr: Sequence[float]) -> List[float]:
    out: List[float] = []
    cur = float("inf")
    for v in arr:
        if v < cur:
            cur = float(v)
        out.append(cur)
    return out


def _stage_counts(stage_log: Sequence[str]) -> Dict[str, int]:
    from collections import Counter
    return dict(Counter(stage_log))


def _make_optimizer(suite, point: AcquisitionGridPoint, penalty
                    ) -> PIBOOptimizer:
    return PIBOOptimizer(
        physics_informed=True,
        penalty=penalty,
        config={
            "gp":          suite.config.get("gp", {}),
            "acquisition": {**suite.config.get("acquisition", {}),
                             **point.to_acq_config()},
            "init_lhs_points":
                suite.opt_cfg.get("init_lhs_points", 15),
            "patience":    suite.opt_cfg.get("patience", 0),
            "burn_in":     suite.opt_cfg.get("burn_in", 0),
            "online_window_recent":
                suite.opt_cfg.get("online_window_recent", 0),
            "online_window_initial":
                suite.opt_cfg.get("online_window_initial", 0),
        })


def run_acquisition_grid(suite,
                         grid: Sequence[AcquisitionGridPoint],
                         replicates: int = 3,
                         budget: int = 100,
                         rng_seed: int = 42,
                         verbose: bool = True) -> List[GridRunRecord]:
    """Run PIBO on each grid point × replicate; return per-run records.

    The suite supplies the runner / specs / training frames / weights;
    this function just swaps in a fresh ``PIBOOptimizer`` per call with the
    grid point's acquisition schedule overlaid on top of the suite's
    baseline ``acquisition`` config.
    """
    penalty_lambda = (suite.config.get("physics_constraints", {})
                      .get("penalty_lambda", 1.0))
    penalty = PhysicsPenalty(specs=suite.specs, lambda_=penalty_lambda)

    records: List[GridRunRecord] = []
    n_total = len(grid) * replicates
    done = 0
    t_grid_start = time.time()

    for point in grid:
        for rep in range(replicates):
            seed = rng_seed + rep * 1000
            rng = np.random.default_rng(seed)
            loss = ReaxFFLoss(suite.runner, suite.specs,
                              suite.train_frames, suite.val_frames,
                              weights=suite.weights)

            opt = _make_optimizer(suite, point, penalty)
            t0 = time.time()
            res = opt.optimize(loss=loss, specs=suite.specs,
                               budget=budget, rng=rng)
            wall = time.time() - t0

            # Re-evaluate at best_x to refresh per-frame predictions
            # (validation_accuracy() perturbs last_ys, so do it after).
            val_rmse = float(loss.validation_accuracy(res.best_x))
            _ = loss(res.best_x)
            ys_pred = (loss.last_ys.copy()
                       if loss.last_ys is not None else np.array([]))
            ys_ref = np.array([f.energy for f in suite.train_frames],
                              dtype=float)

            ys_shifted = (_shift_to_match_mean(ys_pred, ys_ref)
                          if len(ys_pred) else ys_pred)
            if len(ys_shifted):
                resid = ys_shifted - ys_ref
                e_mae = float(np.mean(np.abs(resid)))
                e_rmse = float(np.sqrt(np.mean(resid ** 2)))
                e_max = float(np.max(np.abs(resid)))
            else:
                e_mae = e_rmse = e_max = float("nan")

            records.append(GridRunRecord(
                ucb_until=point.ucb_until,
                ei_until=point.ei_until,
                replicate=rep,
                seed=seed,
                best_loss=float(res.best_loss),
                best_loss_history=_running_min(res.history),
                val_rmse=val_rmse,
                energy_mae=e_mae,
                energy_rmse=e_rmse,
                energy_max_err=e_max,
                predictions_eV=ys_shifted.tolist() if len(ys_shifted) else [],
                references_eV=ys_ref.tolist(),
                frame_categories=[f.category for f in suite.train_frames],
                n_evals=int(res.n_evals),
                wall_clock_s=float(wall),
                stage_counts=_stage_counts(res.extras.get("stage_log", [])),
                best_x=res.best_x.tolist(),
            ))
            done += 1
            if verbose:
                avg = (time.time() - t_grid_start) / done
                eta = avg * (n_total - done)
                print(f"  [{done}/{n_total}] {point.label} rep={rep} "
                      f"loss={res.best_loss:.4f}  e_mae={e_mae:.4f}eV  "
                      f"val={val_rmse:.4f}  wall={wall:.1f}s  "
                      f"ETA~{eta:.0f}s")

    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarize_grid(records: Sequence[GridRunRecord]):
    """Return ``(per_run_df, aggregate_df)``.

    ``per_run_df`` is one row per (grid_point, replicate) with scalar
    metrics; ``aggregate_df`` is one row per grid_point with mean ± std
    across replicates. Both are pandas DataFrames.
    """
    import pandas as pd  # local import keeps module lean for non-pandas callers
    rows = []
    for r in records:
        rows.append({
            "ucb_until":      r.ucb_until,
            "ei_until":       r.ei_until,
            "replicate":      r.replicate,
            "seed":           r.seed,
            "best_loss":      r.best_loss,
            "val_rmse":       r.val_rmse,
            "energy_mae":     r.energy_mae,
            "energy_rmse":    r.energy_rmse,
            "energy_max_err": r.energy_max_err,
            "n_evals":        r.n_evals,
            "wall_clock_s":   r.wall_clock_s,
            "stage_counts":   r.stage_counts,
        })
    per_run = pd.DataFrame(rows)
    agg = (per_run.groupby(["ucb_until", "ei_until"])
                  .agg(best_loss_mean=("best_loss",      "mean"),
                       best_loss_std =("best_loss",      "std"),
                       val_rmse_mean =("val_rmse",       "mean"),
                       energy_mae_mean=("energy_mae",     "mean"),
                       energy_rmse_mean=("energy_rmse",   "mean"),
                       energy_max_err_mean=("energy_max_err", "mean"),
                       wall_clock_s_mean=("wall_clock_s", "mean"),
                       n_replicates  =("best_loss",      "count"))
                  .reset_index()
                  .sort_values("best_loss_mean"))
    return per_run, agg


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_records(records: Sequence[GridRunRecord], path: str) -> None:
    """Serialize records to JSON (ndarrays already converted to lists)."""
    data = [asdict(r) for r in records]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_records(path: str) -> List[GridRunRecord]:
    with open(path) as f:
        data = json.load(f)
    return [GridRunRecord(**d) for d in data]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_loss_heatmap(agg_df, value_col: str = "best_loss_mean",
                      title: Optional[str] = None,
                      save_path: Optional[str] = None):
    """2-D heatmap of ``value_col`` over the (ucb_until, ei_until) grid.

    When the grid is sparse / non-rectangular, pivot returns NaNs in the
    empty cells; the heatmap shows them as masked (white) cells.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    pivot = agg_df.pivot(index="ucb_until", columns="ei_until",
                         values=value_col)
    fig, ax = plt.subplots(figsize=(7, 5))
    masked = np.ma.array(pivot.values, mask=np.isnan(pivot.values))
    im = ax.imshow(masked, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])
    ax.set_xlabel("EI stage end (fraction of BO budget)")
    ax.set_ylabel("UCB stage end (fraction of BO budget)")
    ax.set_title(title or f"PIBO acquisition sweep — {value_col}")

    # Annotate each cell with the value (skip NaN)
    for i, r in enumerate(pivot.index):
        for j, c in enumerate(pivot.columns):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="w" if v > np.nanmean(masked) else "k",
                        fontsize=8)

    plt.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_parity(records_for_combo: Sequence[GridRunRecord],
                title: Optional[str] = None,
                save_path: Optional[str] = None):
    """DFT vs PIBO-ReaxFF parity scatter for one (ucb%, ei%) combination.

    Aggregates per-frame predictions across replicates (mean ± std) and
    color-codes by Manuscript Table 1 category.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not records_for_combo:
        raise ValueError("records_for_combo is empty.")

    preds = np.stack([np.array(r.predictions_eV) for r in records_for_combo])
    refs = np.array(records_for_combo[0].references_eV)
    cats = records_for_combo[0].frame_categories
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0) if preds.shape[0] > 1 else np.zeros_like(pred_mean)

    fig, ax = plt.subplots(figsize=(6, 6))
    color_map = {
        "equilibrium":  "#d62728",
        "bond_pes":     "#1f77b4",
        "angle_pes":    "#2ca02c",
        "dihedral_pes": "#ff7f0e",
        "strained":     "#9467bd",
        "default":      "#7f7f7f",
    }
    unique_cats = sorted(set(cats))
    for cat in unique_cats:
        mask = np.array([c == cat for c in cats])
        if not mask.any():
            continue
        ax.errorbar(refs[mask], pred_mean[mask], yerr=pred_std[mask],
                    fmt="o", label=f"{cat} (n={mask.sum()})",
                    color=color_map.get(cat, "#7f7f7f"),
                    capsize=3, alpha=0.75, markersize=6)

    lo = float(min(refs.min(), pred_mean.min())) - 0.5
    hi = float(max(refs.max(), pred_mean.max())) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("DFT energy (eV)")
    ax.set_ylabel("PIBO-ReaxFF energy (eV, mean-shifted)")

    pt = records_for_combo[0]
    default_title = (f"DFT vs PIBO-ReaxFF parity  "
                     f"(ucb={pt.ucb_until:.2f}, ei={pt.ei_until:.2f}, "
                     f"n_rep={len(records_for_combo)})")
    ax.set_title(title or default_title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_loss_traces(records: Sequence[GridRunRecord],
                     save_path: Optional[str] = None):
    """One curve per grid point: best-so-far loss vs evaluation index.

    Replicates within a grid point are mean ± shaded-std. Useful for
    seeing *when* each schedule starts plateauing.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    by_point: Dict[Tuple[float, float], List[List[float]]] = defaultdict(list)
    for r in records:
        by_point[(r.ucb_until, r.ei_until)].append(r.best_loss_history)

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    for i, ((u, e), histories) in enumerate(sorted(by_point.items())):
        # Pad histories to common length (in case patience triggered early stops)
        L = max(len(h) for h in histories)
        padded = np.array([h + [h[-1]] * (L - len(h)) for h in histories])
        mean = padded.mean(axis=0)
        std = padded.std(axis=0) if padded.shape[0] > 1 else np.zeros_like(mean)
        x = np.arange(L)
        color = cmap(i / max(1, len(by_point) - 1))
        ax.plot(x, mean, color=color, label=f"ucb={u:.2f}, ei={e:.2f}")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Evaluation index")
    ax.set_ylabel("Best-so-far loss")
    ax.set_title("PIBO loss trace by acquisition schedule")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
