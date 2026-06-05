"""One-off bootstrap: persist the GP surrogate and optimizer-replicate
cloud at the paths the Phase 1+ pipeline expects.

Outputs:
    data/gp_surrogate.pkl       (sklearn GP + StandardScaler + param order)
    data/optimizer_replicates.csv
        rows: 100 staged_bo trials
        cols: rep_id, trial, acq_type, <40 params...>, loss_eV
        loss_eV = mean PES-RMSE across (bond, angle, torsion, nonbond)

Seed: fixed at 42 (see CLAUDE.md).
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from retrain_gp_posterior_in_eV import (  # noqa: E402
    build_training_set, fit_gp, PARAM_GROUPS_FLAT,
)

BO_DIR    = ROOT / "results" / "reviewer_response" / "recalib_staged_bo"
LOG_CSV   = BO_DIR / "RECALIB_LOG_clean.csv"
RMSE_CSV  = BO_DIR / "manuscript_figs" / "cache" / "trial_scan_rmse.csv"
DATA_DIR  = ROOT / "data"

SEED = 42


def main():
    np.random.seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1) Train GP and persist ----
    print(f"[seed] {SEED}")
    X, y, used = build_training_set()
    gp, scaler = fit_gp(X, y)
    blob = {
        "gp":              gp,
        "scaler":          scaler,
        "param_names":     used,
        "loss_units":      "eV  (mean of bond/angle/torsion/nonbond PES-RMSE)",
        "n_train":         int(X.shape[0]),
        "input_dim":       int(X.shape[1]),
        "training_X":      X.copy(),
        "training_y":      y.copy(),
        "kernel_repr":     str(gp.kernel_),
        "seed":            SEED,
    }
    out_pkl = DATA_DIR / "gp_surrogate.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[pkl] {out_pkl.name}  (n_train={blob['n_train']}, d={blob['input_dim']})")

    # ---- 2) Build optimizer_replicates.csv ----
    log = pd.read_csv(LOG_CSV)
    rmse = pd.read_csv(RMSE_CSV)
    rmse_cols = ["bond_rmse_eV", "angle_rmse_eV", "torsion_rmse_eV", "nonbond_rmse_eV"]
    rmse["loss_eV"] = rmse[rmse_cols].mean(axis=1)

    merged = log.merge(rmse[["trial", "loss_eV"]], on="trial", how="inner")
    # Order columns: rep_id, trial, acq_type, [40 params], loss_eV
    keep = ["trial", "acq_type"] + used + ["loss_eV"]
    merged = merged[keep].sort_values("trial").reset_index(drop=True)
    merged.insert(0, "rep_id", np.arange(1, len(merged) + 1))

    out_csv = DATA_DIR / "optimizer_replicates.csv"
    merged.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"[csv] {out_csv.name}  (n_replicates={len(merged)})")
    print(f"        loss_eV range: [{merged['loss_eV'].min():.3f}, "
          f"{merged['loss_eV'].max():.3f}]  mean={merged['loss_eV'].mean():.3f}")
    best_idx = merged["loss_eV"].idxmin()
    print(f"        best trial: trial={int(merged.loc[best_idx, 'trial'])}, "
          f"loss={merged.loc[best_idx, 'loss_eV']:.6f} eV")


if __name__ == "__main__":
    main()
