"""Re-train the BO surrogate in absolute-energy units and regenerate the
joint posterior distribution that F6 visualises.

Why this script exists
----------------------
The original recalib_staged_bo GP was trained on ``worst-of-5`` percent
error (a dimensionless BO target that aggregates biaxial / uniaxial /
V_S / diffusion observables). When F8 was reframed in absolute-energy
units, the F6 joint plot (gp_mean × gp_std in %) became dimensionally
inconsistent with the rest of the new figure set.

This script retrains a fresh GP whose **target is the mean per-scan
PES-RMSE (eV)** averaged over the four DFT scan types
(bond / angle / torsion / nonbond), using each of the 100 trials in
RECALIB_LOG_clean.csv as a training point. It then re-evaluates the
GP at the 200 LHS candidate points captured in every
``posterior_snapshots/posterior_trial*.csv`` and writes a new joint
distribution in eV units.

Outputs (in ``manuscript_figs/cache/``):
    - posterior_eV_trial{NNNN}.csv          new per-snapshot CSVs in eV
    - posterior_eV_concat.csv               concatenated (snapshot, mean, std)
    - gp_calibration_diag.json              cross-validation MAE for the new GP
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, ConstantKernel, WhiteKernel,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BO_DIR    = ROOT / "results" / "reviewer_response" / "recalib_staged_bo"
LOG_CSV   = BO_DIR / "RECALIB_LOG_clean.csv"
POST_DIR  = BO_DIR / "posterior_snapshots"
FIG_DIR   = BO_DIR / "manuscript_figs"
CACHE     = FIG_DIR / "cache"
TRIAL_RMSE_CSV = CACHE / "trial_scan_rmse.csv"
OUT_DIR   = CACHE  # write new posterior_eV_*.csv here

# Parameter names matching the BO spec order (must align with RECALIB_LOG cols)
PARAM_GROUPS_FLAT = [
    "De_sigma_MoS", "De_pi_MoS", "p_be1_MoS", "p_be2_MoS",
    "p_bo1_MoS", "p_bo2_MoS", "p_bo3_MoS", "p_bo4_MoS",
    "p_bo5_MoS", "p_bo6_MoS", "p_ovun1_MoS",
    "Dij_MoS", "RvdW_MoS", "Alfa_MoS", "ro_sigma_MoS",
    "Thetao_SMoS", "Thetao_MoSMo", "p_val1_SMoS",
    "p_val1_MoSMo", "p_val2_SMoS", "p_val7_SMoS", "p_val4_SMoS",
    "Mo_RvdW", "Mo_Dij", "Mo_gamma", "Mo_Alfa", "Mo_gamma_w",
    "S_RvdW", "S_Dij", "S_gamma", "S_Alfa", "S_gamma_w",
    "Mo_p_boc4", "Mo_p_boc3", "Mo_p_boc5",
    "Mo_p_ovun2", "Mo_p_val3", "Mo_p_val5",
    "S_p_ovun2", "S_p_val3",
]


def build_training_set():
    """Combine RECALIB_LOG (40-param vectors) with trial_scan_rmse (eV target).

    Returns ``(X, y, used_param_names)``:
        X: shape (n_trials, n_params) parameter values
        y: shape (n_trials,) mean per-scan PES-RMSE in eV
    """
    df_log  = pd.read_csv(LOG_CSV)
    df_rmse = pd.read_csv(TRIAL_RMSE_CSV)

    merged = df_log.merge(df_rmse[["trial", "bond_rmse_eV", "angle_rmse_eV",
                                     "torsion_rmse_eV", "nonbond_rmse_eV"]],
                            on="trial", how="inner")
    rmse_cols = ["bond_rmse_eV", "angle_rmse_eV",
                  "torsion_rmse_eV", "nonbond_rmse_eV"]
    merged["mean_rmse_eV"] = merged[rmse_cols].mean(axis=1)
    merged = merged.dropna(subset=["mean_rmse_eV"])

    used = [n for n in PARAM_GROUPS_FLAT if n in merged.columns]
    X = merged[used].values.astype(float)
    y = merged["mean_rmse_eV"].values.astype(float)
    print(f"[train] {len(X)} trials, {len(used)} params, "
          f"y mean={y.mean():.3f} eV, std={y.std():.3f} eV, "
          f"range=[{y.min():.3f}, {y.max():.3f}]")
    return X, y, used


def fit_gp(X, y):
    """Fit a standard ARD-Matern GP with white-noise nugget (per-param scaler).

    Mirrors the broad family of kernels the original staged BO used so the
    new GP is comparable in inductive bias.
    """
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.ones(Xs.shape[1]),
                        length_scale_bounds=(1e-2, 1e3), nu=2.5)
              + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-4, 10.0)))
    gp = GaussianProcessRegressor(kernel=kernel,
                                   normalize_y=True,
                                   n_restarts_optimizer=4,
                                   random_state=42)
    gp.fit(Xs, y)
    print(f"[gp]    optimized kernel: {gp.kernel_}")
    return gp, scaler


def cv_diagnostic(gp, scaler, X, y, k=5):
    """Quick k-fold MAE/RMSE for the new GP."""
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, k)
    maes, rmses = [], []
    for j in range(k):
        test_idx = folds[j]
        train_idx = np.concatenate([folds[i] for i in range(k) if i != j])
        gp_j = GaussianProcessRegressor(kernel=gp.kernel_,
                                          normalize_y=True, random_state=42)
        gp_j.fit(scaler.transform(X[train_idx]), y[train_idx])
        pred = gp_j.predict(scaler.transform(X[test_idx]))
        err = pred - y[test_idx]
        maes.append(float(np.mean(np.abs(err))))
        rmses.append(float(np.sqrt(np.mean(err ** 2))))
    diag = {
        "k_fold":      k,
        "MAE_mean":    float(np.mean(maes)),
        "MAE_std":     float(np.std(maes)),
        "RMSE_mean":   float(np.mean(rmses)),
        "RMSE_std":    float(np.std(rmses)),
        "MAE_per_fold":  maes,
        "RMSE_per_fold": rmses,
    }
    print(f"[cv]    {k}-fold MAE={diag['MAE_mean']:.3f}±{diag['MAE_std']:.3f} eV, "
          f"RMSE={diag['RMSE_mean']:.3f}±{diag['RMSE_std']:.3f} eV")
    return diag


_SNAP_RE = re.compile(r"posterior_trial(\d+)\.csv$")


def regenerate_snapshots(gp, scaler, used_params):
    """For every existing posterior_trial*.csv snapshot, predict (mean, std) in eV."""
    out_paths = []
    long_rows = []
    for sp in sorted(POST_DIR.glob("posterior_trial*.csv")):
        m = _SNAP_RE.search(sp.name)
        if not m:
            continue
        snap_trial = int(m.group(1))
        df = pd.read_csv(sp)
        # The snapshot CSV stores 40 param columns + candidate_idx + old gp_mean/gp_std
        miss = [n for n in used_params if n not in df.columns]
        if miss:
            print(f"  [skip] {sp.name}: missing params {miss}")
            continue
        Xc = df[used_params].values.astype(float)
        Xc_s = scaler.transform(Xc)
        mean_eV, std_eV = gp.predict(Xc_s, return_std=True)
        out_df = pd.DataFrame({
            "candidate_idx": df["candidate_idx"].values,
            "gp_mean_eV":    mean_eV,
            "gp_std_eV":     std_eV,
        })
        # also keep the old % values for cross-reference
        if "gp_mean" in df.columns:
            out_df["gp_mean_pct_old"] = df["gp_mean"].values
        if "gp_std" in df.columns:
            out_df["gp_std_pct_old"] = df["gp_std"].values
        out_path = OUT_DIR / f"posterior_eV_trial{snap_trial:04d}.csv"
        out_df.to_csv(out_path, index=False, float_format="%.6f")
        out_paths.append(out_path)
        out_df["snapshot_trial"] = snap_trial
        long_rows.append(out_df)
        print(f"  [csv] {out_path.name}  n={len(out_df)}, "
              f"mean ∈ [{mean_eV.min():.2f}, {mean_eV.max():.2f}] eV, "
              f"std ∈ [{std_eV.min():.2f}, {std_eV.max():.2f}] eV")
    if long_rows:
        full = pd.concat(long_rows, axis=0, ignore_index=True)
        concat_path = OUT_DIR / "posterior_eV_concat.csv"
        full.to_csv(concat_path, index=False, float_format="%.6f")
        print(f"[csv]  {concat_path.name}  ({len(full)} rows across {len(long_rows)} snapshots)")
    return out_paths


def main():
    if not TRIAL_RMSE_CSV.exists():
        raise FileNotFoundError(f"need {TRIAL_RMSE_CSV} — run Phase 2 first")

    X, y, used = build_training_set()
    gp, scaler = fit_gp(X, y)
    diag = cv_diagnostic(gp, scaler, X, y)
    (OUT_DIR / "gp_calibration_diag.json").write_text(
        json.dumps(diag, indent=2), encoding="utf-8")
    print(f"[json] gp_calibration_diag.json")
    regenerate_snapshots(gp, scaler, used)
    print("[done] new posterior_eV_*.csv ready for F6 regeneration")


if __name__ == "__main__":
    main()
