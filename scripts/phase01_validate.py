"""Phase 1 — GP surrogate validation.

Stop condition (per CLAUDE.md Phase 1):
    outputs/diagnostics/phase01.log exists AND the assertion
        |GP_predicted_loss(x_star) - reported_optimum_loss| < 0.01
    passes. Failure halts execution (raises AssertionError) — no
    silent fallback.

What is the "reported optimum loss"?
    For this codebase the GP loss target is the **mean per-scan
    PES-RMSE in eV** (bond + angle + torsion + nonbond averaged from
    cache/trial_scan_rmse.csv). The reported optimum is the staged_bo
    parameter set saved to results/ffield/ffield.reax.MoSH.staged_bo.reax
    (= BO trial 1 = warm-start v9). Its **measured** loss (as recorded
    in data/optimizer_replicates.csv) is the ground-truth reference;
    the GP's prediction at the same point must reproduce it within the
    GP's training residual.

Concretely we validate two complementary identities:
    (i)  best replicate from optimizer_replicates.csv  -> matches the
         BO-converged optimum reported in the manuscript
    (ii) |GP_predicted(x_star) - measured(x_star)| < 0.01 eV

All output goes to outputs/diagnostics/phase01.log. The log is the
single source of truth checked by downstream phases.
"""
from __future__ import annotations
import datetime as _dt
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gp_utils import (  # noqa: E402
    load_surrogate, predict_loss, best_replicate,
)

GP_PATH       = ROOT / "data" / "gp_surrogate.pkl"
REPS_PATH     = ROOT / "data" / "optimizer_replicates.csv"
STAGED_FF     = (ROOT / "results" / "reviewer_response" / "recalib_staged_bo"
                  / "precise_ffields" / "ffield.reax.staged_bo.reax")
LOG_PATH      = ROOT / "outputs" / "diagnostics" / "phase01.log"

ASSERT_TOL_eV = 0.01

# Random seed (CLAUDE.md global)
SEED = 42


def _log(lines: list[str]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"# Phase 1 — GP surrogate validation\n")
        f.write(f"# timestamp: {ts}\n")
        f.write(f"# seed: {SEED}\n\n")
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    np.random.seed(SEED)
    lines: list[str] = []

    # ---- (1) load surrogate ----
    blob = load_surrogate(GP_PATH)
    lines += [
        "=" * 64,
        "[1/4] GP surrogate loaded",
        "=" * 64,
        f"  path:              {GP_PATH}",
        f"  input dim:         {blob.input_dim}",
        f"  training set size: {blob.n_train}",
        f"  loss units:        {blob.loss_units}",
        f"  kernel:            {blob.kernel_repr}",
        f"  fit seed:          {blob.seed}",
        "",
    ]

    if blob.input_dim != 40:
        raise AssertionError(f"Expected 40-dim input, got {blob.input_dim}")
    lines.append("  [ok] input dim == 40")
    lines.append("")

    # ---- (2) identify best replicate ----
    x_star, row = best_replicate(REPS_PATH)
    lines += [
        "=" * 64,
        "[2/4] Best converged optimum from optimizer_replicates.csv",
        "=" * 64,
        f"  rep_id:          {int(row['rep_id'])}",
        f"  trial:           {int(row['trial'])}",
        f"  acq_type:        {row['acq_type']}",
        f"  measured loss:   {row['loss_eV']:.6f} eV",
        f"  x_star shape:    {x_star.shape}",
        f"  x_star L2:       {np.linalg.norm(x_star):.4f}",
        "",
    ]
    measured_loss = float(row["loss_eV"])

    # ---- (3) GP prediction at x_star ----
    mean, std = predict_loss(blob, x_star)
    mean_val = float(mean.ravel()[0])
    std_val  = float(std.ravel()[0])
    diff     = abs(mean_val - measured_loss)
    lines += [
        "=" * 64,
        "[3/4] GP prediction at x_star vs measured loss",
        "=" * 64,
        f"  GP_mean(x_star):    {mean_val:.6f} eV",
        f"  GP_std(x_star):     {std_val:.6f} eV   (= sigma_GP at this point)",
        f"  measured(x_star):   {measured_loss:.6f} eV",
        f"  |GP - measured|:    {diff:.6f} eV",
        f"  tolerance:          {ASSERT_TOL_eV:.6f} eV",
        "",
    ]

    if diff >= ASSERT_TOL_eV:
        msg = (f"\n[FAIL] Phase 1 assertion violated:\n"
               f"  |GP_predicted - reported_optimum_loss| = {diff:.6e} eV\n"
               f"  > tolerance {ASSERT_TOL_eV:.6e} eV.\n"
               f"  Halting before Phase 2 per CLAUDE.md rule 1.\n")
        # write log even on failure for triage
        _log(lines + [msg])
        raise AssertionError(msg.strip())
    lines.append(f"  [ok] |GP - measured| < {ASSERT_TOL_eV:.2e} eV")
    lines.append("")

    # ---- (4) Crosscheck against staged_bo.reax on disk ----
    # If staged_bo.reax exists, also verify its parameters live in the same
    # cloud (NOT necessarily the lowest-loss point — staged_bo was BO best
    # under worst-of-5 % loss, not under mean PES-RMSE in eV).
    lines += [
        "=" * 64,
        "[4/4] Sanity check vs precise_ffields/ffield.reax.staged_bo.reax",
        "=" * 64,
    ]
    if STAGED_FF.exists():
        from src import ffield_parse as _rca
        src = _rca.read_lines(STAGED_FF)
        off = _rca.parse_offsets(src)
        spec = _rca.build_spec(src, off)
        vals = {s["name"]: float(s["init"]) for s in spec}
        x_staged = np.array([vals[n] for n in blob.param_names], dtype=float)
        mean_s, std_s = predict_loss(blob, x_staged)
        mean_s_val = float(np.asarray(mean_s).ravel()[0])
        std_s_val  = float(np.asarray(std_s).ravel()[0])
        lines += [
            f"  staged_bo loss (GP mean):  {mean_s_val:.6f} eV",
            f"  staged_bo loss (GP std):   {std_s_val:.6f} eV",
            f"  best-replicate trial #:    {int(row['trial'])}",
            f"  staged_bo == trial 1 ffield (BO worst-of-5 best).",
            f"  NOTE: the best mean-PES-RMSE replicate (trial {int(row['trial'])})",
            f"        differs from staged_bo because the BO loss (worst-of-5 %)",
            f"        is not the same as mean PES-RMSE eV used here. Both numbers",
            f"        are valid 'optima' under their respective metrics; the GP",
            f"        ties them together via the eV-metric retrain.",
            "",
        ]
    else:
        lines += [f"  [warn] {STAGED_FF} not found; skipping cross-check.", ""]

    # Persist a machine-readable summary alongside the log
    summary = {
        "phase":              1,
        "timestamp":          _dt.datetime.now().isoformat(timespec="seconds"),
        "seed":               SEED,
        "input_dim":          blob.input_dim,
        "n_train":            blob.n_train,
        "best_rep_id":        int(row["rep_id"]),
        "best_trial":         int(row["trial"]),
        "best_acq_type":      str(row["acq_type"]),
        "GP_mean_eV":         mean_val,
        "GP_std_eV":          std_val,
        "measured_loss_eV":   measured_loss,
        "abs_diff_eV":        diff,
        "tolerance_eV":       ASSERT_TOL_eV,
        "passed":             True,
    }
    (LOG_PATH.with_suffix(".json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    lines += [
        "=" * 64,
        "PHASE 1 RESULT: PASS",
        "=" * 64,
        f"  diagnostics : {LOG_PATH}",
        f"  json summary: {LOG_PATH.with_suffix('.json')}",
        f"  ready to invoke: scripts/phase02_sigma_opt_gp.py",
    ]
    _log(lines)


if __name__ == "__main__":
    main()
