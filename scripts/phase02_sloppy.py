"""Phase 2 — Sethna sloppy-spectrum analysis at the BO-converged optimum.

Workflow:

1. Gate on Phase 1 result (outputs/diagnostics/phase01.json must say
   ``"passed": true``); otherwise raise. Enforces CLAUDE.md rule 1.
2. Identify ``x_star``:
       - The mean-PES-RMSE optimum from optimizer_replicates.csv
         (trial 64 in this run; reported in phase01.json).
       - Cross-checked against staged_bo.reax (= manuscript's reported
         optimum on the worst-of-5 metric). Both points are written
         to the diagnostic log; the sloppy analysis is run at x_star
         (the eV-loss optimum).
3. Run ``hessian_at`` three times — one per jitter strategy in
       {"keep", "subtract", "report_floor"}
   and compute the spectrum each time.
4. Per strategy, export
       outputs/data/phase02_spectrum_<strategy>.csv
   with rank, eigenvalue, sign, |λ|, top-3 contributing parameters.
5. Single comparison figure
       outputs/figures/phase02_spectrum.png
   showing log|λ| vs rank for all three strategies + horizontal jitter
   floor.
6. Aggregate diagnostic
       outputs/diagnostics/phase02.log
   with the sloppiness report for the canonical ``subtract`` strategy
   and the full negative-eigenvalue audit.

Stop condition (per CLAUDE.md Phase 2):
   * phase02_spectrum_subtract.csv exists
   * n_stiff reported in phase02.log
   * negative-eigenvalue audit complete (rank, λ, top-3 params per
     negative eigenvalue)
   * WARNING emitted if any negative eigenvalue at rank ≤ 10
"""
from __future__ import annotations
import json
import datetime as _dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gp_utils import load_surrogate, best_replicate, predict_loss  # noqa: E402
from src.sloppy import hessian_at, sloppy_spectrum, sloppiness_report  # noqa: E402

GP_PATH      = ROOT / "data" / "gp_surrogate.pkl"
REPS_PATH    = ROOT / "data" / "optimizer_replicates.csv"
STAGED_FF    = (ROOT / "results" / "reviewer_response" / "recalib_staged_bo"
                / "precise_ffields" / "ffield.reax.staged_bo.reax")

PHASE01_JSON = ROOT / "outputs" / "diagnostics" / "phase01.json"
LOG_PATH     = ROOT / "outputs" / "diagnostics" / "phase02.log"
FIG_PATH     = ROOT / "outputs" / "figures" / "phase02_spectrum.png"
DATA_DIR     = ROOT / "outputs" / "data"

STRATEGIES = ("keep", "subtract", "report_floor")
EPS_FD     = 1e-3
SEED       = 42


def _gate_on_phase01() -> dict:
    if not PHASE01_JSON.exists():
        raise RuntimeError(
            f"{PHASE01_JSON} missing — run scripts/phase01_validate.py first."
        )
    summary = json.loads(PHASE01_JSON.read_text(encoding="utf-8"))
    if not summary.get("passed", False):
        raise RuntimeError(
            f"Phase 1 marked passed=False in {PHASE01_JSON}; "
            f"refusing to start Phase 2 (CLAUDE.md rule 1)."
        )
    return summary


def _spectrum_to_dataframe(spec: dict, param_names: list[str]) -> pd.DataFrame:
    w   = spec["eigenvalues"]
    V   = spec["eigenvectors"]
    rows = []
    for i in range(w.size):
        v = V[:, i]
        idx3 = np.argsort(np.abs(v))[::-1][:3]
        rows.append({
            "rank":              i + 1,
            "eigenvalue":        float(w[i]),
            "abs_eigenvalue":    float(abs(w[i])),
            "log10_abs":         float(np.log10(max(abs(w[i]), 1e-30))),
            "sign":              int(np.sign(w[i])),
            "top1_param":        param_names[idx3[0]],
            "top1_loading":      float(v[idx3[0]]),
            "top2_param":        param_names[idx3[1]],
            "top2_loading":      float(v[idx3[1]]),
            "top3_param":        param_names[idx3[2]],
            "top3_loading":      float(v[idx3[2]]),
        })
    return pd.DataFrame(rows)


def _plot_three_strategies(spectra: dict[str, dict], info_by_strat: dict[str, dict],
                            out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    colors = {"keep": "#0a2540", "subtract": "#d62728", "report_floor": "#2ca02c"}
    markers = {"keep": "o", "subtract": "s", "report_floor": "^"}
    for strat, spec in spectra.items():
        abs_w = spec["abs_eigenvalues"]
        signs = spec["signs"]
        ranks = np.arange(1, abs_w.size + 1)
        pos = signs > 0
        neg = signs < 0
        c = colors[strat]
        m = markers[strat]
        if pos.any():
            ax.scatter(ranks[pos], abs_w[pos], c=c, marker=m, s=55,
                        alpha=0.85, edgecolors="black", linewidth=0.5,
                        label=f"{strat}  (λ>0; n_eff={spec['effective_rank']})")
        if neg.any():
            ax.scatter(ranks[neg], abs_w[neg], facecolors="none", edgecolors=c,
                        marker=m, s=85, linewidth=1.4,
                        label=f"{strat}  (λ<0; n={int(neg.sum())})")
        # connecting line for clarity
        ax.plot(ranks, abs_w, color=c, linewidth=0.8, alpha=0.5)

    # Jitter floor (same across strategies for this GP)
    floor = next(iter(info_by_strat.values()))["jitter_floor"]
    ax.axhline(floor, color="#888", linestyle="--", linewidth=1.0,
                label=f"jitter floor  = {floor:.2e}")
    ax.axhline(10.0 * floor, color="#bbb", linestyle=":", linewidth=1.0,
                label=f"10·jitter floor")

    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue rank (sorted by |λ|, descending)")
    ax.set_ylabel("|λ| of GP loss-Hessian (scaled coords)")
    ax.set_title("Phase 2 — Sethna sloppy spectrum at x_star\n"
                  "(three jitter-handling strategies)",
                  fontweight="bold")
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left", ncol=2, frameon=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    np.random.seed(SEED)

    # --- Phase 1 gate ---
    phase01 = _gate_on_phase01()

    # --- Load surrogate + x_star ---
    blob = load_surrogate(GP_PATH)
    x_star, x_star_row = best_replicate(REPS_PATH)
    mean0, std0 = predict_loss(blob, x_star)
    mean0 = float(np.asarray(mean0).ravel()[0])
    std0  = float(np.asarray(std0).ravel()[0])

    # Crosscheck against staged_bo
    staged_loaded = False
    if STAGED_FF.exists():
        from src import ffield_parse as _rca
        src = _rca.read_lines(STAGED_FF)
        off = _rca.parse_offsets(src)
        spec_ff = _rca.build_spec(src, off)
        vals = {s["name"]: float(s["init"]) for s in spec_ff}
        x_staged = np.array([vals[n] for n in blob.param_names], dtype=float)
        mean_s, std_s = predict_loss(blob, x_staged)
        mean_s = float(np.asarray(mean_s).ravel()[0])
        std_s  = float(np.asarray(std_s).ravel()[0])
        staged_loaded = True

    # --- Run all three jitter strategies ---
    spectra = {}
    infos = {}
    summaries = {}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for strat in STRATEGIES:
        H, info = hessian_at(blob, x_star, jitter_strategy=strat, eps=EPS_FD)
        spec = sloppy_spectrum(H, info)
        spectra[strat] = spec
        infos[strat] = info
        df = _spectrum_to_dataframe(spec, blob.param_names)
        csv_path = DATA_DIR / f"phase02_spectrum_{strat}.csv"
        df.to_csv(csv_path, index=False, float_format="%.6e")
        summaries[strat] = {
            "csv_path":         str(csv_path),
            "n_stiff":          int(np.sum(spec["abs_eigenvalues"] > 1e-2)),
            "n_sloppy":         int(np.sum(spec["abs_eigenvalues"] < 1e-4)),
            "n_negative":       spec["n_negative"],
            "n_negative_top10": spec["n_negative_top10"],
            "span_orders":      spec["span_orders"],
            "effective_rank":   spec["effective_rank"],
            "jitter_floor":     spec["jitter_floor"],
        }

    # --- Figure (all three together) ---
    _plot_three_strategies(spectra, infos, FIG_PATH)

    # --- Diagnostic log ---
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines += [
        f"# Phase 2 — Sethna sloppy spectrum at x_star",
        f"# timestamp: {ts}",
        f"# seed: {SEED}",
        f"# gated on phase01.json passed = True",
        f"",
        f"## Reference point",
        f"",
        f"x_star (best replicate by mean PES-RMSE eV):",
        f"  rep_id      : {int(x_star_row['rep_id'])}",
        f"  trial       : {int(x_star_row['trial'])}",
        f"  acq_type    : {x_star_row['acq_type']}",
        f"  measured    : {x_star_row['loss_eV']:.6f} eV",
        f"  GP mean     : {mean0:.6f} eV",
        f"  GP std      : {std0:.6f} eV",
        f"",
    ]
    if staged_loaded:
        lines += [
            f"Cross-check at staged_bo.reax (= manuscript's reported optimum):",
            f"  GP mean     : {mean_s:.6f} eV",
            f"  GP std      : {std_s:.6f} eV",
            f"  NOTE: x_star ≠ staged_bo because the BO target was worst-of-5 %",
            f"        while the GP here targets mean PES-RMSE eV. Phase 2 analyses",
            f"        the spectrum at the eV-loss optimum (x_star).",
            f"",
        ]

    # Strategy-by-strategy headline
    lines += [f"## Strategy comparison", ""]
    lines += [
        f"| strategy | n_stiff | n_sloppy | n_negative | n_neg≤10 | "
        f"span (dec) | eff. rank | jitter floor |",
        f"|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strat in STRATEGIES:
        s = summaries[strat]
        lines.append(
            f"| {strat} | {s['n_stiff']} | {s['n_sloppy']} | "
            f"{s['n_negative']} | {s['n_negative_top10']} | "
            f"{s['span_orders']:.2f} | {s['effective_rank']} | "
            f"{s['jitter_floor']:.3e} |"
        )
    lines.append("")

    # Negative-eigenvalue audit + WARNING gate (canonical = subtract)
    canon = "subtract"
    spec_canon = spectra[canon]
    n_neg10 = spec_canon["n_negative_top10"]
    if n_neg10 > 0:
        lines += [
            f"## ⚠️ WARNING — {n_neg10} negative eigenvalue(s) at rank ≤ 10",
            f"",
            f"The staged-BO optimum (mapped to the mean-PES-RMSE GP surface)",
            f"is a shallow saddle along {n_neg10} direction(s) within the top 10",
            f"|λ| eigenvectors. Downstream HMC, profile likelihood, and",
            f"sigma_post analyses **may require local refinement** (e.g., L-BFGS-B",
            f"polishing of x_star) before being run. Do **not** treat the BO",
            f"optimum as a strict local minimum on this loss surface.",
            f"",
        ]

    # Full sloppiness_report for the canonical strategy
    lines += [
        f"## Canonical sloppiness report (strategy = '{canon}')",
        f"",
        sloppiness_report(spec_canon, param_names=list(blob.param_names),
                           k_top=5),
        f"",
        f"## Output inventory",
        f"",
    ]
    for strat in STRATEGIES:
        lines.append(
            f"- outputs/data/phase02_spectrum_{strat}.csv  → "
            f"n_stiff={summaries[strat]['n_stiff']}, "
            f"n_negative={summaries[strat]['n_negative']}"
        )
    lines.append(f"- outputs/figures/phase02_spectrum.png  → 3-strategy overlay")
    lines.append(f"- outputs/diagnostics/phase02.json     → machine summary")
    lines.append("")
    lines += [
        f"## Stop-condition check (CLAUDE.md Phase 2)",
        f"",
        f"- [{'x' if (DATA_DIR / 'phase02_spectrum_subtract.csv').exists() else ' '}] phase02_spectrum_subtract.csv exists",
        f"- [x] n_stiff reported above (subtract → {summaries['subtract']['n_stiff']})",
        f"- [x] negative-eigenvalue audit complete "
        f"({summaries['subtract']['n_negative']} total, "
        f"{n_neg10} at rank ≤ 10)",
        f"",
    ]

    LOG_PATH.write_text("\n".join(lines), encoding="utf-8")

    # Machine-readable JSON summary for Phase 3 gate
    summary = {
        "phase":      2,
        "timestamp":  _dt.datetime.now().isoformat(timespec="seconds"),
        "seed":       SEED,
        "x_star_trial": int(x_star_row["trial"]),
        "x_star_loss_eV_measured": float(x_star_row["loss_eV"]),
        "x_star_loss_eV_GP":       float(mean0),
        "x_star_loss_eV_GP_std":   float(std0),
        "strategies": summaries,
        "warning_negative_top10": bool(n_neg10 > 0),
        "n_negative_top10":       int(n_neg10),
        "passed": True,  # Phase 2 only fails on missing inputs (raised above)
    }
    (LOG_PATH.with_suffix(".json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    # Console echo
    print("\n".join(lines[:35]) + "\n...")
    print(f"\n[done] phase02.log + phase02.json + 3 CSVs + figure written.")
    if n_neg10 > 0:
        print(f"[WARN] {n_neg10} negative eigenvalue(s) at rank ≤ 10 — see log.")


if __name__ == "__main__":
    main()
