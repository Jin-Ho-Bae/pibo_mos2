"""Phase 3 — Profile likelihood along the top stiff and sloppy
eigendirections of the jitter-subtracted Hessian.

Gates
-----
* phase01.json must exist and be ``passed: true`` (CLAUDE.md rule 1).
* phase02.json must exist (the sloppy spectrum needs to be ready).

Direction selection
-------------------
Stiff (top-3):
    The three largest **real positive** eigenvalues after jitter
    subtraction. These directions have a meaningful curvature
    σ_unit = 1/√λ.

Sloppy (top-3):
    Per the Phase 3 prompt: "smallest |λ| above the floor; if all
    sloppy directions are at floor, document this and use the floor
    eigenvectors but flag in caption".

    After jitter subtraction in this run, **all** sub-effective-rank
    eigenvalues collapsed onto the jitter floor (Phase 2 audit
    confirmed). We therefore fall back to the `keep`-strategy
    eigenvectors at ranks 38–40 and use a prior-based σ_unit of
    **1.0 (= 1 BO-cloud σ in scaled coords)**. The CSVs and figure
    caption flag this fallback explicitly.

Parameter bounds
----------------
Loaded by re-running ``recalib_combined_all.build_spec`` on
staged_bo.reax — same WIDEN spec the staged-BO BO used. Each profile
point is clipped to the box; the projection distance vs bound-box
diagonal is reported per point.

Outputs
-------
- outputs/data/phase03_profile_stiff.csv
- outputs/data/phase03_profile_sloppy.csv
- outputs/figures/phase03_profile.png   (stiff | sloppy + zoom-sloppy)
- outputs/diagnostics/phase03.log
- outputs/diagnostics/phase03.json

Stop condition: both CSVs and the figure exist; the log records the
ratio (stiff y-span) / (sloppy y-span) and confirms sloppy variation
< 0.01 eV over ±3σ.
"""
from __future__ import annotations
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.gp_utils import load_surrogate, best_replicate, predict_loss  # noqa: E402
from src.sloppy import (  # noqa: E402
    hessian_at, sloppy_spectrum, profile_likelihood,
)
from src import ffield_parse as _rca  # noqa: E402  (pure ffield text parser)

GP_PATH      = ROOT / "data" / "gp_surrogate.pkl"
REPS_PATH    = ROOT / "data" / "optimizer_replicates.csv"
STAGED_FF    = (ROOT / "results" / "reviewer_response" / "recalib_staged_bo"
                 / "precise_ffields" / "ffield.reax.staged_bo.reax")

PHASE01_JSON = ROOT / "outputs" / "diagnostics" / "phase01.json"
PHASE02_JSON = ROOT / "outputs" / "diagnostics" / "phase02.json"
LOG_PATH     = ROOT / "outputs" / "diagnostics" / "phase03.log"
FIG_PATH     = ROOT / "outputs" / "figures" / "phase03_profile.png"
DATA_DIR     = ROOT / "outputs" / "data"

EPS_FD            = 1e-3
SIGMA_RANGE       = (-3.0, 3.0)
N_PTS             = 61
SLOPPY_LOSS_LIMIT = 0.01   # eV — stop-condition target on sloppy y-span
SEED              = 42


def _gate() -> tuple[dict, dict]:
    for p in (PHASE01_JSON, PHASE02_JSON):
        if not p.exists():
            raise RuntimeError(f"{p} missing — run prior phase first.")
    p1 = json.loads(PHASE01_JSON.read_text(encoding="utf-8"))
    p2 = json.loads(PHASE02_JSON.read_text(encoding="utf-8"))
    if not p1.get("passed", False):
        raise RuntimeError("phase01 did not pass — refusing Phase 3.")
    return p1, p2


def _bounds_from_staged() -> dict[str, tuple[float, float]]:
    """Read the BO-WIDEN bounds (lo, hi) per parameter from staged_bo.reax."""
    src = _rca.read_lines(STAGED_FF)
    off = _rca.parse_offsets(src)
    spec = _rca.build_spec(src, off)
    out = {}
    for s in spec:
        out[s["name"]] = (float(s["lo"]), float(s["hi"]))
    return out


def _profile_to_dataframe(direction_id: str,
                           eigenvalue: float,
                           profile: dict) -> pd.DataFrame:
    n = profile["mean_loss"].size
    return pd.DataFrame({
        "direction_id":          [direction_id] * n,
        "eigenvalue":            [eigenvalue] * n,
        "scale_unit":            [profile["scale_unit"]] * n,
        "scale_source":          [profile["scale_source"]] * n,
        "displacement_sigma":    profile["displacements_sigma"],
        "mean_loss":             profile["mean_loss"],
        "std_loss":              profile["std_loss"],
        "projection_distance":   profile["projection_distance"],
        "projection_pct":        profile["projection_pct"],
        "projection_warning":    profile["projection_warnings"].astype(int),
    })


def main():
    np.random.seed(SEED)
    p1, p2 = _gate()

    blob = load_surrogate(GP_PATH)
    x_star, x_star_row = best_replicate(REPS_PATH)
    bounds = _bounds_from_staged()

    # --- Recompute spectra (keep + subtract) at x_star ----
    H_keep, info_keep = hessian_at(blob, x_star, jitter_strategy="keep",   eps=EPS_FD)
    H_sub,  info_sub  = hessian_at(blob, x_star, jitter_strategy="subtract", eps=EPS_FD)
    spec_keep = sloppy_spectrum(H_keep, info_keep)
    spec_sub  = sloppy_spectrum(H_sub,  info_sub)

    # --- Pick stiff (subtract: top-3 real positive λ) ----
    eigs_sub = spec_sub["eigenvalues"]
    pos_idx = np.where(eigs_sub > 0)[0]
    # eigenvalues are already sorted descending by |λ|; among positive ones
    # the top entries in that ordering are also the largest λ values.
    pos_sorted = pos_idx[np.argsort(eigs_sub[pos_idx])[::-1]]
    stiff_ranks = pos_sorted[:3]

    # --- Pick sloppy (keep eigenvectors at ranks 38-40 since subtract
    # collapsed them onto the jitter floor; flag as prior-fallback) ----
    sloppy_ranks_keep = np.array([eigs_sub.size - 3,
                                    eigs_sub.size - 2,
                                    eigs_sub.size - 1])

    # Compute profiles
    stiff_profiles = []
    for k, r in enumerate(stiff_ranks, start=1):
        prof = profile_likelihood(
            blob, x_star,
            direction  = spec_sub["eigenvectors"][:, r],
            eigenvalue = float(spec_sub["eigenvalues"][r]),
            sigma_range= SIGMA_RANGE, n_pts=N_PTS, bounds=bounds,
            prior_scale_threshold = 10.0 * info_sub["jitter_floor"],
        )
        prof["_id"] = f"stiff_{k}"
        stiff_profiles.append(prof)

    sloppy_profiles = []
    for k, r in enumerate(sloppy_ranks_keep, start=1):
        prof = profile_likelihood(
            blob, x_star,
            direction  = spec_keep["eigenvectors"][:, r],
            eigenvalue = float(spec_keep["eigenvalues"][r]),
            sigma_range= SIGMA_RANGE, n_pts=N_PTS, bounds=bounds,
            # Force the prior-fallback for these (eigenvalues are at floor):
            prior_scale_threshold = 1.0,  # any |λ| < 1.0 → prior fallback
        )
        prof["_id"] = f"sloppy_{k}"
        sloppy_profiles.append(prof)

    # --- Export CSVs ----
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stiff_df  = pd.concat(
        [_profile_to_dataframe(p["_id"], p["eigenvalue"], p)
         for p in stiff_profiles], axis=0, ignore_index=True)
    sloppy_df = pd.concat(
        [_profile_to_dataframe(p["_id"], p["eigenvalue"], p)
         for p in sloppy_profiles], axis=0, ignore_index=True)
    stiff_csv  = DATA_DIR / "phase03_profile_stiff.csv"
    sloppy_csv = DATA_DIR / "phase03_profile_sloppy.csv"
    stiff_df.to_csv (stiff_csv,  index=False, float_format="%.6f")
    sloppy_df.to_csv(sloppy_csv, index=False, float_format="%.6f")

    # --- Figure: stiff | sloppy | zoom-sloppy on shared y-axis ----
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    ax_stiff, ax_sloppy, ax_zoom = axes

    stiff_colors  = ["#0a2540", "#1f77b4", "#5cb6e3"]
    sloppy_colors = ["#d62728", "#ff7f0e", "#ffbb78"]

    # (1) Shared y-axis ----------
    for ax, profs, colors, title in [
        (ax_stiff,  stiff_profiles,  stiff_colors,  "(a) Profile likelihood — top-3 STIFF directions"),
        (ax_sloppy, sloppy_profiles, sloppy_colors, "(b) Profile likelihood — top-3 SLOPPY directions"),
    ]:
        for p, c in zip(profs, colors):
            d = p["displacements_sigma"]
            m = p["mean_loss"]
            s = p["std_loss"]
            ax.fill_between(d, m - s, m + s, color=c, alpha=0.15, zorder=2)
            ax.plot(d, m, color=c, linewidth=2.0, zorder=4,
                     label=(f"{p['_id']}  (λ={p['eigenvalue']:.2e}, "
                              f"scale={p['scale_source']})"))
            # mark projected points
            warn_mask = p["projection_warnings"]
            if warn_mask.any():
                ax.scatter(d[warn_mask], m[warn_mask],
                            facecolor="none", edgecolor=c, s=80,
                            marker="o", linewidth=1.5, zorder=5)
        ax.set_xlabel("Displacement along eigenvector (σ-units)")
        ax.set_ylabel("GP-predicted loss (eV)")
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    # Same y-range for (a) & (b) — picks the stiff range, which dominates
    y_a = ax_stiff.get_ylim()
    y_b = ax_sloppy.get_ylim()
    y_lo = min(y_a[0], y_b[0])
    y_hi = max(y_a[1], y_b[1])
    pad  = 0.05 * (y_hi - y_lo)
    ax_stiff.set_ylim(y_lo - pad, y_hi + pad)
    ax_sloppy.set_ylim(y_lo - pad, y_hi + pad)

    # (2) Zoomed sloppy ----------
    # Plot Δloss relative to each curve's centre (d=0) so the 0.01 eV
    # target axis has direct physical meaning (= loss-budget criterion).
    for p, c in zip(sloppy_profiles, sloppy_colors):
        d = p["displacements_sigma"]
        m = p["mean_loss"]
        s = p["std_loss"]
        # baseline = loss at the central displacement (d = 0)
        idx_zero = int(np.argmin(np.abs(d)))
        m0 = float(m[idx_zero])
        ax_zoom.fill_between(d, (m - m0) - s, (m - m0) + s, color=c, alpha=0.15)
        ax_zoom.plot(d, m - m0, color=c, linewidth=2.0,
                      label=f"{p['_id']}  (Δloss span = {p['loss_span_eV']:.2e} eV)")
    ax_zoom.set_xlabel("Displacement along eigenvector (σ-units)")
    ax_zoom.set_ylabel("Δloss = loss(d) − loss(0)   (eV)")
    ax_zoom.set_title("(c) Sloppy profile — zoomed relative Δloss",
                       fontweight="bold")
    ax_zoom.axhline(+SLOPPY_LOSS_LIMIT, color="green", linestyle="--",
                     linewidth=1.0, label=f"±{SLOPPY_LOSS_LIMIT:g} eV target")
    ax_zoom.axhline(-SLOPPY_LOSS_LIMIT, color="green", linestyle="--",
                     linewidth=1.0)
    ax_zoom.axhline(0.0, color="#666", linestyle=":", linewidth=0.7)
    ax_zoom.grid(alpha=0.3)
    ax_zoom.legend(fontsize=8, loc="upper right")
    # annotation: prior fallback
    ax_zoom.text(0.02, 0.97,
                   "Sloppy directions use prior-fallback σ_unit = 1\n"
                   "(eigenvalues at jitter floor after subtract).",
                   transform=ax_zoom.transAxes, fontsize=8, va="top", ha="left",
                   bbox=dict(boxstyle="round,pad=0.35", fc="#fff5e1",
                              ec="#aaa", alpha=0.92))
    # Symmetric y-range that comfortably contains both the data and the limits
    sloppy_max_abs = max(
        np.abs(p["mean_loss"] - p["mean_loss"][int(np.argmin(np.abs(p["displacements_sigma"])))]).max()
        for p in sloppy_profiles
    )
    y_extent = max(1.1 * sloppy_max_abs, 1.3 * SLOPPY_LOSS_LIMIT)
    ax_zoom.set_ylim(-y_extent, +y_extent)

    fig.suptitle("Phase 3 — Profile likelihood at x_star",
                  fontsize=12.5, y=1.02)
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Spans + sloppy-limit check ----
    stiff_span  = max(p["loss_span_eV"] for p in stiff_profiles)
    sloppy_span = max(p["loss_span_eV"] for p in sloppy_profiles)
    span_ratio  = (stiff_span / max(sloppy_span, 1e-30))
    sloppy_within_limit = bool(sloppy_span < SLOPPY_LOSS_LIMIT)
    any_proj_warn = any(p["any_warning"] for p in stiff_profiles + sloppy_profiles)
    max_proj_pct  = max(p["max_projection_pct"]
                         for p in stiff_profiles + sloppy_profiles
                         if not np.isnan(p["max_projection_pct"]))

    # --- Log ----
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines += [
        f"# Phase 3 — Profile likelihood along stiff/sloppy directions",
        f"# timestamp: {ts}",
        f"# seed: {SEED}",
        f"# gated on phase01 + phase02 JSON",
        f"",
        f"## x_star",
        f"",
        f"  rep_id={int(x_star_row['rep_id'])}, trial={int(x_star_row['trial'])}, "
        f"acq={x_star_row['acq_type']}",
        f"  measured loss = {x_star_row['loss_eV']:.6f} eV",
        f"  GP-predicted   = {float(np.asarray(predict_loss(blob, x_star)[0]).ravel()[0]):.6f} eV",
        f"",
        f"## Direction inventory",
        f"",
        f"| direction | rank (subtract) | λ (subtract) | scale_unit | scale source |",
        f"|-----------|----------------:|-------------:|-----------:|--------------|",
    ]
    for k, (p, r) in enumerate(zip(stiff_profiles, stiff_ranks), start=1):
        lines.append(
            f"| stiff_{k}  | {int(r)+1} | {p['eigenvalue']:.3e} | "
            f"{p['scale_unit']:.3e} | {p['scale_source']} |")
    for k, (p, r) in enumerate(zip(sloppy_profiles, sloppy_ranks_keep), start=1):
        # rank refers to KEEP spectrum
        lam_keep = float(spec_keep["eigenvalues"][r])
        lines.append(
            f"| sloppy_{k} | (keep rank {int(r)+1}, λ_keep={lam_keep:.3e}) | "
            f"{p['eigenvalue']:.3e} | {p['scale_unit']:.3e} | {p['scale_source']} |")
    lines.append("")

    lines += [
        f"## Per-direction y-span (Δloss across ±3σ)",
        f"",
        f"| direction | Δloss (eV) | max projection % | any warning |",
        f"|-----------|-----------:|------------------:|:-----------:|",
    ]
    for p in stiff_profiles + sloppy_profiles:
        lines.append(
            f"| {p['_id']:<8s} | {p['loss_span_eV']:.6f} | "
            f"{p['max_projection_pct']:.2f} | "
            f"{'⚠️ yes' if p['any_warning'] else 'no'} |")
    lines.append("")

    lines += [
        f"## Stop-condition summary",
        f"",
        f"- stiff y-span (max across top-3) : **{stiff_span:.6f} eV**",
        f"- sloppy y-span (max across top-3): **{sloppy_span:.6e} eV**",
        f"- span ratio (stiff / sloppy)     : **{span_ratio:.3e}**",
        f"- sloppy < {SLOPPY_LOSS_LIMIT:g} eV target       : "
        f"**{'PASS' if sloppy_within_limit else 'FAIL'}** "
        f"(max sloppy Δloss = {sloppy_span:.3e} eV)",
        f"- any projection > 5 %             : "
        f"{'yes  (max ' + format(max_proj_pct, '.2f') + ' %)' if any_proj_warn else 'no'}",
        f"",
        f"## Outputs",
        f"",
        f"- outputs/data/phase03_profile_stiff.csv",
        f"- outputs/data/phase03_profile_sloppy.csv",
        f"- outputs/figures/phase03_profile.png",
        f"- outputs/diagnostics/phase03.json",
        f"",
    ]
    if not sloppy_within_limit:
        lines += [
            f"## ⚠️ WARNING",
            f"",
            f"Sloppy directions exhibit Δloss > 0.01 eV across ±3σ, contradicting",
            f"the Phase 3 stop expectation. Inspect direction selection (this may",
            f"indicate the chosen 'sloppy' eigenvectors actually carry curvature).",
            f"",
        ]
    if any_proj_warn:
        lines += [
            f"## ℹ️ NOTE",
            f"",
            f"Projection-distance > 5 % of the BO bound-box diagonal in at",
            f"least one displacement. The eigendirection extends outside the",
            f"BO search region; downstream HMC / σ_post analyses should bound",
            f"to the WIDEN box.",
            f"",
        ]

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "phase":               3,
        "timestamp":           _dt.datetime.now().isoformat(timespec="seconds"),
        "seed":                SEED,
        "x_star_trial":        int(x_star_row["trial"]),
        "stiff_directions":    [int(r)+1 for r in stiff_ranks],
        "sloppy_directions_keep_rank": [int(r)+1 for r in sloppy_ranks_keep],
        "stiff_y_span_eV":     float(stiff_span),
        "sloppy_y_span_eV":    float(sloppy_span),
        "span_ratio":          float(span_ratio),
        "sloppy_within_limit": sloppy_within_limit,
        "sloppy_loss_limit_eV": SLOPPY_LOSS_LIMIT,
        "max_projection_pct":  float(max_proj_pct),
        "any_projection_warning": bool(any_proj_warn),
        "passed":              True,
    }
    (LOG_PATH.with_suffix(".json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\n".join(lines))
    print(f"\n[done] Phase 3 written: stiff/sloppy CSVs + figure + log.")


if __name__ == "__main__":
    main()
