"""Phase 6 — Manuscript Figure 9 assembly.

Six-panel publication-ready figure consolidating Phases 2 – 5:

  (a) Sloppy spectrum  (jitter-subtracted, GP-regularisation floor marked)
  (b) Profile likelihood — top-3 STIFF directions
  (c) Profile likelihood — top-3 SLOPPY directions, with an inset that
      overlays (b) and (c) on the same y-axis so the magnitude gap is
      immediately visible
  (d) σ_opt: optimiser-replicate correlation
      (40 × 40 Pearson r over top-20 BO trial parameter values — same
      sample set Phase 5 used for σ_opt; documented proxy for "500
      restarts" in the staged-BO context.)
  (e) Parameter-wise σ_opt / σ_GP / σ_post comparison (from Phase 5)
  (f) σ_post: parameter posterior correlation (HMC)
      (40 × 40 Pearson r over the pooled 4 × 5000 NUTS draws.)

Stop condition: outputs/figures/F9.pdf AND outputs/figures/F9.png exist
and outputs/diagnostics/phase06.log records the SHA-256 checksum +
row count of every source CSV.
"""
from __future__ import annotations
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.plotting import (  # noqa: E402
    consistent_style, panel_letter, shared_correlation_colorbar,
)

OUT_FIGS  = ROOT / "outputs" / "figures"
OUT_DIAG  = ROOT / "outputs" / "diagnostics"
OUT_DATA  = ROOT / "outputs" / "data"
OUT_TBLS  = ROOT / "outputs" / "tables"

SOURCES = {
    "(a) spectrum":   OUT_DATA / "phase02_spectrum_subtract.csv",
    "(b) stiff":      OUT_DATA / "phase03_profile_stiff.csv",
    "(c) sloppy":     OUT_DATA / "phase03_profile_sloppy.csv",
    "(d) sigma_opt":  ROOT / "data" / "optimizer_replicates.csv",
    "(e) comparison": OUT_TBLS / "uncertainty_comparison.csv",
    "(f) sigma_post": OUT_DATA / "phase04_posterior_samples.npz",
}

# Group taxonomy (same as Phase 5)
GROUP_ORDER  = ["Bond (Mo-S)", "Off-diagonal", "Angle",
                  "Atomic non-bonded", "Atomic over-coordination"]
GROUP_COLORS = {
    "Bond (Mo-S)":              "#0a2540",
    "Off-diagonal":             "#ff7f0e",
    "Angle":                    "#2ca02c",
    "Atomic non-bonded":        "#d62728",
    "Atomic over-coordination": "#9467bd",
}


def _file_checksum(p: Path) -> tuple[str, int]:
    """SHA-256 hex digest and byte size of a file (or '∅', 0 if missing)."""
    if not p.exists():
        return "∅", 0
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk); n += len(chunk)
    return h.hexdigest()[:16], n   # 16-hex prefix is plenty for source-id


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------

def _panel_a_spectrum(ax) -> int:
    df = pd.read_csv(SOURCES["(a) spectrum"])
    abs_l = df["abs_eigenvalue"].values
    sign  = df["sign"].values
    rank  = df["rank"].values

    pos = sign > 0; neg = sign < 0
    ax.scatter(rank[pos], abs_l[pos], s=28, color="#0a2540",
                edgecolors="black", linewidths=0.4,
                zorder=3, label="λ > 0")
    ax.scatter(rank[neg], abs_l[neg], s=40, facecolors="none",
                edgecolors="#d62728", linewidths=1.0, marker="v",
                zorder=3, label="λ < 0 (post-subtract)")
    ax.plot(rank, abs_l, color="#666", linewidth=0.5, alpha=0.5, zorder=1)

    # Jitter floor (= WhiteKernel noise level used in Phase 2)
    floor = 1e-4
    ax.axhline(floor, color="#888", linestyle="--", linewidth=0.8,
                label="GP regularization floor")
    ax.axhline(10.0 * floor, color="#bbb", linestyle=":", linewidth=0.7)

    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue rank (sorted by |λ|, descending)")
    ax.set_ylabel("|λ| of GP loss-Hessian")
    ax.set_title("Sloppy spectrum (jitter-subtracted)")
    ax.grid(which="both", alpha=0.25)
    ax.legend(loc="lower left", fontsize=6, frameon=True, ncol=1)
    return len(df)


def _panel_b_stiff(ax) -> int:
    df = pd.read_csv(SOURCES["(b) stiff"])
    colors = {"stiff_1": "#0a2540", "stiff_2": "#1f77b4", "stiff_3": "#5cb6e3"}
    for did, sub in df.groupby("direction_id"):
        sub = sub.sort_values("displacement_sigma")
        c = colors.get(did, "#333")
        eig = float(sub["eigenvalue"].iloc[0])
        ax.fill_between(sub["displacement_sigma"],
                          sub["mean_loss"] - sub["std_loss"],
                          sub["mean_loss"] + sub["std_loss"],
                          color=c, alpha=0.15)
        ax.plot(sub["displacement_sigma"], sub["mean_loss"],
                 color=c, linewidth=1.5, label=f"{did} (λ={eig:.2e})")
    ax.set_xlabel("Displacement (σ-units)")
    ax.set_ylabel("GP-predicted loss (eV)")
    ax.set_title("Profile likelihood — top-3 STIFF")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", fontsize=6, frameon=True, ncol=3,
                bbox_to_anchor=(0.5, -0.18))
    return len(df)


def _panel_c_sloppy_with_inset(ax) -> int:
    df_sloppy = pd.read_csv(SOURCES["(c) sloppy"])
    df_stiff  = pd.read_csv(SOURCES["(b) stiff"])
    sloppy_colors = {"sloppy_1": "#d62728", "sloppy_2": "#ff7f0e", "sloppy_3": "#ffbb78"}
    for did, sub in df_sloppy.groupby("direction_id"):
        sub = sub.sort_values("displacement_sigma")
        c = sloppy_colors.get(did, "#aaa")
        m0 = float(sub.loc[sub["displacement_sigma"].abs().idxmin(), "mean_loss"])
        ax.fill_between(sub["displacement_sigma"],
                          (sub["mean_loss"] - m0) - sub["std_loss"],
                          (sub["mean_loss"] - m0) + sub["std_loss"],
                          color=c, alpha=0.18)
        ax.plot(sub["displacement_sigma"], sub["mean_loss"] - m0,
                 color=c, linewidth=1.5,
                 label=f"{did} (Δ ≤ {sub['mean_loss'].max() - sub['mean_loss'].min():.1e} eV)")
    # 0.01 eV target lines
    ax.axhline(+0.01, color="#0a8f00", linestyle="--", linewidth=0.7)
    ax.axhline(-0.01, color="#0a8f00", linestyle="--", linewidth=0.7,
                label="±0.01 eV target")
    ax.axhline(0.0, color="#666", linestyle=":", linewidth=0.6)
    ax.set_xlabel("Displacement (σ-units)")
    ax.set_ylabel(r"$\Delta$loss (eV)")
    ax.set_title("Profile likelihood — top-3 SLOPPY (zoomed Δloss)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", fontsize=6, frameon=True, ncol=2,
                bbox_to_anchor=(0.5, -0.18))

    # ---- INSET: overlay (b)+(c) on the same axis ----
    inset = ax.inset_axes([0.62, 0.55, 0.36, 0.40])
    for did, sub in df_stiff.groupby("direction_id"):
        sub = sub.sort_values("displacement_sigma")
        c = {"stiff_1": "#0a2540", "stiff_2": "#1f77b4",
             "stiff_3": "#5cb6e3"}.get(did, "#333")
        inset.plot(sub["displacement_sigma"], sub["mean_loss"],
                    color=c, linewidth=1.0, alpha=0.9)
    for did, sub in df_sloppy.groupby("direction_id"):
        sub = sub.sort_values("displacement_sigma")
        c = sloppy_colors.get(did, "#aaa")
        inset.plot(sub["displacement_sigma"], sub["mean_loss"],
                    color=c, linewidth=1.0, alpha=0.9)
    inset.set_title("(b)+(c) on same y", fontsize=6, fontweight="normal")
    inset.set_xlabel("σ", fontsize=6); inset.set_ylabel("loss (eV)", fontsize=6)
    inset.tick_params(labelsize=5)
    inset.grid(alpha=0.3)
    return len(df_sloppy) + len(df_stiff)


def _panel_d_sigma_opt_corr(ax) -> int:
    reps = pd.read_csv(SOURCES["(d) sigma_opt"])
    # Same selection as Phase 5: top-20 trials by loss_eV
    top = reps.nsmallest(20, "loss_eV")
    book = {"rep_id", "trial", "acq_type", "loss_eV"}
    param_cols = [c for c in reps.columns if c not in book]
    M = top[param_cols].values.astype(float)
    corr = np.corrcoef(M.T)

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(r"$\sigma_{\rm opt}$: replicate correlation"
                  + f"  (top-{len(top)} BO trials)")
    ax.set_xlabel("parameter index"); ax.set_ylabel("parameter index")
    return int(M.shape[0])


def _panel_e_comparison(ax) -> int:
    df = pd.read_csv(SOURCES["(e) comparison"])
    # x = log10(sigma_opt), y = log10(sigma_post), colored by group, size = sigma_GP
    eps = 1e-12
    so = np.maximum(df["sigma_opt"].values, eps)
    sp = np.maximum(df["sigma_post"].values, eps)
    sg = np.maximum(df["sigma_GP_mean"].values, eps)
    # Marker size: log-scale, bounded
    sg_log = np.log10(sg)
    sg_norm = (sg_log - sg_log.min()) / max(sg_log.max() - sg_log.min(), 1e-9)
    sizes = 25.0 + 100.0 * sg_norm

    for g, color in GROUP_COLORS.items():
        m = df["group"].values == g
        if not m.any():
            continue
        ax.scatter(so[m], sp[m], s=sizes[m], c=color, alpha=0.78,
                    edgecolors="black", linewidths=0.4, label=g)
    # Diagonal σ_post = σ_opt
    lo, hi = min(so.min(), sp.min()) / 1.5, max(so.max(), sp.max()) * 1.5
    ax.plot([lo, hi], [lo, hi], color="#444", linestyle="--", linewidth=0.8,
             zorder=0, label=r"$\sigma_{\rm post} = \sigma_{\rm opt}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\sigma_{\rm opt}$  (parameter units)")
    ax.set_ylabel(r"$\sigma_{\rm post}$  (parameter units)")
    ax.set_title("σ_opt vs σ_GP (marker size) vs σ_post")
    ax.grid(which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=6, frameon=True, ncol=1)
    # Marker-size legend (size = σ_GP)
    handles = [
        Line2D([0], [0], marker="o", linestyle="None",
                markerfacecolor="white", markeredgecolor="black",
                markersize=np.sqrt(s) * 0.7,
                label=f"σ_GP = {v:.1e} eV")
        for s, v in zip([30, 70, 130],
                          [10 ** (sg_log.min()),
                            10 ** (0.5 * (sg_log.min() + sg_log.max())),
                            10 ** (sg_log.max())])
    ]
    ax.legend(handles=ax.get_legend().legend_handles + handles,
                loc="upper left", fontsize=6, frameon=True, ncol=1)
    return len(df)


def _panel_f_sigma_post_corr(ax) -> int:
    npz = np.load(SOURCES["(f) sigma_post"], allow_pickle=True)
    samples = npz["samples"]  # (chains, draws, d)
    flat = samples.reshape(-1, samples.shape[-1])
    corr = np.corrcoef(flat.T)

    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(r"$\sigma_{\rm post}$: posterior correlation (HMC)"
                  + f"  (n={flat.shape[0]})")
    ax.set_xlabel("parameter index"); ax.set_ylabel("parameter index")
    return int(flat.shape[0])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    consistent_style()
    # Single-column-friendly two-page-spread aspect: 16 × 10
    fig, axes = plt.subplots(2, 3, figsize=(16, 10),
                              constrained_layout=False)
    ax_a, ax_b, ax_c = axes[0]
    ax_d, ax_e, ax_f = axes[1]

    n_rows: dict[str, int] = {}
    n_rows["(a) spectrum"]   = _panel_a_spectrum(ax_a);            panel_letter(ax_a, "a")
    n_rows["(b) stiff"]      = _panel_b_stiff(ax_b);                panel_letter(ax_b, "b")
    n_rows["(c) sloppy"]     = _panel_c_sloppy_with_inset(ax_c);    panel_letter(ax_c, "c")
    n_rows["(d) sigma_opt"]  = _panel_d_sigma_opt_corr(ax_d);       panel_letter(ax_d, "d")
    n_rows["(e) comparison"] = _panel_e_comparison(ax_e);           panel_letter(ax_e, "e")
    n_rows["(f) sigma_post"] = _panel_f_sigma_post_corr(ax_f);      panel_letter(ax_f, "f")

    # Shared correlation colorbar for panels (d) and (f).
    shared_correlation_colorbar(fig, [ax_d, ax_f],
                                  label="Pearson r")

    fig.suptitle("Figure 9 — ReaxFF uncertainty quantification at $x_\\star$",
                  fontsize=12, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.06, right=0.93, top=0.94, bottom=0.07,
                          wspace=0.32, hspace=0.40)

    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_FIGS / "F9.pdf"
    png_path = OUT_FIGS / "F9.png"
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    # ---- Diagnostic log with CSV checksums ----
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Phase 6 — Figure 9 manuscript assembly",
        f"# timestamp: {ts}",
        f"",
        f"## Outputs",
        f"  outputs/figures/F9.pdf  ({pdf_path.stat().st_size:,} bytes)",
        f"  outputs/figures/F9.png  ({png_path.stat().st_size:,} bytes)",
        f"",
        f"## Source CSV / NPZ provenance",
        f"",
        f"| panel | path | rows used | sha256 prefix | size (bytes) |",
        f"|-------|------|----------:|---------------|-------------:|",
    ]
    digests = {}
    for panel, path in SOURCES.items():
        sha, sz = _file_checksum(path)
        digests[panel] = {"path": str(path), "sha256_16": sha, "size_bytes": sz,
                          "rows_or_samples": n_rows[panel]}
        lines.append(
            f"| {panel} | {path.relative_to(ROOT)} | "
            f"{n_rows[panel]:,} | `{sha}` | {sz:,} |"
        )

    lines += [
        f"",
        f"## Stop condition",
        f"",
        f"- F9.pdf exists  : {pdf_path.exists()}",
        f"- F9.png exists  : {png_path.exists()}",
        f"- every source provenance recorded above (sha256 + row count)",
    ]
    log_path = OUT_DIAG / "phase06.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    (OUT_DIAG / "phase06.json").write_text(json.dumps({
        "phase":     6,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "pdf":       str(pdf_path),
        "png":       str(png_path),
        "sources":   digests,
        "passed":    True,
    }, indent=2), encoding="utf-8")
    print(f"\n[done] Phase 6 → F9.pdf + F9.png + phase06.log/.json")


if __name__ == "__main__":
    main()
