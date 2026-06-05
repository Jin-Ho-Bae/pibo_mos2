"""Phase 5 — σ_opt vs σ_GP vs σ_post three-uncertainty comparison.

Gates:
  outputs/diagnostics/phase04.json must be passed=True.

Outputs:
  outputs/tables/uncertainty_comparison.csv
  outputs/tables/uncertainty_comparison.tex
  outputs/figures/phase05_three_uncertainties.png
  outputs/diagnostics/phase05.log
  outputs/diagnostics/phase05.json

Stop condition:
  CSV exists; the log records the median σ_post / σ_opt ratio across
  the 40 parameters and explicitly flags any parameter whose marginal
  posterior is prior-dominated.
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
from src.gp_utils import load_surrogate, best_replicate  # noqa: E402
from src.uncertainty import comparison_table, PARAM_GROUP_LOOKUP  # noqa: E402

GP_PATH       = ROOT / "data" / "gp_surrogate.pkl"
REPS_PATH     = ROOT / "data" / "optimizer_replicates.csv"
BOUNDS_PATH   = ROOT / "data" / "parameter_bounds.csv"
SAMPLES_NPZ   = ROOT / "outputs" / "data" / "phase04_posterior_samples.npz"
PHASE04_JSON  = ROOT / "outputs" / "diagnostics" / "phase04.json"

OUT_TABLES    = ROOT / "outputs" / "tables"
OUT_DIAG      = ROOT / "outputs" / "diagnostics"
OUT_FIGS      = ROOT / "outputs" / "figures"
SEED          = 42
TOP_K         = 20


GROUP_COLORS = {
    "Bond (Mo-S)":              "#0a2540",
    "Off-diagonal":             "#ff7f0e",
    "Angle":                    "#2ca02c",
    "Atomic non-bonded":        "#d62728",
    "Atomic over-coordination": "#9467bd",
}


def _gate() -> dict:
    if not PHASE04_JSON.exists():
        raise RuntimeError("phase04.json missing — Phase 4 must succeed first.")
    p4 = json.loads(PHASE04_JSON.read_text(encoding="utf-8"))
    if not p4.get("passed", False):
        raise RuntimeError("phase04 not passed — refusing Phase 5.")
    return p4


def _to_latex(df: pd.DataFrame, out_tex: Path) -> None:
    sub = df[[
        "parameter", "group", "x_star",
        "sigma_opt", "sigma_GP_mean", "sigma_post",
        "ratio_post_over_opt", "ci_95_low", "ci_95_high",
        "prior_dominated",
    ]].copy()
    sub["x_star"]              = sub["x_star"].map(lambda v: f"{v:.4g}")
    sub["sigma_opt"]           = sub["sigma_opt"].map(lambda v: f"{v:.3e}")
    sub["sigma_GP_mean"]       = sub["sigma_GP_mean"].map(lambda v: f"{v:.3e}")
    sub["sigma_post"]          = sub["sigma_post"].map(lambda v: f"{v:.3e}")
    sub["ratio_post_over_opt"] = sub["ratio_post_over_opt"].map(lambda v: f"{v:.2f}")
    sub["ci_95_low"]           = sub["ci_95_low"].map(lambda v: f"{v:.3e}")
    sub["ci_95_high"]          = sub["ci_95_high"].map(lambda v: f"{v:.3e}")
    sub["prior_dominated"]     = sub["prior_dominated"].map(lambda v: "yes" if v else "no")
    sub.columns = [
        "parameter", "group", r"$x_\star$",
        r"$\sigma_{\rm opt}$", r"$\sigma_{\rm GP}$", r"$\sigma_{\rm post}$",
        r"$\sigma_{\rm post}/\sigma_{\rm opt}$", r"$\sigma_{\rm post}$ 2.5\%", r"$\sigma_{\rm post}$ 97.5\%",
        "prior-dom.",
    ]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    sub.to_latex(out_tex, index=False, escape=False,
                  caption="Per-parameter three-uncertainty comparison ($\\sigma_{\\rm opt}$, $\\sigma_{\\rm GP}$, $\\sigma_{\\rm post}$). Sorted by posterior/optimizer ratio.",
                  label="tab:phase05_uncertainty")


def _three_bar_figure(df: pd.DataFrame, out: Path) -> None:
    """Horizontal grouped-bar plot, one row per parameter, three bars
    (σ_opt, σ_GP, σ_post) on log scale, grouped by physical block.
    """
    # Sort by group then by parameter name for visual stability
    group_order = ["Bond (Mo-S)", "Off-diagonal", "Angle",
                    "Atomic non-bonded", "Atomic over-coordination"]
    df2 = (df.assign(_group_rank=df["group"].map({g: i for i, g in enumerate(group_order)}))
              .sort_values(["_group_rank", "parameter"])
              .drop(columns="_group_rank")
              .reset_index(drop=True))

    n = len(df2)
    fig_height = max(0.30 * n, 8.0)
    fig, ax = plt.subplots(figsize=(11, fig_height), constrained_layout=True)
    y = np.arange(n)
    h = 0.27

    s_opt   = df2["sigma_opt"].values
    s_gp    = df2["sigma_GP_mean"].values
    s_post  = df2["sigma_post"].values

    # Replace zeros with floor for the log scale to render them
    floor = max(1e-12, np.nanmin([s_opt[s_opt > 0].min() if (s_opt > 0).any() else 1e-9,
                                    s_gp[s_gp > 0].min()   if (s_gp > 0).any() else 1e-9,
                                    s_post[s_post > 0].min() if (s_post > 0).any() else 1e-9]) * 0.5)
    s_opt_p  = np.where(s_opt > 0, s_opt, floor)
    s_gp_p   = np.where(s_gp > 0, s_gp, floor)
    s_post_p = np.where(s_post > 0, s_post, floor)

    ax.barh(y - h, s_opt_p,  height=h, color="#7f7f7f", label=r"$\sigma_{\rm opt}$  (top-20 BO)")
    ax.barh(y,      s_gp_p,   height=h, color="#1f77b4", label=r"$\sigma_{\rm GP}$  (1-D scan std of GP mean loss)")
    ax.barh(y + h, s_post_p, height=h, color="#d62728", label=r"$\sigma_{\rm post}$ (HMC marginal)")

    ax.set_yticks(y)
    label_str = [f"{p}  [{df2.loc[i,'group'][:4]}]" + (" *" if df2.loc[i,'prior_dominated'] else "")
                  for i, p in enumerate(df2['parameter'])]
    ax.set_yticklabels(label_str, fontsize=7)
    ax.invert_yaxis()

    ax.set_xscale("log")
    ax.set_xlabel("magnitude  (parameter units for σ_opt/σ_post; eV for σ_GP)")
    ax.grid(axis="x", which="both", alpha=0.3)

    # Group separators
    bounds = []
    last_g = None
    for i, g in enumerate(df2["group"]):
        if g != last_g and i > 0:
            ax.axhline(i - 0.5, color="black", linestyle=":", linewidth=0.5, alpha=0.4)
        last_g = g

    # Color-band the parameter name area by group
    for i, g in enumerate(df2["group"]):
        ax.axhspan(i - 0.45, i + 0.45, xmin=0, xmax=0.012,
                    color=GROUP_COLORS.get(g, "#888"), alpha=0.85, zorder=0)

    ax.set_title("Phase 5 — Per-parameter three-uncertainty comparison\n"
                  "(σ_opt = optimizer-replicate; σ_GP = surrogate epistemic via 1-D scan; σ_post = HMC marginal)\n"
                  "* = prior-dominated marginal (data does not constrain)",
                  fontsize=10.5, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p4 = _gate()
    np.random.seed(SEED)

    # ---- Load all inputs ----
    blob = load_surrogate(GP_PATH)
    reps = pd.read_csv(REPS_PATH)
    bounds = pd.read_csv(BOUNDS_PATH).set_index("name").reindex(blob.param_names)
    npz = np.load(SAMPLES_NPZ, allow_pickle=True)
    samples = npz["samples"]                  # (chains, draws, d)
    saved_names = list(npz["param_names"])
    if saved_names != list(blob.param_names):
        raise RuntimeError(
            f"Sample param_names mismatch with GP blob:\n"
            f"  GP:     {blob.param_names[:5]}...\n"
            f"  saved:  {saved_names[:5]}...")
    x_star, _ = best_replicate(REPS_PATH)

    # ---- Build comparison table ----
    df = comparison_table(reps, blob, samples, bounds,
                           blob.param_names, x_star, top_K=TOP_K)
    df_sorted = df.sort_values("ratio_post_over_opt", ascending=False).reset_index(drop=True)

    # ---- Export tables ----
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_TABLES / "uncertainty_comparison.csv"
    tex_path = OUT_TABLES / "uncertainty_comparison.tex"
    df_sorted.to_csv(csv_path, index=False, float_format="%.6e")
    _to_latex(df_sorted, tex_path)

    # ---- Figure ----
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    fig_path = OUT_FIGS / "phase05_three_uncertainties.png"
    _three_bar_figure(df_sorted, fig_path)

    # ---- Headline stats ----
    median_ratio    = float(df_sorted["ratio_post_over_opt"].median())
    mean_ratio      = float(df_sorted["ratio_post_over_opt"].mean())
    p10_ratio       = float(df_sorted["ratio_post_over_opt"].quantile(0.10))
    p90_ratio       = float(df_sorted["ratio_post_over_opt"].quantile(0.90))
    n_prior_dom     = int(df_sorted["prior_dominated"].sum())
    n_post_gt_opt   = int((df_sorted["ratio_post_over_opt"] > 1).sum())
    median_gp       = float(df_sorted["sigma_GP_mean"].median())
    median_gp_unc   = float(df_sorted["sigma_GP_std"].median())

    # ---- Log ----
    OUT_DIAG.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Phase 5 — σ_opt vs σ_GP vs σ_post comparison",
        f"# timestamp: {ts}",
        f"# seed: {SEED}",
        f"# gated on phase04.json passed = True",
        f"",
        f"## Method choices",
        f"",
        f"- σ_opt  : sample std (ddof=1) over top-{TOP_K} BO trials by loss_eV",
        f"         (proxy for 500-restart spec; documented in CLAUDE.md)",
        f"- σ_GP   : 1-D scan at posterior medians, parameter i varied from",
        f"         2.5 → 97.5 % posterior quantile (41 pts);",
        f"         σ_GP = std of GP-predicted mean loss across the scan (eV)",
        f"- σ_post : marginal std of HMC chain (chains×draws pooled, ddof=1)",
        f"- CI on σ_post: bootstrap (B=200, resample pooled samples)",
        f"- Prior-dominated flag: posterior 95 % HDI ≥ 90 % of bound range",
        f"",
        f"## Headline stats (40 parameters)",
        f"",
        f"| stat | value |",
        f"|------|------:|",
        f"| median(σ_post / σ_opt) | **{median_ratio:.3f}** |",
        f"| mean  (σ_post / σ_opt) | {mean_ratio:.3f} |",
        f"| 10 % quantile          | {p10_ratio:.3f} |",
        f"| 90 % quantile          | {p90_ratio:.3f} |",
        f"| # params with σ_post > σ_opt | {n_post_gt_opt} / 40 |",
        f"| # prior-dominated params     | {n_prior_dom} / 40 |",
        f"| median σ_GP (1-D scan)        | {median_gp:.3e} eV |",
        f"| median (mean GP std on scan)  | {median_gp_unc:.3e} eV |",
        f"",
    ]
    if n_prior_dom > 0:
        lines += [
            f"## ⚠️ Prior-dominated parameters ({n_prior_dom})",
            f"",
            f"The marginal posterior 95 % HDI covers ≥ 90 % of the prior",
            f"box for the following parameters → the data does **not**",
            f"constrain them; reported σ_post equals (a constant times) the",
            f"prior width and should be reported as a *lower bound* on the",
            f"true parameter uncertainty.",
            f"",
            f"| parameter | group | σ_opt | σ_post | post_q2.5 | post_q97.5 | prior_lo | prior_hi |",
            f"|-----------|-------|------:|-------:|----------:|-----------:|---------:|---------:|",
        ]
        for _, r in df_sorted[df_sorted["prior_dominated"]].iterrows():
            lines.append(
                f"| {r['parameter']} | {r['group']} | "
                f"{r['sigma_opt']:.3e} | {r['sigma_post']:.3e} | "
                f"{r['post_q2_5']:.4g} | {r['post_q97_5']:.4g} | "
                f"{float(bounds.loc[r['parameter'], 'lo']):.4g} | "
                f"{float(bounds.loc[r['parameter'], 'hi']):.4g} |"
            )
        lines.append("")
    else:
        lines += [
            f"## ⚠️ Prior-dominated parameters",
            f"",
            f"  none (the data constrains every parameter inside the BO box).",
            f"",
        ]

    lines += [
        f"## Top-5 most over-confident BO replicates (largest σ_post / σ_opt)",
        f"",
        f"| parameter | group | σ_opt | σ_post | ratio | prior_dom |",
        f"|-----------|-------|------:|-------:|------:|:---------:|",
    ]
    for _, r in df_sorted.head(5).iterrows():
        lines.append(
            f"| {r['parameter']} | {r['group']} | {r['sigma_opt']:.3e} | "
            f"{r['sigma_post']:.3e} | {r['ratio_post_over_opt']:.2f} | "
            f"{'yes' if r['prior_dominated'] else 'no'} |")
    lines.append("")

    lines += [
        f"## Outputs",
        f"  outputs/tables/uncertainty_comparison.csv",
        f"  outputs/tables/uncertainty_comparison.tex",
        f"  outputs/figures/phase05_three_uncertainties.png",
        f"  outputs/diagnostics/phase05.json",
        f"",
        f"## Stop condition",
        f"",
        f"- comparison CSV exists: {csv_path.exists()}",
        f"- median σ_post / σ_opt reported: **{median_ratio:.3f}**",
        f"- prior-bound flag complete: {n_prior_dom} flagged parameters",
    ]

    log_path = OUT_DIAG / "phase05.log"
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    summary = {
        "phase":              5,
        "timestamp":          _dt.datetime.now().isoformat(timespec="seconds"),
        "seed":               SEED,
        "top_K":              TOP_K,
        "n_params":           len(blob.param_names),
        "median_ratio_post_over_opt": median_ratio,
        "mean_ratio_post_over_opt":   mean_ratio,
        "n_post_gt_opt":      n_post_gt_opt,
        "n_prior_dominated":  n_prior_dom,
        "median_sigma_GP_eV": median_gp,
        "median_GP_std_eV":   median_gp_unc,
        "csv":  str(csv_path),
        "tex":  str(tex_path),
        "fig":  str(fig_path),
        "passed": True,
    }
    (OUT_DIAG / "phase05.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] Phase 5 written.")


if __name__ == "__main__":
    main()
