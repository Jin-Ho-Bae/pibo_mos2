"""Phase 4 — HMC posterior over ReaxFF parameters.

Gating:
  * outputs/diagnostics/phase01.json must be passed=True
  * outputs/diagnostics/phase02.json + phase03.json must exist
  * data/gp_surrogate.pkl, data/parameter_bounds.csv must exist

Sampler:
  * NumPyro NUTS, target_accept=0.85, max_tree_depth=12
  * warmup=2000, samples=5000, 4 chains, sequential
  * Each chain initialised at x_star + 2 % * (hi-lo) Gaussian jitter
    (different seeds per chain, all derived from global seed = 42)

Likelihood:
  * sigma_n FIXED at 0.28 eV (5-fold CV RMSE of the GP, from
    cache/gp_calibration_diag.json) — option (i) of the Phase 4 spec.
    This is the documented choice.

Retry loop:
  * Up to 3 retries on diagnostic failure (R̂, ESS, divergent count,
    realized acceptance, tree-depth saturation).
  * On retry: warmup *= 1.5 ; target_accept ↑ to 0.95 ; tree_depth ↑.
  * 3 failures → AssertionError (CLAUDE.md rule 6).

Outputs:
  outputs/data/phase04_posterior_samples.npz
  outputs/diagnostics/phase04_diagnostics.csv
  outputs/diagnostics/phase04_diagnostics.txt
  outputs/diagnostics/phase04.log
  outputs/diagnostics/phase04.json
  outputs/figures/phase04_traces.png
  outputs/figures/phase04_pairs_top10.png  (top-10 by σ_post / cloud-σ
       since σ_opt is Phase 5; we use cloud σ as a proxy here.)
"""
from __future__ import annotations
import datetime as _dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.gp_utils import load_surrogate, best_replicate  # noqa: E402
from src.hmc import (  # noqa: E402
    jax_gp_from_blob, verify_jax_matches_sklearn, run_nuts, diagnostic_pass,
)

# -- IO ---------------------------------------------------------------------
GP_PATH     = ROOT / "data" / "gp_surrogate.pkl"
REPS_PATH   = ROOT / "data" / "optimizer_replicates.csv"
BOUNDS_PATH = ROOT / "data" / "parameter_bounds.csv"
CV_JSON     = (ROOT / "results" / "reviewer_response" / "recalib_staged_bo"
                 / "manuscript_figs" / "cache" / "gp_calibration_diag.json")
PHASE01     = ROOT / "outputs" / "diagnostics" / "phase01.json"
PHASE02     = ROOT / "outputs" / "diagnostics" / "phase02.json"
PHASE03     = ROOT / "outputs" / "diagnostics" / "phase03.json"

OUT_DATA    = ROOT / "outputs" / "data"
OUT_DIAG    = ROOT / "outputs" / "diagnostics"
OUT_FIGS    = ROOT / "outputs" / "figures"

# -- Defaults (Phase 4 spec) ------------------------------------------------
SEED                 = 42
N_WARMUP             = 2000
N_SAMPLES            = 5000
N_CHAINS             = 4
TARGET_ACCEPT        = 0.85
MAX_TREE_DEPTH       = 12
JITTER_FRAC          = 0.02
ESS_MIN              = 400.0
RHAT_MAX             = 1.01
TREE_SAT_MAX         = 0.01      # 1 % of post-warmup samples
ACC_RANGE            = (0.75, 0.95)
MAX_RETRIES          = 3


def _gate() -> dict:
    for p in (PHASE01, PHASE02, PHASE03):
        if not p.exists():
            raise RuntimeError(f"{p.name} missing — run prior phase first.")
    p1 = json.loads(PHASE01.read_text(encoding="utf-8"))
    if not p1.get("passed", False):
        raise RuntimeError("phase01 did not pass — refusing Phase 4.")
    return p1


def _sigma_n_from_cv() -> tuple[float, str]:
    """Documented option (i): fix sigma_n at the 5-fold CV RMSE."""
    if not CV_JSON.exists():
        # Fall back to a hard-coded reasonable value, but raise about it
        raise RuntimeError(
            f"{CV_JSON} missing — cannot resolve sigma_n. "
            f"Re-run retrain_gp_posterior_in_eV.py."
        )
    diag = json.loads(CV_JSON.read_text(encoding="utf-8"))
    sigma = float(diag["RMSE_mean"])
    return sigma, f"5-fold CV RMSE = {sigma:.4f} eV (cache/gp_calibration_diag.json)"


def _build_arviz_summary(mcmc, param_names: list[str]) -> tuple[az.InferenceData, pd.DataFrame]:
    samples = mcmc.get_samples(group_by_chain=True)
    posterior = {k: np.asarray(samples[k]) for k in param_names}
    idata = az.from_dict(posterior=posterior)
    summary = az.summary(idata, var_names=param_names, round_to=6)
    summary.index.name = "param"
    return idata, summary


def _trace_figure(idata: az.InferenceData, param_names: list[str], out: Path) -> None:
    # Trace plot of every parameter — capped at 8x5 grid (40 params).
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(param_names)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 1.8 * rows),
                              constrained_layout=True)
    axes = np.asarray(axes).ravel()
    chains = idata.posterior.dims["chain"]
    draws  = idata.posterior.dims["draw"]
    for i, name in enumerate(param_names):
        ax = axes[i]
        arr = np.asarray(idata.posterior[name].values)  # (chain, draw)
        for c in range(chains):
            ax.plot(arr[c], linewidth=0.4, alpha=0.7,
                     label=f"chain {c}" if i == 0 else None)
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=6)
    for i in range(len(param_names), len(axes)):
        axes[i].axis("off")
    fig.suptitle("Phase 4 — NUTS traces (40 ReaxFF parameters)", fontsize=11)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _pair_figure(samples_flat: np.ndarray, param_names: list[str],
                  top_idx: np.ndarray, out: Path) -> None:
    """Pair plot of top-10 parameters by σ_post / cloud_σ (proxy for σ_opt)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    sub = samples_flat[:, top_idx]
    names = [param_names[i] for i in top_idx]
    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(2.0 * n, 2.0 * n),
                              constrained_layout=False)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(sub[:, i], bins=40, color="#1f77b4", alpha=0.75)
            elif i > j:
                ax.scatter(sub[:, j], sub[:, i], s=2, alpha=0.15, color="#0a2540")
            else:
                ax.axis("off")
            if i == n - 1:
                ax.set_xlabel(names[j], fontsize=7)
                ax.tick_params(labelsize=6)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(names[i], fontsize=7)
                ax.tick_params(labelsize=6)
            else:
                ax.set_yticklabels([])
    fig.suptitle("Phase 4 — Top-10 posterior pair plot\n"
                  "(highest σ_post / cloud_σ ratio)", fontsize=11, y=0.995)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    _gate()
    numpyro.set_host_device_count(N_CHAINS)
    np.random.seed(SEED)

    # ---- Load GP + bounds + x_star ----
    blob   = load_surrogate(GP_PATH)
    bounds = pd.read_csv(BOUNDS_PATH)
    bounds = bounds.set_index("name").reindex(blob.param_names)
    lo = bounds["lo"].values.astype(float)
    hi = bounds["hi"].values.astype(float)
    x_star, x_star_row = best_replicate(REPS_PATH)

    # ---- JAX GP + verification ----
    jaxgp = jax_gp_from_blob(blob)
    rng = np.random.default_rng(SEED)
    X_check = blob.training_X[rng.choice(blob.training_X.shape[0],
                                            size=min(20, blob.training_X.shape[0]),
                                            replace=False)]
    verify = verify_jax_matches_sklearn(blob, jaxgp, X_check, rtol=1e-3)

    # ---- sigma_n ----
    sigma_n, sigma_n_source = _sigma_n_from_cv()

    OUT_DATA.mkdir(parents=True, exist_ok=True); OUT_DIAG.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)

    # ---- Retry loop ----
    cfg = {
        "n_warmup":       N_WARMUP,
        "n_samples":      N_SAMPLES,
        "target_accept":  TARGET_ACCEPT,
        "max_tree_depth": MAX_TREE_DEPTH,
    }
    history: list[dict] = []
    final_mcmc = None
    final_info = None
    final_summary = None
    passed = False
    for attempt in range(1, MAX_RETRIES + 2):  # 1 base + 3 retries
        print(f"\n[attempt {attempt}] warmup={cfg['n_warmup']}, "
              f"samples={cfg['n_samples']}, target={cfg['target_accept']:.2f}, "
              f"max_depth={cfg['max_tree_depth']}")
        t0 = time.time()
        mcmc, info = run_nuts(
            jaxgp, x_star, lo, hi, blob.param_names, sigma_n,
            n_warmup        = cfg["n_warmup"],
            n_samples       = cfg["n_samples"],
            n_chains        = N_CHAINS,
            target_accept   = cfg["target_accept"],
            max_tree_depth  = cfg["max_tree_depth"],
            seed            = SEED + attempt - 1,
            jitter_frac     = JITTER_FRAC,
        )
        info["wall_s"] = time.time() - t0
        idata, summary = _build_arviz_summary(mcmc, blob.param_names)
        ok, fails = diagnostic_pass(
            summary, info,
            ess_min=ESS_MIN, rhat_max=RHAT_MAX,
            tree_sat_max=TREE_SAT_MAX,
            acc_lo=ACC_RANGE[0], acc_hi=ACC_RANGE[1])
        history.append({
            "attempt":       attempt,
            "cfg":           dict(cfg),
            "n_divergent":   info["n_divergent"],
            "tree_depth_sat": info["tree_depth_sat"],
            "realized_acc":  info["realized_accept"],
            "max_rhat":      float(summary["r_hat"].max()),
            "min_ess_bulk":  float(summary["ess_bulk"].min()),
            "wall_s":        info["wall_s"],
            "passed":        ok,
            "fail_reasons":  fails,
        })
        print(f"  diag: divergent={info['n_divergent']}, "
              f"tree_sat={info['tree_depth_sat']:.3%}, "
              f"acc={info['realized_accept']:.3f}, "
              f"max_R̂={summary['r_hat'].max():.4f}, "
              f"min_ESS={summary['ess_bulk'].min():.0f}")
        if ok:
            final_mcmc = mcmc; final_info = info; final_summary = summary
            final_idata = idata
            passed = True
            break
        # Adjust hyperparams for next retry
        print(f"  [retry] fails: {fails}")
        cfg["n_warmup"]       = int(cfg["n_warmup"] * 1.5)
        cfg["target_accept"]  = min(0.97, cfg["target_accept"] + 0.05)
        cfg["max_tree_depth"] = min(15, cfg["max_tree_depth"] + 1)

    if not passed:
        raise AssertionError(
            f"Phase 4 HMC failed all {MAX_RETRIES + 1} attempts. "
            f"See history: {history}")

    # ---- Persist samples ----
    samples = final_mcmc.get_samples(group_by_chain=True)
    samples_arr = np.stack(
        [np.asarray(samples[n]) for n in blob.param_names], axis=-1
    )  # shape (chains, draws, d)
    np.savez(OUT_DATA / "phase04_posterior_samples.npz",
              param_names=np.array(blob.param_names),
              samples=samples_arr,
              sigma_n=sigma_n,
              x_star=x_star)
    print(f"[npz] phase04_posterior_samples.npz  shape={samples_arr.shape}")

    # ---- Diagnostics CSV ----
    diag_df = final_summary.reset_index().rename(columns={
        "param":   "name", "mean": "mean", "sd": "std",
        "hdi_3%":  "p3", "hdi_97%": "p97",
        "r_hat":   "r_hat", "ess_bulk": "ess_bulk", "ess_tail": "ess_tail",
    })
    # Add empirical 2.5 / 50 / 97.5 percentiles from samples for the
    # exact columns the spec asks for.
    flat = samples_arr.reshape(-1, samples_arr.shape[-1])
    q025 = np.percentile(flat, 2.5,  axis=0)
    q50  = np.percentile(flat, 50.0, axis=0)
    q975 = np.percentile(flat, 97.5, axis=0)
    extra = pd.DataFrame({
        "name": blob.param_names,
        "q2_5": q025, "q50": q50, "q97_5": q975,
    })
    diag_df = diag_df.merge(extra, on="name")
    diag_df = diag_df[[
        "name", "mean", "std", "q2_5", "q50", "q97_5",
        "r_hat", "ess_bulk", "ess_tail",
    ]]
    diag_df.to_csv(OUT_DIAG / "phase04_diagnostics.csv",
                    index=False, float_format="%.6f")
    print(f"[csv] phase04_diagnostics.csv")

    # ---- Diagnostics TXT (full ArviZ summary + divergent count) ----
    with open(OUT_DIAG / "phase04_diagnostics.txt", "w", encoding="utf-8") as f:
        f.write(f"# Phase 4 — ArviZ summary\n")
        f.write(f"# n_chains: {N_CHAINS}\n")
        f.write(f"# n_samples per chain: {cfg['n_samples']}\n")
        f.write(f"# n_warmup (final): {cfg['n_warmup']}\n")
        f.write(f"# sigma_n: {sigma_n_source}\n")
        f.write(f"# realized acceptance: {final_info['realized_accept']:.4f}  "
                f"per-chain: {final_info['realized_accept_per_chain']}\n")
        f.write(f"# divergent transitions: {final_info['n_divergent']}\n")
        f.write(f"# tree-depth saturation: {final_info['tree_depth_sat']:.4%}\n")
        f.write(f"# wall_s: {final_info['wall_s']:.1f}\n\n")
        f.write(final_summary.to_string())

    # ---- Traces figure ----
    _trace_figure(final_idata, blob.param_names,
                    OUT_FIGS / "phase04_traces.png")
    print(f"[fig] phase04_traces.png")

    # ---- Top-10 pair plot (σ_post / cloud_σ proxy ratio) ----
    cloud_sigma = np.std(blob.training_X, axis=0)
    sigma_post = diag_df["std"].values
    # Avoid division by zero
    ratio = sigma_post / np.maximum(cloud_sigma, 1e-9)
    top10 = np.argsort(ratio)[::-1][:10]
    _pair_figure(flat, blob.param_names, top10,
                   OUT_FIGS / "phase04_pairs_top10.png")
    print(f"[fig] phase04_pairs_top10.png  (top-10 by σ_post/cloud_σ ratio)")

    # ---- Log + JSON summary ----
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines = [
        f"# Phase 4 — HMC posterior over ReaxFF parameters",
        f"# timestamp: {ts}",
        f"# seed: {SEED}",
        f"",
        f"## JAX GP verification vs sklearn (20 random training points)",
        f"  max_abs_diff : {verify['max_abs_diff']:.3e}",
        f"  max_rel_diff : {verify['max_rel_diff']:.3e}  (rtol=1e-3)",
        f"",
        f"## Likelihood",
        f"  sigma_n      : {sigma_n_source}",
        f"  log p(D|θ)   : -0.5 * L_gp(θ) / σ_n²",
        f"  prior        : Uniform on parameter_bounds.csv (NumPyro auto-transforms)",
        f"",
        f"## Sampler — final settings (after retries)",
        f"  warmup       : {cfg['n_warmup']}",
        f"  samples      : {cfg['n_samples']}",
        f"  chains       : {N_CHAINS}",
        f"  target_accept: {cfg['target_accept']:.2f}",
        f"  max_tree_depth: {cfg['max_tree_depth']}",
        f"  init jitter  : {JITTER_FRAC:.0%} of bound range, per chain",
        f"",
        f"## Realised diagnostics",
        f"  divergent transitions    : {final_info['n_divergent']}",
        f"  tree-depth saturation     : {final_info['tree_depth_sat']:.4%}",
        f"  realised acceptance (avg) : {final_info['realized_accept']:.4f}",
        f"  realised acceptance per chain: "
        + ", ".join(f"{a:.3f}" for a in final_info['realized_accept_per_chain']),
        f"  max R̂                    : {final_summary['r_hat'].max():.4f}",
        f"  min ESS_bulk              : {final_summary['ess_bulk'].min():.1f}",
        f"  min ESS_tail              : {final_summary['ess_tail'].min():.1f}",
        f"  wall time                 : {final_info['wall_s']:.1f} s",
        f"",
        f"## Retry history",
    ]
    for h in history:
        log_lines.append(
            f"  attempt {h['attempt']}: "
            f"divergent={h['n_divergent']}, tree_sat={h['tree_depth_sat']:.2%}, "
            f"acc={h['realized_acc']:.3f}, max_R̂={h['max_rhat']:.4f}, "
            f"min_ESS={h['min_ess_bulk']:.0f}, wall={h['wall_s']:.0f}s, "
            f"passed={h['passed']}"
        )
    log_lines += [
        f"",
        f"## Stop-condition check",
        f"",
        f"- R̂ < {RHAT_MAX}    : "
        f"{'PASS' if final_summary['r_hat'].max() < RHAT_MAX else 'FAIL'} "
        f"(max = {final_summary['r_hat'].max():.4f})",
        f"- ESS_bulk ≥ {ESS_MIN:.0f}: "
        f"{'PASS' if final_summary['ess_bulk'].min() >= ESS_MIN else 'FAIL'} "
        f"(min = {final_summary['ess_bulk'].min():.0f})",
        f"- 0 divergent     : "
        f"{'PASS' if final_info['n_divergent'] == 0 else 'FAIL'} "
        f"(observed = {final_info['n_divergent']})",
        f"- tree_sat < {TREE_SAT_MAX*100:.0f} %  : "
        f"{'PASS' if final_info['tree_depth_sat'] < TREE_SAT_MAX else 'FAIL'} "
        f"({final_info['tree_depth_sat']:.3%})",
        f"- realised acc ∈ [{ACC_RANGE[0]:.2f},{ACC_RANGE[1]:.2f}] : "
        f"{'PASS' if ACC_RANGE[0] <= final_info['realized_accept'] <= ACC_RANGE[1] else 'FAIL'} "
        f"({final_info['realized_accept']:.3f})",
        f"",
        f"## Outputs",
        f"  outputs/data/phase04_posterior_samples.npz",
        f"  outputs/diagnostics/phase04_diagnostics.csv",
        f"  outputs/diagnostics/phase04_diagnostics.txt",
        f"  outputs/diagnostics/phase04.json",
        f"  outputs/figures/phase04_traces.png",
        f"  outputs/figures/phase04_pairs_top10.png",
    ]
    (OUT_DIAG / "phase04.log").write_text("\n".join(log_lines), encoding="utf-8")
    print("\n".join(log_lines[-20:]))

    summary_json = {
        "phase":            4,
        "timestamp":        _dt.datetime.now().isoformat(timespec="seconds"),
        "seed":             SEED,
        "sigma_n":          sigma_n,
        "sigma_n_source":   sigma_n_source,
        "final_cfg":        dict(cfg),
        "n_divergent":      final_info["n_divergent"],
        "tree_depth_sat":   final_info["tree_depth_sat"],
        "realized_accept":  final_info["realized_accept"],
        "max_rhat":         float(final_summary["r_hat"].max()),
        "min_ess_bulk":     float(final_summary["ess_bulk"].min()),
        "min_ess_tail":     float(final_summary["ess_tail"].min()),
        "wall_s":           final_info["wall_s"],
        "attempts":         len(history),
        "passed":           True,
    }
    (OUT_DIAG / "phase04.json").write_text(
        json.dumps(summary_json, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
