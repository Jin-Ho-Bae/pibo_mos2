"""Phase 5 — three-uncertainty comparison (σ_opt, σ_GP, σ_post).

The reviewer-response uncertainty budget has three sources; this module
computes them on the same 40-parameter scale and returns a
publication-ready table.

Definitions
-----------
σ_opt (parameter)
    Spread of the best converged optimum across optimiser replications.
    The Phase 4 / CLAUDE.md spec calls for ~500 restarts; this codebase
    has 100 sequential staged_bo trials, not 500 independent restarts.
    We use the **sample std (ddof=1) over the top-20 BO trials** as a
    documented proxy. This is the same set Revision#1 Table 3 reports
    as the "uncertainty-aware" spread.

σ_GP (parameter)
    Variation of the GP-predicted loss when **only this parameter is
    varied** along its marginal posterior support, with all other
    parameters held at their marginal-posterior medians. Computed by a
    1-D scan from the 2.5 % to 97.5 % posterior quantile (41 points);
    σ_GP[i] = std of the GP mean-loss along that scan. We also record
    the average GP **predictive std** along the same scan as a
    diagnostic (mean of `gp.predict(x).std`).

σ_post (parameter)
    Marginal std of the parameter posterior from the HMC chain
    (`outputs/data/phase04_posterior_samples.npz`). Computed with
    ddof=1 over chains and draws pooled.

Bootstrap CI on σ_post
----------------------
Block bootstrap (chain-wise) is overkill given 4 chains × 5000 draws;
we use a simple resample-with-replacement (B=200) of the pooled samples
to get a 95 % CI on σ_post[i].

Prior-bound flagging
--------------------
A parameter's posterior is considered "prior-dominated" when the 95 %
HDI of the marginal posterior covers ≥ 90 % of the bound interval
``(hi − lo)``. The corresponding row is flagged in the table.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .gp_utils import GPBlob, predict_loss


# Parameter group taxonomy (same as F9 / M3)
PARAM_GROUP_LOOKUP = {
    "De_sigma_MoS": "Bond (Mo-S)",   "De_pi_MoS":   "Bond (Mo-S)",
    "p_be1_MoS":    "Bond (Mo-S)",   "p_be2_MoS":   "Bond (Mo-S)",
    "p_bo1_MoS":    "Bond (Mo-S)",   "p_bo2_MoS":   "Bond (Mo-S)",
    "p_bo3_MoS":    "Bond (Mo-S)",   "p_bo4_MoS":   "Bond (Mo-S)",
    "p_bo5_MoS":    "Bond (Mo-S)",   "p_bo6_MoS":   "Bond (Mo-S)",
    "p_ovun1_MoS":  "Bond (Mo-S)",
    "Dij_MoS":      "Off-diagonal",  "RvdW_MoS":    "Off-diagonal",
    "Alfa_MoS":     "Off-diagonal",  "ro_sigma_MoS":"Off-diagonal",
    "Thetao_SMoS":  "Angle",         "Thetao_MoSMo":"Angle",
    "p_val1_SMoS":  "Angle",         "p_val1_MoSMo":"Angle",
    "p_val2_SMoS":  "Angle",         "p_val7_SMoS": "Angle",
    "p_val4_SMoS":  "Angle",
    "Mo_RvdW":      "Atomic non-bonded",
    "Mo_Dij":       "Atomic non-bonded",
    "Mo_gamma":     "Atomic non-bonded",
    "Mo_Alfa":      "Atomic non-bonded",
    "Mo_gamma_w":   "Atomic non-bonded",
    "S_RvdW":       "Atomic non-bonded",
    "S_Dij":        "Atomic non-bonded",
    "S_gamma":      "Atomic non-bonded",
    "S_Alfa":       "Atomic non-bonded",
    "S_gamma_w":    "Atomic non-bonded",
    "Mo_p_boc4":    "Atomic over-coordination",
    "Mo_p_boc3":    "Atomic over-coordination",
    "Mo_p_boc5":    "Atomic over-coordination",
    "Mo_p_ovun2":   "Atomic over-coordination",
    "Mo_p_val3":    "Atomic over-coordination",
    "Mo_p_val5":    "Atomic over-coordination",
    "S_p_ovun2":    "Atomic over-coordination",
    "S_p_val3":     "Atomic over-coordination",
}


def sigma_opt(replicates_df: pd.DataFrame, param_names: list[str],
               top_K: int = 20, loss_col: str = "loss_eV") -> pd.Series:
    """Per-parameter std (ddof=1) over the top-K best BO trials.

    Documented proxy for σ_opt — see module docstring.
    """
    if loss_col not in replicates_df.columns:
        raise KeyError(f"'{loss_col}' column missing from replicates_df")
    top = replicates_df.nsmallest(top_K, loss_col)
    out = {n: float(np.std(top[n].values, ddof=1)) for n in param_names}
    return pd.Series(out, name="sigma_opt")


def sigma_post(samples: np.ndarray, param_names: list[str]) -> pd.Series:
    """Per-parameter posterior std (ddof=1) over the pooled HMC chain.

    `samples` shape: (chains, draws, n_params).
    """
    flat = samples.reshape(-1, samples.shape[-1])
    out = {n: float(np.std(flat[:, i], ddof=1)) for i, n in enumerate(param_names)}
    return pd.Series(out, name="sigma_post")


def bootstrap_ci_sigma(samples: np.ndarray, param_names: list[str],
                        B: int = 200, ci: float = 0.95,
                        rng=None) -> pd.DataFrame:
    """Bootstrap-based 95 % CI on σ_post[i] for each parameter."""
    if rng is None:
        rng = np.random.default_rng(42)
    flat = samples.reshape(-1, samples.shape[-1])
    n = flat.shape[0]
    lo_p = (1.0 - ci) / 2.0
    hi_p = 1.0 - lo_p
    rows = []
    for i, name in enumerate(param_names):
        stds = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            stds[b] = float(np.std(flat[idx, i], ddof=1))
        rows.append({
            "name":      name,
            "ci_low":    float(np.quantile(stds, lo_p)),
            "ci_high":   float(np.quantile(stds, hi_p)),
            "ci_median": float(np.median(stds)),
        })
    return pd.DataFrame(rows)


def sigma_gp(blob: GPBlob, samples: np.ndarray,
              param_names: list[str], n_scan: int = 41,
              q_lo: float = 0.025, q_hi: float = 0.975,
              ) -> pd.DataFrame:
    """For each parameter i, 1-D scan its marginal posterior support
    holding all others at their posterior medians; return both:
      - σ_GP_mean : std of GP-predicted mean loss along the scan (eV)
      - σ_GP_std  : average GP epistemic predictive std along the scan

    Documented choice: 1-D scan at posterior medians (Phase 5 spec
    option b).
    """
    flat = samples.reshape(-1, samples.shape[-1])
    medians = np.median(flat, axis=0)            # (n_params,)
    q_lo_v  = np.quantile(flat, q_lo, axis=0)
    q_hi_v  = np.quantile(flat, q_hi, axis=0)

    rows = []
    for i, name in enumerate(param_names):
        if q_hi_v[i] - q_lo_v[i] < 1e-12:
            # Degenerate parameter (posterior is a delta); both terms = 0
            rows.append({
                "name":         name,
                "sigma_GP_mean": 0.0,
                "sigma_GP_std":  0.0,
                "scan_lo":      float(q_lo_v[i]),
                "scan_hi":      float(q_hi_v[i]),
            })
            continue
        scan = np.linspace(q_lo_v[i], q_hi_v[i], n_scan)
        # batch query: build (n_scan, n_params) with only column i varied
        X_q = np.tile(medians, (n_scan, 1))
        X_q[:, i] = scan
        mean, std = predict_loss(blob, X_q)
        rows.append({
            "name":         name,
            "sigma_GP_mean": float(np.std(np.asarray(mean), ddof=1)),
            "sigma_GP_std":  float(np.mean(np.asarray(std))),
            "scan_lo":      float(q_lo_v[i]),
            "scan_hi":      float(q_hi_v[i]),
        })
    return pd.DataFrame(rows)


def prior_dominance_flag(samples: np.ndarray, bounds: pd.DataFrame,
                          param_names: list[str], frac_threshold: float = 0.90
                          ) -> pd.Series:
    """True when the posterior 95 % HDI covers ≥ frac_threshold of the
    bound interval (hi − lo) → the prior, not the data, is constraining.
    """
    flat = samples.reshape(-1, samples.shape[-1])
    out = {}
    for i, name in enumerate(param_names):
        q025 = float(np.quantile(flat[:, i], 0.025))
        q975 = float(np.quantile(flat[:, i], 0.975))
        lo = float(bounds.loc[name, "lo"])
        hi = float(bounds.loc[name, "hi"])
        span_post = q975 - q025
        span_prior = hi - lo
        out[name] = bool(span_post / max(span_prior, 1e-30) >= frac_threshold)
    return pd.Series(out, name="prior_dominated")


def comparison_table(replicates_df: pd.DataFrame,
                      blob: GPBlob,
                      samples: np.ndarray,
                      bounds: pd.DataFrame,
                      param_names: list[str],
                      x_star: np.ndarray,
                      top_K: int = 20) -> pd.DataFrame:
    """Assemble the per-parameter comparison table.

    Columns:
        parameter, group, x_star,
        sigma_opt, sigma_GP_mean, sigma_GP_std, sigma_post,
        ratio_post_over_opt, ci_95_low, ci_95_high,
        prior_dominated, post_q2_5, post_q97_5, scan_lo, scan_hi
    """
    s_opt  = sigma_opt(replicates_df, param_names, top_K=top_K)
    s_post = sigma_post(samples, param_names)
    s_gp   = sigma_gp(blob, samples, param_names).set_index("name")
    ci     = bootstrap_ci_sigma(samples, param_names).set_index("name")
    prior_dom = prior_dominance_flag(samples, bounds, param_names)

    flat = samples.reshape(-1, samples.shape[-1])
    rows = []
    for i, name in enumerate(param_names):
        rows.append({
            "parameter":            name,
            "group":                PARAM_GROUP_LOOKUP.get(name, "?"),
            "x_star":               float(x_star[i]),
            "sigma_opt":            float(s_opt[name]),
            "sigma_GP_mean":        float(s_gp.loc[name, "sigma_GP_mean"]),
            "sigma_GP_std":         float(s_gp.loc[name, "sigma_GP_std"]),
            "sigma_post":           float(s_post[name]),
            "ratio_post_over_opt":  float(s_post[name] / s_opt[name])
                                     if s_opt[name] > 0 else float("inf"),
            "ci_95_low":            float(ci.loc[name, "ci_low"]),
            "ci_95_high":           float(ci.loc[name, "ci_high"]),
            "prior_dominated":      bool(prior_dom[name]),
            "post_q2_5":            float(np.quantile(flat[:, i], 0.025)),
            "post_q97_5":           float(np.quantile(flat[:, i], 0.975)),
            "scan_lo":              float(s_gp.loc[name, "scan_lo"]),
            "scan_hi":              float(s_gp.loc[name, "scan_hi"]),
        })
    df = pd.DataFrame(rows)
    return df
