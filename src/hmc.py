"""Phase 4 — Hamiltonian Monte Carlo posterior over ReaxFF parameters.

The sklearn GP surrogate is reimplemented in JAX so NumPyro's NUTS can
autograd through it. We extract the trained kernel hyperparameters,
training data, and dual coefficients from the pickle, then evaluate
the predictive mean exactly the same way sklearn does:

    f*(x) = y_mean + y_std * K(x, X_train; θ) @ alpha + intercept

where ``alpha = K_train_train_inv @ (y_normalised)`` is precomputed at
fit time (sklearn stores it as ``gp.alpha_``).

Likelihood
----------
We use a chi-squared-like residual:

    log p(D | θ_phys) = -0.5 * L_gp(θ_phys) / sigma_n^2

The noise scale ``sigma_n`` is **fixed** at the 5-fold cross-validation
RMSE recorded in ``cache/gp_calibration_diag.json`` (~ 0.28 eV) —
i.e. how badly the GP would predict on held-out trials. Documented in
the diagnostic log per the Phase 4 spec.

Parameter prior
---------------
Uniform on the WIDEN box from ``data/parameter_bounds.csv``. To keep
HMC stable we transform each parameter to an unconstrained variable
via the logit warp implemented by ``numpyro.distributions.Uniform``
with NumPyro's built-in support constraint — the sampler proposes in
unconstrained space and NUTS handles the Jacobian automatically.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from .gp_utils import GPBlob


# ---------------------------------------------------------------------------
# JAX-native GP mean predictor (mirrors sklearn's normalize_y=True path)
# ---------------------------------------------------------------------------

@dataclass
class JaxGP:
    """JAX-callable GP mean predictor.

    Reproduces sklearn's ``GaussianProcessRegressor.predict(...).mean`` for
    a Matern (nu=2.5) + WhiteKernel composite. The WhiteKernel adds noise
    only to the training-time kernel (already absorbed into ``alpha``),
    so prediction simply uses the Matern part.
    """
    X_train_scaled:   jnp.ndarray   # (n, d)
    alpha:            jnp.ndarray   # (n,)
    length_scale:     jnp.ndarray   # (d,) ARD
    constant_value:   float         # outer scaling constant in the kernel
    y_mean:           float
    y_std:            float
    scaler_mean:      jnp.ndarray   # (d,)
    scaler_scale:     jnp.ndarray   # (d,)
    nu:               float = 2.5

    def _kernel(self, x_query_scaled: jnp.ndarray) -> jnp.ndarray:
        """Matern(nu=2.5) kernel from one query point (already scaled) to
        all training points."""
        # standardised distance per ARD axis
        dx = (x_query_scaled[None, :] - self.X_train_scaled) / self.length_scale
        # Euclidean norm in standardised coords
        r = jnp.sqrt(jnp.maximum(jnp.sum(dx * dx, axis=1), 1e-30))
        sqrt5 = jnp.sqrt(5.0)
        k = (1.0 + sqrt5 * r + (5.0 / 3.0) * r * r) * jnp.exp(-sqrt5 * r)
        return self.constant_value * k                            # (n,)

    def predict_mean(self, x_orig: jnp.ndarray) -> float:
        """GP-predicted mean at one point in **original** parameter space."""
        x_scaled = (x_orig - self.scaler_mean) / self.scaler_scale
        k_star = self._kernel(x_scaled)
        # Sklearn convention with normalize_y=True
        return self.y_mean + self.y_std * jnp.dot(k_star, self.alpha)


def jax_gp_from_blob(blob: GPBlob) -> JaxGP:
    """Extract the trained sklearn GP into a JAX-callable JaxGP.

    We robustly walk the kernel parameter tree to grab the Matern
    length_scale (ARD), the constant_value of the ConstantKernel, the
    training data, and the dual coefficients.
    """
    gp = blob.gp
    params = gp.kernel_.get_params(deep=True)

    # length_scale: scan for *__length_scale on the Matern component
    length_scale = None
    for name, val in params.items():
        if name.endswith("__length_scale") and not isinstance(val, str):
            length_scale = np.asarray(val, dtype=float)
            break
    if length_scale is None:
        raise RuntimeError("Could not locate Matern length_scale in kernel.")

    # constant_value: from the ConstantKernel
    constant_value = 1.0
    for name, val in params.items():
        if name.endswith("__constant_value") and isinstance(val, (int, float)):
            constant_value = float(val)
            break

    # alpha (dual coefficients) and training X
    if not hasattr(gp, "alpha_") or not hasattr(gp, "X_train_"):
        raise RuntimeError("Trained GP missing alpha_ / X_train_.")
    alpha = np.asarray(gp.alpha_, dtype=float).ravel()
    # sklearn stores X in the SCALED space if ``normalize_y=True`` keeps
    # X unscaled — but here we *explicitly* applied StandardScaler outside
    # the GP, so X_train_ is already in scaled coordinates.
    X_train_scaled = np.asarray(gp.X_train_, dtype=float)

    # normalize_y bookkeeping
    if getattr(gp, "_y_train_mean", None) is None:
        # sklearn ≥ 1.4 keeps these as _y_train_mean / _y_train_std
        y_mean = float(getattr(gp, "_y_train_mean", 0.0))
        y_std  = float(getattr(gp, "_y_train_std",  1.0))
    else:
        y_mean = float(gp._y_train_mean)
        y_std  = float(gp._y_train_std) if hasattr(gp, "_y_train_std") else 1.0

    scaler = blob.scaler
    return JaxGP(
        X_train_scaled = jnp.asarray(X_train_scaled),
        alpha          = jnp.asarray(alpha),
        length_scale   = jnp.asarray(length_scale),
        constant_value = constant_value,
        y_mean         = y_mean,
        y_std          = y_std,
        scaler_mean    = jnp.asarray(scaler.mean_),
        scaler_scale   = jnp.asarray(scaler.scale_),
    )


def verify_jax_matches_sklearn(blob: GPBlob, jaxgp: JaxGP,
                                  X_check: np.ndarray,
                                  rtol: float = 1e-3) -> dict:
    """Sanity-check that the JAX predictor agrees with sklearn within rtol.

    Raises if max relative error exceeds rtol on the check points.
    """
    sklearn_pred = blob.gp.predict(blob.scaler.transform(X_check))
    jax_pred = np.asarray(
        jax.vmap(jaxgp.predict_mean)(jnp.asarray(X_check))
    )
    abs_diff = np.abs(sklearn_pred - jax_pred)
    rel_diff = abs_diff / np.maximum(np.abs(sklearn_pred), 1e-12)
    if rel_diff.max() > rtol:
        raise RuntimeError(
            f"JAX vs sklearn GP mismatch: max abs={abs_diff.max():.3e}, "
            f"max rel={rel_diff.max():.3e}, rtol={rtol:.3e}")
    return {
        "max_abs_diff": float(abs_diff.max()),
        "max_rel_diff": float(rel_diff.max()),
        "n_check":      int(len(X_check)),
    }


# ---------------------------------------------------------------------------
# NumPyro model and sampling driver
# ---------------------------------------------------------------------------

def make_model(jaxgp: JaxGP, lo: jnp.ndarray, hi: jnp.ndarray,
                sigma_n: float, param_names: list[str]):
    """Closure-based NumPyro model factory.

    log p(D | θ) = -0.5 * L_gp(θ) / sigma_n^2
    log p(θ)     = uniform(lo, hi)  (per parameter)
    """
    sigma_n2_inv = 1.0 / (sigma_n ** 2)

    def model():
        # Stack parameters in a single (d,) vector for the GP call.
        theta = jnp.stack([
            numpyro.sample(name, dist.Uniform(lo[i], hi[i]))
            for i, name in enumerate(param_names)
        ])
        loss = jaxgp.predict_mean(theta)
        # χ² → equivalent additive log-likelihood term
        numpyro.factor("gp_loss_likelihood", -0.5 * loss * sigma_n2_inv)
    return model


def init_chains(x_star: np.ndarray, bounds_lo: np.ndarray, bounds_hi: np.ndarray,
                 n_chains: int, seed: int, jitter_frac: float = 0.02
                 ) -> dict[str, jnp.ndarray]:
    """Initial states near x_star, jittered per chain. Each parameter is
    jittered by ``jitter_frac * (hi - lo)``; clipped to (lo, hi)."""
    rng = np.random.default_rng(seed)
    init = {}
    n_params = x_star.size
    for c in range(n_chains):
        delta = rng.normal(0.0, jitter_frac, size=n_params) * (bounds_hi - bounds_lo)
        proposed = np.clip(x_star + delta, bounds_lo + 1e-6, bounds_hi - 1e-6)
        for i in range(n_params):
            init.setdefault(f"_chain{c}", []).append(proposed[i])
    return init


def run_nuts(jaxgp: JaxGP, x_star: np.ndarray,
              lo: np.ndarray, hi: np.ndarray, param_names: list[str],
              sigma_n: float,
              n_warmup: int = 2000, n_samples: int = 5000,
              n_chains: int = 4, target_accept: float = 0.85,
              max_tree_depth: int = 12,
              seed: int = 42, jitter_frac: float = 0.02,
              ) -> tuple[MCMC, dict]:
    """Run NUTS with the prescribed Phase 4 settings.

    Returns the MCMC object and an info dict (realized acceptance per
    chain, divergent count, tree-depth saturation).
    """
    model = make_model(jaxgp,
                       jnp.asarray(lo), jnp.asarray(hi),
                       sigma_n, param_names)
    # Build per-chain init dict in the structure numpyro expects:
    # {param_name: array shape (n_chains,)}
    rng = np.random.default_rng(seed)
    init_dict = {}
    for i, name in enumerate(param_names):
        deltas = rng.normal(0.0, jitter_frac, size=n_chains) * (hi[i] - lo[i])
        vals   = np.clip(x_star[i] + deltas, lo[i] + 1e-6, hi[i] - 1e-6)
        init_dict[name] = jnp.asarray(vals)

    kernel = NUTS(model,
                  target_accept_prob = target_accept,
                  max_tree_depth     = max_tree_depth,
                  dense_mass         = False)
    mcmc = MCMC(kernel,
                num_warmup       = n_warmup,
                num_samples      = n_samples,
                num_chains       = n_chains,
                chain_method     = "sequential",
                progress_bar     = False)
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, init_params=init_dict, extra_fields=(
        "diverging", "num_steps", "accept_prob"
    ))

    extras = mcmc.get_extra_fields()
    info = {
        "n_warmup":         n_warmup,
        "n_samples":        n_samples,
        "n_chains":         n_chains,
        "target_accept":    target_accept,
        "max_tree_depth":   max_tree_depth,
        "seed":             seed,
        "n_divergent":      int(jnp.sum(extras["diverging"])),
        "tree_depth_sat":   float(
            jnp.mean(extras["num_steps"] >= 2 ** max_tree_depth - 1)),
        "realized_accept":  float(jnp.mean(extras["accept_prob"])),
        "realized_accept_per_chain": np.asarray(
            jnp.mean(extras["accept_prob"]
                      .reshape(n_chains, -1), axis=1)).tolist(),
    }
    return mcmc, info


# ---------------------------------------------------------------------------
# Diagnostic gates
# ---------------------------------------------------------------------------

def diagnostic_pass(arviz_summary: pd.DataFrame, info: dict,
                     ess_min: float = 400.0, rhat_max: float = 1.01,
                     tree_sat_max: float = 0.01,
                     acc_lo: float = 0.75, acc_hi: float = 0.95) -> tuple[bool, list[str]]:
    """Apply Phase 4's pass criteria. Returns (passed, list_of_failure_messages)."""
    fails: list[str] = []
    if info["n_divergent"] > 0:
        fails.append(f"divergent_transitions={info['n_divergent']} > 0")
    if info["tree_depth_sat"] > tree_sat_max:
        fails.append(
            f"tree_depth_saturation={info['tree_depth_sat']:.3f} "
            f"> {tree_sat_max}")
    if info["realized_accept"] < acc_lo:
        fails.append(
            f"realized_accept={info['realized_accept']:.3f} < {acc_lo}")
    if info["realized_accept"] > acc_hi:
        fails.append(
            f"realized_accept={info['realized_accept']:.3f} > {acc_hi}  "
            f"(possibly under-tuned)")
    bad_rhat = arviz_summary[arviz_summary["r_hat"] > rhat_max]
    if not bad_rhat.empty:
        fails.append(
            f"{len(bad_rhat)}/{len(arviz_summary)} params have R̂ > {rhat_max}  "
            f"(max R̂={arviz_summary['r_hat'].max():.4f})")
    bad_ess = arviz_summary[arviz_summary["ess_bulk"] < ess_min]
    if not bad_ess.empty:
        fails.append(
            f"{len(bad_ess)}/{len(arviz_summary)} params have ESS_bulk < {ess_min}  "
            f"(min ESS_bulk={arviz_summary['ess_bulk'].min():.1f})")
    return (len(fails) == 0), fails
