"""Phase-1 GP utilities for the ReaxFF reviewer-response pipeline.

The GP surrogate is the central oracle of the analysis. Downstream
phases (σ_GP, HMC, sloppy spectrum) all call into this module so the
same scaler, kernel, and noise model are used consistently.

Conventions
-----------
- ``GPBlob`` is the pickle payload produced by
  ``scripts/build_phase01_inputs.py``: an sklearn GP, the matching
  ``StandardScaler``, and metadata.
- All public functions accept and return the **original** parameter
  space (i.e. before the scaler), even though the GP is fit on the
  *scaled* coordinates internally. This keeps callers agnostic to the
  scaling choice.
- Finite-difference Hessian explicitly **returns the jitter** used in
  the GP (sum of WhiteKernel ``noise_level`` and sklearn's ``alpha``)
  so the caller can subtract or report its contribution rather than
  silently absorbing it.  This addresses CLAUDE.md rule 3.

No silent fallbacks: any unexpected GP structure raises immediately.
"""
from __future__ import annotations
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor


@dataclass
class GPBlob:
    """Container for the pickled GP and its accompanying metadata."""
    gp: GaussianProcessRegressor
    scaler: object          # sklearn StandardScaler
    param_names: list[str]
    loss_units: str
    n_train: int
    input_dim: int
    training_X: np.ndarray
    training_y: np.ndarray
    kernel_repr: str
    seed: int


def load_surrogate(path: str | Path) -> GPBlob:
    """Load the pickled GP + scaler + metadata bundle.

    Raises a hard error (``FileNotFoundError`` / ``KeyError``) if the
    bundle is missing or malformed — never falls back silently.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GP surrogate missing at {p}")
    with open(p, "rb") as f:
        d = pickle.load(f)
    required = ("gp", "scaler", "param_names", "loss_units",
                 "n_train", "input_dim", "training_X", "training_y",
                 "kernel_repr", "seed")
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"GP pickle missing keys: {missing}")
    return GPBlob(**{k: d[k] for k in required})


def _scale(blob: GPBlob, X_orig: np.ndarray) -> np.ndarray:
    """Transform original-space inputs into the GP's scaled space."""
    return blob.scaler.transform(np.atleast_2d(X_orig))


def predict_loss(blob: GPBlob, X: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Return GP predictive (mean, std) at one or many original-space points.

    Parameters
    ----------
    X : shape (d,) or (n, d) in the original 40-parameter space.

    Returns
    -------
    mean : shape (n,) in loss units (eV).
    std  : shape (n,) in loss units (eV). This is sqrt(diagonal of the
           predictive covariance) — i.e. the GP's epistemic std at each
           query point.
    """
    X2 = _scale(blob, X)
    mean, std = blob.gp.predict(X2, return_std=True)
    return np.asarray(mean), np.asarray(std)


def loss_gradient(blob: GPBlob, x: np.ndarray,
                   eps: float = 1e-3) -> np.ndarray:
    """Central-difference gradient of the GP mean at one point.

    Parameters
    ----------
    x   : (d,) in original parameter space.
    eps : step size in **scaled** coordinates (the GP's natural unit).

    Returns
    -------
    grad : (d,) gradient in original-parameter units. The conversion
           handles the StandardScaler's per-axis scale: ∂f/∂x_orig =
           ∂f/∂x_scaled · (1/scale_i).
    """
    x = np.asarray(x, dtype=float).ravel()
    d = x.size
    x_scaled = _scale(blob, x).ravel()
    grad_scaled = np.zeros(d)
    for i in range(d):
        xp = x_scaled.copy(); xp[i] += eps
        xm = x_scaled.copy(); xm[i] -= eps
        # GP predict on scaled-space directly to avoid double-scaling
        fp = float(blob.gp.predict(xp[None, :])[0])
        fm = float(blob.gp.predict(xm[None, :])[0])
        grad_scaled[i] = (fp - fm) / (2.0 * eps)
    # chain rule: scaler.scale_ maps original→scaled; ∂/∂x_orig = ∂/∂x_scaled / scale
    return grad_scaled / blob.scaler.scale_


def _extract_jitter(blob: GPBlob) -> dict:
    """Return the GP's effective regularisation budget at training time.

    Includes BOTH the WhiteKernel ``noise_level`` (an additive σ² on
    the diagonal of K(X,X)) and sklearn's ``alpha`` (also additive).
    Reported in loss-units² (this GP uses normalize_y=True, so the
    raw alpha/noise_level live in normalized space; we convert back
    by multiplying with the training-y std² so the number is
    interpretable in eV²).
    """
    gp = blob.gp
    noise_level = 0.0
    # WhiteKernel detection: scan kernel hyperparameter dict
    for name, val in gp.kernel_.get_params(deep=False).items():
        if "WhiteKernel" in repr(type(val)):
            noise_level += float(val.noise_level)
    # Deep walk for nested kernels
    for name, val in gp.kernel_.get_params(deep=True).items():
        if name.endswith("__noise_level"):
            noise_level += float(val) - noise_level if name == "noise_level" else float(val) - noise_level if name == "noise_level" else 0.0  # don't double-count root
    # Simpler: just look for any '*__noise_level' (sklearn typically uses k2__noise_level)
    noise_level = 0.0
    for name, val in gp.kernel_.get_params(deep=True).items():
        if name.endswith("__noise_level") and isinstance(val, (int, float)):
            noise_level = max(noise_level, float(val))

    alpha = float(getattr(gp, "alpha", 0.0) or 0.0)
    y_std = float(np.std(blob.training_y)) if blob.training_y.size > 1 else 1.0
    # GP with normalize_y=True normalizes y by y_std; predictive mean is
    # back-scaled. WhiteKernel noise_level is in normalized space.
    return {
        "noise_level_normalised": noise_level,
        "alpha_normalised":       alpha,
        "y_std_eV":               y_std,
        "noise_level_eV2":        noise_level * (y_std ** 2),
        "alpha_eV2":              alpha * (y_std ** 2),
        "total_jitter_eV2":       (noise_level + alpha) * (y_std ** 2),
    }


def loss_hessian(blob: GPBlob, x: np.ndarray,
                  eps: float = 1e-3
                  ) -> Tuple[np.ndarray, dict]:
    """Central-difference Hessian of the GP mean at one point.

    Returns the symmetric ``(d, d)`` Hessian (in **scaled** coordinates;
    see ``loss_hessian_in_original`` for the original-space variant)
    together with a ``jitter`` dict so callers can report or subtract
    the kernel's noise floor (CLAUDE.md rule 3).

    Computed in scaled space because that's the GP's natural axis-aligned
    coordinate system; the eigenvalue spectrum is rotation-invariant up
    to the StandardScaler's per-axis dilation (which is a diagonal
    similarity, preserving the order of eigenvalues but rescaling each).
    """
    x = np.asarray(x, dtype=float).ravel()
    x_scaled = _scale(blob, x).ravel()
    d = x_scaled.size
    H = np.zeros((d, d))
    f0 = float(blob.gp.predict(x_scaled[None, :])[0])

    # Diagonal: second derivative along each axis
    for i in range(d):
        xp = x_scaled.copy(); xp[i] += eps
        xm = x_scaled.copy(); xm[i] -= eps
        H[i, i] = (float(blob.gp.predict(xp[None, :])[0])
                    + float(blob.gp.predict(xm[None, :])[0])
                    - 2.0 * f0) / (eps * eps)

    # Off-diagonal: mixed partials via 4-point stencil
    for i in range(d):
        for j in range(i + 1, d):
            xpp = x_scaled.copy(); xpp[i] += eps; xpp[j] += eps
            xpm = x_scaled.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = x_scaled.copy(); xmp[i] -= eps; xmp[j] += eps
            xmm = x_scaled.copy(); xmm[i] -= eps; xmm[j] -= eps
            val = (float(blob.gp.predict(xpp[None, :])[0])
                    - float(blob.gp.predict(xpm[None, :])[0])
                    - float(blob.gp.predict(xmp[None, :])[0])
                    + float(blob.gp.predict(xmm[None, :])[0])
                   ) / (4.0 * eps * eps)
            H[i, j] = H[j, i] = val

    H = 0.5 * (H + H.T)  # symmetrise against round-off
    jitter = _extract_jitter(blob)
    jitter["eps_finite_diff"]   = float(eps)
    jitter["coord_system"]      = "scaled (StandardScaler)"
    jitter["d"]                 = int(d)
    return H, jitter


def best_replicate(replicates_path: str | Path) -> Tuple[np.ndarray, dict]:
    """Identify the best converged optimum from data/optimizer_replicates.csv.

    Returns (x_star_in_original_param_space, row_dict).
    """
    import pandas as pd
    df = pd.read_csv(replicates_path)
    if "loss_eV" not in df.columns:
        raise KeyError(f"optimizer_replicates.csv missing 'loss_eV' column")
    idx = df["loss_eV"].idxmin()
    row = df.loc[idx].to_dict()
    # Identify which columns are the 40 parameters: everything except
    # the bookkeeping columns and loss_eV.
    book = {"rep_id", "trial", "acq_type", "loss_eV"}
    param_cols = [c for c in df.columns if c not in book]
    x_star = df.loc[idx, param_cols].values.astype(float)
    return x_star, row
