"""Phase 2 & 3 — Sethna sloppy spectrum + profile likelihood.

Phase 2 surface:
* ``hessian_at`` — wraps ``gp_utils.loss_hessian`` with a jitter strategy.
* ``sloppy_spectrum`` — eigendecomposition + bookkeeping.
* ``sloppiness_report`` — Markdown summary.

Phase 3 surface:
* ``profile_likelihood`` — 1-D profile of the GP mean (and predictive
  std) along a unit eigendirection, with parameter-bound projection
  and per-point warning flags.

All Hessian-related work lives in the GP's **scaled** parameter space
(StandardScaler). To get back to original ReaxFF units rotate by
``v_orig = v_scaled / scaler.scale_``.

No silent fallbacks: malformed jitter_strategy raises immediately.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .gp_utils import GPBlob, loss_hessian, predict_loss


# ---------------------------------------------------------------------------
# Jitter strategies
# ---------------------------------------------------------------------------

VALID_STRATEGIES = ("keep", "subtract", "report_floor")


def _kernel_noise_level(blob: GPBlob) -> float:
    """Scrape the WhiteKernel's ``noise_level`` from the trained kernel.

    Returns 0.0 if no WhiteKernel is present. Uses sklearn's nested
    ``get_params(deep=True)`` so it works for any compound kernel.
    """
    noise = 0.0
    for name, val in blob.gp.kernel_.get_params(deep=True).items():
        if name.endswith("__noise_level") and isinstance(val, (int, float)):
            # multiple WhiteKernels (rare) → keep the largest as a
            # conservative floor; this never under-estimates jitter.
            noise = max(noise, float(val))
    return noise


def hessian_at(blob: GPBlob, x_star: np.ndarray,
               jitter_strategy: str = "report_floor",
               eps: float = 1e-3,
               ) -> tuple[np.ndarray, dict]:
    """Compute the loss-Hessian of the GP at x_star and apply the chosen
    jitter strategy. Returns ``(H, info)``.

    Strategies
    ----------
    ``keep``         : H as returned by the finite-difference routine
                       (jitter is implicitly present in the kernel's
                       smoothness floor).
    ``subtract``     : ``H_returned = H - λ_jitter · I`` where λ_jitter
                       is the WhiteKernel noise level scrubbed from the
                       trained kernel. This is the "best estimate of
                       the true model curvature" interpretation; it
                       *can* turn small positive eigenvalues negative.
    ``report_floor`` : H is returned unchanged but ``info["jitter_floor"]``
                       carries the value below which the spectrum is
                       not numerically resolved.

    ``info`` always contains:
        jitter_floor          (scaled units, ~ noise_level)
        kernel_noise_level    (raw WhiteKernel value)
        eps_finite_diff
        strategy
        coord_system
    """
    if jitter_strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"jitter_strategy must be one of {VALID_STRATEGIES}, "
            f"got {jitter_strategy!r}")

    H_raw, fd_info = loss_hessian(blob, x_star, eps=eps)
    noise = _kernel_noise_level(blob)

    # In scaled space the GP is normalize_y=True so noise_level is dimensionless
    # variance of the (normalized) target. The Hessian computed from f(x_scaled)
    # has the same unit-system, so subtracting `noise` from the diagonal is the
    # correct (if conservative) jitter removal.
    jitter_floor = float(noise)

    if jitter_strategy == "keep":
        H = H_raw
    elif jitter_strategy == "subtract":
        H = H_raw - jitter_floor * np.eye(H_raw.shape[0])
    elif jitter_strategy == "report_floor":
        H = H_raw
    else:  # pragma: no cover (covered by VALID_STRATEGIES check above)
        raise AssertionError("unreachable")

    info = {
        "jitter_floor":       jitter_floor,
        "kernel_noise_level": noise,
        "eps_finite_diff":    float(eps),
        "strategy":           jitter_strategy,
        "coord_system":       "scaled (StandardScaler)",
        "y_std_eV":           fd_info.get("y_std_eV"),
        "alpha_normalised":   fd_info.get("alpha_normalised"),
    }
    return H, info


# ---------------------------------------------------------------------------
# Eigendecomposition + report
# ---------------------------------------------------------------------------

def sloppy_spectrum(H: np.ndarray, info: dict) -> dict:
    """Compute the sloppy spectrum of a symmetric Hessian.

    Returns a dict with keys
        eigenvalues         (40,)       descending by |λ|
        eigenvectors        (40, 40)    columns = eigenvectors in scaled coords
        signs               (40,)       sign of each eigenvalue
        abs_eigenvalues     (40,)       |λ|, descending
        log10_abs           (40,)       log10|λ| (with floor at 1e-30)
        span_orders         scalar      log10(max|λ| / min|λ|)
        jitter_floor        scalar      from info
        effective_rank      int         count of |λ| > 10·jitter_floor
        strategy            str
        n_negative          int
        n_negative_top10    int         (|λ|-rank ≤ 10)
    """
    H = 0.5 * (H + H.T)  # robust symmetrisation
    w, V = np.linalg.eigh(H)
    order = np.argsort(np.abs(w))[::-1]
    w = w[order]
    V = V[:, order]
    abs_w = np.abs(w)
    floor = float(info.get("jitter_floor", 0.0))

    eff_rank = int(np.sum(abs_w > 10.0 * max(floor, 1e-30)))
    span_orders = float(np.log10(abs_w.max() / max(abs_w.min(), 1e-30)))

    return {
        "eigenvalues":      w,
        "eigenvectors":     V,
        "signs":            np.sign(w),
        "abs_eigenvalues":  abs_w,
        "log10_abs":        np.log10(np.clip(abs_w, 1e-30, None)),
        "span_orders":      span_orders,
        "jitter_floor":     floor,
        "effective_rank":   eff_rank,
        "strategy":         str(info.get("strategy", "?")),
        "n_negative":       int(np.sum(w < 0)),
        "n_negative_top10": int(np.sum(w[:10] < 0)),
        "eps_finite_diff":  float(info.get("eps_finite_diff", float("nan"))),
    }


def sloppiness_report(spectrum: dict, param_names: list[str] | None = None,
                       k_top: int = 5) -> str:
    """Format a Markdown-flavoured sloppiness report for the diagnostic log."""
    w   = spectrum["eigenvalues"]
    V   = spectrum["eigenvectors"]
    abs_w = spectrum["abs_eigenvalues"]
    floor = spectrum["jitter_floor"]
    eff = spectrum["effective_rank"]
    span = spectrum["span_orders"]
    n_neg = spectrum["n_negative"]
    n_neg10 = spectrum["n_negative_top10"]

    lines: list[str] = []
    lines.append(f"## Sloppiness report — strategy = {spectrum['strategy']!r}")
    lines.append(f"")
    lines.append(f"- d                  : **{w.size}**")
    lines.append(f"- |λ| max            : **{abs_w[0]:.3e}**")
    lines.append(f"- |λ| min            : **{abs_w[-1]:.3e}**")
    lines.append(f"- span (decades)     : **{span:.2f}**")
    lines.append(f"- jitter floor       : **{floor:.3e}**")
    lines.append(f"- effective rank     : **{eff}**  (|λ| > 10·floor)")
    lines.append(f"- n_stiff (|λ|>1e-2) : **{int(np.sum(abs_w > 1e-2))}**")
    lines.append(f"- n_sloppy (|λ|<1e-4): **{int(np.sum(abs_w < 1e-4))}**")
    lines.append(f"- n_negative (total) : **{n_neg}**")
    lines.append(f"- n_negative top-10  : **{n_neg10}**")
    if n_neg > 0:
        max_neg_abs = float(np.max(np.abs(w[w < 0]))) if n_neg else 0.0
        lines.append(f"- max |λ_negative|   : **{max_neg_abs:.3e}**")
    lines.append(f"- eps (finite diff)  : {spectrum['eps_finite_diff']:.3e}")
    lines.append("")

    # Negative-eigenvalue audit — top-3 contributing parameter loadings
    if param_names is not None and n_neg > 0:
        lines.append("### Negative-eigenvalue audit (top-3 contributing params)")
        lines.append("")
        lines.append("| rank | λ | top-3 |v_i| · param |")
        lines.append("|-----:|---:|------------------------|")
        for rank, lam in enumerate(w, start=1):
            if lam < 0:
                v = V[:, rank - 1]
                # top-3 by |v_i|
                idx3 = np.argsort(np.abs(v))[::-1][:3]
                triples = [f"{abs(v[i]):.3f}·{param_names[i]}" for i in idx3]
                lines.append(f"| {rank} | {lam:.3e} | "
                              + ", ".join(triples) + " |")
        lines.append("")

    # Top-k stiff and top-k sloppy directions with their dominant parameters
    if param_names is not None:
        lines.append(f"### Top-{k_top} stiff directions")
        lines.append("")
        lines.append("| rank | λ | top-3 |v_i| · param |")
        lines.append("|-----:|---:|------------------------|")
        for i in range(min(k_top, w.size)):
            v = V[:, i]
            idx3 = np.argsort(np.abs(v))[::-1][:3]
            triples = [f"{abs(v[j]):.3f}·{param_names[j]}" for j in idx3]
            lines.append(f"| {i+1} | {w[i]:.3e} | " + ", ".join(triples) + " |")
        lines.append("")

        lines.append(f"### Top-{k_top} sloppy directions")
        lines.append("")
        lines.append("| rank | λ | top-3 |v_i| · param |")
        lines.append("|-----:|---:|------------------------|")
        for i in range(w.size - k_top, w.size):
            v = V[:, i]
            idx3 = np.argsort(np.abs(v))[::-1][:3]
            triples = [f"{abs(v[j]):.3f}·{param_names[j]}" for j in idx3]
            lines.append(f"| {i+1} | {w[i]:.3e} | " + ", ".join(triples) + " |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 3 — Profile likelihood
# ---------------------------------------------------------------------------

def profile_likelihood(blob: GPBlob,
                        x_star: np.ndarray,
                        direction: np.ndarray,
                        eigenvalue: float,
                        sigma_range: tuple[float, float] = (-3.0, 3.0),
                        n_pts: int = 61,
                        bounds: "dict[str, tuple[float, float]] | None" = None,
                        prior_scale_threshold: float = 1e-3,
                        projection_warn_pct: float = 5.0,
                        ) -> dict:
    """1-D profile of the GP-predicted loss along a unit eigendirection.

    Parameters
    ----------
    blob          : GP surrogate bundle.
    x_star        : (d,) reference point in **original** parameter space.
    direction     : (d,) unit eigenvector in **scaled** coordinates.
    eigenvalue    : λ associated with `direction` (in the same units as
                    the scaled-space Hessian). Used to set the per-σ
                    displacement scale.
    sigma_range   : (lo, hi) range in σ-units along the eigendirection.
    n_pts         : number of evaluation points.
    bounds        : optional ``{param_name: (lo, hi)}`` in original space.
                    When supplied the proposed point is clipped to the
                    bound box and the projection distance is reported.
    prior_scale_threshold
                  : if |λ| ≤ this value the direction is treated as
                    "sloppy at floor" and the displacement scale falls
                    back to **1 BO-cloud σ** (= 1.0 in scaled coords),
                    avoiding numerically meaningless 1/√λ→∞ excursions.
    projection_warn_pct
                  : a point is flagged when its projection distance is
                    larger than `projection_warn_pct` % of the bound-box
                    diagonal length.

    Returns
    -------
    dict with
        displacements_sigma     (n_pts,)
        scale_unit              float        the σ unit used
        scale_source            'curvature' | 'prior_fallback'
        x_orig_grid             (n_pts, d)   proposed (pre-projection) points
        x_projected_grid        (n_pts, d)   bound-clipped points actually queried
        mean_loss               (n_pts,)
        std_loss                (n_pts,)     GP epistemic uncertainty
        projection_distance     (n_pts,)     L2 distance proposed → projected
        projection_pct          (n_pts,)     distance / bound-box diagonal × 100
        projection_warnings     (n_pts,) bool
        any_warning             bool
        max_projection_pct      float
    """
    x_star = np.asarray(x_star, dtype=float).ravel()
    direction = np.asarray(direction, dtype=float).ravel()
    # Normalise direction in case the caller passed a non-unit vector
    nrm = np.linalg.norm(direction)
    if nrm < 1e-12:
        raise ValueError("profile_likelihood: zero-norm direction")
    direction = direction / nrm

    abs_lam = abs(float(eigenvalue))
    if abs_lam <= prior_scale_threshold or eigenvalue <= 0.0:
        scale_unit = 1.0
        scale_source = "prior_fallback"
    else:
        scale_unit = 1.0 / np.sqrt(abs_lam)
        scale_source = "curvature"

    x_star_scaled = blob.scaler.transform(x_star[None, :]).ravel()
    displacements = np.linspace(sigma_range[0], sigma_range[1], n_pts)

    if bounds is not None:
        lo = np.array([bounds[n][0] for n in blob.param_names], dtype=float)
        hi = np.array([bounds[n][1] for n in blob.param_names], dtype=float)
        bound_box_diag = float(np.linalg.norm(hi - lo))
    else:
        lo = hi = None
        bound_box_diag = float("nan")

    x_orig_grid     = np.zeros((n_pts, x_star.size))
    x_proj_grid     = np.zeros((n_pts, x_star.size))
    mean_loss       = np.zeros(n_pts)
    std_loss        = np.zeros(n_pts)
    proj_dist       = np.zeros(n_pts)
    proj_pct        = np.zeros(n_pts)
    warnings        = np.zeros(n_pts, dtype=bool)

    for i, d in enumerate(displacements):
        x_new_scaled = x_star_scaled + d * scale_unit * direction
        x_new_orig = blob.scaler.inverse_transform(x_new_scaled[None, :]).ravel()
        x_orig_grid[i] = x_new_orig
        if bounds is not None:
            x_proj = np.minimum(np.maximum(x_new_orig, lo), hi)
            proj_dist[i] = float(np.linalg.norm(x_new_orig - x_proj))
            proj_pct[i]  = proj_dist[i] / max(bound_box_diag, 1e-12) * 100.0
            warnings[i]  = proj_pct[i] > projection_warn_pct
            x_query = x_proj
        else:
            x_query = x_new_orig
        x_proj_grid[i] = x_query
        m, s = predict_loss(blob, x_query)
        mean_loss[i] = float(np.asarray(m).ravel()[0])
        std_loss[i]  = float(np.asarray(s).ravel()[0])

    return {
        "displacements_sigma":   displacements,
        "scale_unit":            float(scale_unit),
        "scale_source":          scale_source,
        "eigenvalue":            float(eigenvalue),
        "abs_eigenvalue":        abs_lam,
        "x_orig_grid":           x_orig_grid,
        "x_projected_grid":      x_proj_grid,
        "mean_loss":             mean_loss,
        "std_loss":              std_loss,
        "projection_distance":   proj_dist,
        "projection_pct":        proj_pct,
        "projection_warnings":   warnings,
        "any_warning":           bool(warnings.any()),
        "max_projection_pct":    float(proj_pct.max())
                                 if not np.isnan(proj_pct).all() else float("nan"),
        "loss_span_eV":          float(mean_loss.max() - mean_loss.min()),
    }
