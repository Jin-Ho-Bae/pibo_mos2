"""
ReaxFF parameter spec for the H/Mo/S system — Note S1 anchored.

Source of truth (in order of precedence):
  1. ``Note S1. Supplementary Information.pdf`` (this manuscript's
     uncertainty-aware ReaxFF tables — values used as the BO prior center).
  2. ``MoS2_reaxff.pdf`` (the already-reviewed Ostadhossein-2017 reference
     ffield — values used as the "reference" point against which the
     manuscript's diff is computed).
  3. ``scripts/notes1_param_diff.json`` (machine-readable diff list,
     written by ``scripts/build_notes1_template.py``).

The calibration set is the 42 Mo+S-relevant parameters that **differ**
between Note S1 and MoS2_reaxff (the dataset has no H atoms, so
H-related diffs are inactive and dropped):

    block      count
    -----      -----
    bond       18   (De(σ)×3, De(π)×2, De(ππ)×1, p(be1)×3, p(bo5)×3,
                     p(bo6)×3, p(ovun1)×3)
    offdiag     5   (Dij, RvdW, Alfa, ro(σ), ro(π) — Mo-S only)
    angle      16   (Thetao, p(val1), p(val2), p(val7) × 4 angles:
                     S-S-S, Mo-Mo-S, Mo-S-Mo, S-Mo-S)
    torsion     3   (V1, V2, V3 for S-S-S-S)

Bound construction
------------------
Per parameter we use ``center = Note S1 value`` and
``half-width = max(|notes1 - mos2_ref| * INTER_SRC_FACTOR,
                   |center| * MIN_HALF_FRAC, ABS_MIN_HALF)``,
clipped against per-family physical limits.

Rationale: the inter-PDF disagreement is a natural lower bound on the
parameter's uncertainty; ``INTER_SRC_FACTOR=5`` brackets ±5σ of that
disagreement. ``MIN_HALF_FRAC=0.05`` (5 % of |center|) keeps the bound
non-degenerate when both sources agree closely. ``ABS_MIN_HALF=0.02``
catches the case where |center| itself is near zero.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Per-family physical-admissibility limits (Ostadhossein 2017 ± 30 % per
# family, clipped against ReaxFF functional-form requirements).
# ---------------------------------------------------------------------------

PHYSICAL_LIMITS: Dict[str, Tuple[float, float]] = {
    "De_sigma":  (30.0, 200.0),       # kcal/mol
    "De_pi":     (10.0, 100.0),
    "De_pipi":   (80.0, 200.0),
    "p_be1":     (-1.5, 1.5),
    "p_bo5":     (-0.7, -0.05),       # must be negative
    "p_bo6":     (5.0, 40.0),         # must be positive
    "p_ovun1":   (0.05, 0.5),
    "Dij":       (0.02, 0.40),
    "RvdW":      (1.4, 2.5),
    "Alfa":      (8.0, 13.0),
    "ro_sigma":  (1.0, 3.0),
    "ro_pi":     (0.5, 2.5),
    "Thetao":    (20.0, 110.0),       # deg
    "p_val1":    (5.0, 60.0),
    "p_val2":    (0.3, 10.0),
    "p_val7":    (0.0, 5.0),
    "V1":        (-5.0, 5.0),
    "V2":        (10.0, 100.0),
    "V3":        (-1.0, 1.0),
    # Hydrogen-bond params — included because the user-requested staged
    # BO uses hbond as one of the 5 calibration blocks. Dataset is Mo+S
    # only so the hbond term doesn't actually contribute to the loss;
    # these bounds keep the BO from exploring obviously-unphysical
    # values.
    "r_hb":      (1.0, 2.5),       # H-bond cutoff radius (A)
    "p_hb1":     (-5.0, -0.5),     # H-bond energy prefactor
    "p_hb2":     (0.5, 3.0),
    "p_hb3":     (1.0, 5.0),
}

INTER_SRC_FACTOR: float = 5.0
MIN_HALF_FRAC:    float = 0.05
ABS_MIN_HALF:     float = 0.02


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParameterSpec:
    """One ReaxFF parameter that the BO is allowed to vary.

    Fields
    ------
    name : str
        Template-placeholder name. Must match ``<<{name}>>`` in
        ``ffield.reax.MoSH.NoteS1.template``.
    lo, hi : float
        Inclusive search bounds in physical units (eV / kcal mol⁻¹ /
        Å / deg / dimensionless depending on family).
    block : str
        ``"bond" | "offdiag" | "angle" | "torsion"``. Drives the
        staged-calibration block subsetting.
    notes1_value : float
        The manuscript supplementary value (Note S1 PDF). Used as the
        BO prior center; also the "manuscript anchor" reported in
        ``posterior_parameters_*.csv``.
    mos2_ref_value : float
        The reviewed reference value (MoS2_reaxff PDF). Used for the
        z-score diagnostic in posterior reports.
    physics_note : str
        Short text describing what the parameter controls physically.
    """
    name: str
    lo: float
    hi: float
    block: str
    notes1_value: float
    mos2_ref_value: float
    physics_note: str = ""

    # Aliases for compatibility with the existing posterior_report /
    # best_export modules that look at ``manuscript_mean`` and
    # ``manuscript_std``.
    @property
    def manuscript_mean(self) -> float:
        return self.notes1_value

    @property
    def manuscript_std(self) -> Optional[float]:
        # Use half the (Note S1 - MoS2_reaxff) gap as a rough "1σ"
        # estimate for the posterior-z diagnostic — proportional to the
        # bound width but small enough to flag real movement.
        gap = abs(self.notes1_value - self.mos2_ref_value)
        return max(gap, abs(self.notes1_value) * 0.01, 1e-3)

    def sample(self, rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng()
        return float(rng.uniform(self.lo, self.hi))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Strip the atom-pair / -triplet / -quartet suffix to get the parameter
# family for PHYSICAL_LIMITS lookup. E.g. "De_sigma_MoS" -> "De_sigma".
_FAMILY_PREFIXES = sorted(PHYSICAL_LIMITS.keys(), key=len, reverse=True)


def _family(name: str) -> str:
    for pref in _FAMILY_PREFIXES:
        if name == pref or name.startswith(pref + "_"):
            return pref
    raise ValueError(f"cannot infer parameter family from name {name!r}")


def _compute_bounds(notes1: float, mos2_ref: float,
                    family: str) -> Tuple[float, float]:
    gap = abs(notes1 - mos2_ref)
    half = max(gap * INTER_SRC_FACTOR,
               abs(notes1) * MIN_HALF_FRAC,
               ABS_MIN_HALF)
    lo, hi = notes1 - half, notes1 + half
    pmin, pmax = PHYSICAL_LIMITS[family]
    lo = max(lo, pmin)
    hi = min(hi, pmax)
    if hi <= lo:
        center = 0.5 * (pmin + pmax)
        lo, hi = max(pmin, center - 0.05), min(pmax, center + 0.05)
    return lo, hi


# ---------------------------------------------------------------------------
# Load the diff list and build REAXFF_PARAMETERS.
# ---------------------------------------------------------------------------

def _load_diff() -> List[Dict]:
    here = os.path.dirname(os.path.abspath(__file__))
    diff_path = os.path.normpath(os.path.join(here, "..", "scripts",
                                              "notes1_param_diff.json"))
    if not os.path.exists(diff_path):
        raise FileNotFoundError(
            f"{diff_path} not found. Run scripts/build_notes1_template.py "
            "first to regenerate the Note-S1 diff list.")
    with open(diff_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


_DIFF = _load_diff()

REAXFF_PARAMETERS: List[ParameterSpec] = []
for d in _DIFF:
    name = d["name"]
    family = _family(name)
    lo, hi = _compute_bounds(d["notes1"], d["mos2_ref"], family)
    REAXFF_PARAMETERS.append(ParameterSpec(
        name=name,
        lo=lo, hi=hi,
        block=d["block"],
        notes1_value=float(d["notes1"]),
        mos2_ref_value=float(d["mos2_ref"]),
        physics_note=f"{d['block']}; atoms={d['atoms']}; "
                     f"family={family}",
    ))

assert len(REAXFF_PARAMETERS) == len(_DIFF), \
    f"parameter count {len(REAXFF_PARAMETERS)} != diff count {len(_DIFF)}"

_BLOCK_COUNTS = {b: sum(1 for p in REAXFF_PARAMETERS if p.block == b)
                 for b in ("bond", "offdiag", "angle", "torsion", "hbond")}
assert _BLOCK_COUNTS == {"bond": 18, "offdiag": 5, "angle": 16,
                         "torsion": 3, "hbond": 4}, \
    f"block counts changed: {_BLOCK_COUNTS}"


# ---------------------------------------------------------------------------
# Block-alias map kept for backward compatibility with code that filters
# on the legacy labels (``vdw_coulomb``, ``nonbond``, ``vdw``, ``coulomb``).
# ``offdiag`` is the canonical Note-S1 label; the aliases all map to it.
# ---------------------------------------------------------------------------

_BLOCK_ALIASES: Dict[str, Tuple[str, ...]] = {
    "vdw_coulomb": ("offdiag",),
    "nonbond":     ("offdiag",),
    "vdw":         ("offdiag",),
    "coulomb":     (),                 # chi/eta dropped from calibration
}


# ---------------------------------------------------------------------------
# Public helpers — signatures preserved for existing callers.
# ---------------------------------------------------------------------------

def parameters_for_blocks(blocks: Iterable[str]) -> List[ParameterSpec]:
    """Return parameters whose ``.block`` matches any of ``blocks``."""
    expanded: set[str] = set()
    for b in blocks:
        if b in _BLOCK_ALIASES:
            expanded.update(_BLOCK_ALIASES[b])
        else:
            expanded.add(b)
    return [p for p in REAXFF_PARAMETERS if p.block in expanded]


def bounds_array(specs: List[ParameterSpec]) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.array([p.lo for p in specs], dtype=float)
    hi = np.array([p.hi for p in specs], dtype=float)
    return lo, hi


def normalize(x: np.ndarray, specs: List[ParameterSpec]) -> np.ndarray:
    lo, hi = bounds_array(specs)
    return (np.asarray(x) - lo) / (hi - lo)


def denormalize(u: np.ndarray, specs: List[ParameterSpec]) -> np.ndarray:
    lo, hi = bounds_array(specs)
    return lo + np.asarray(u) * (hi - lo)


def names(specs: List[ParameterSpec]) -> List[str]:
    return [p.name for p in specs]


def latin_hypercube(n: int, specs: List[ParameterSpec],
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """LHS in physical space (pyDOE if available, NumPy fallback otherwise)."""
    rng = rng or np.random.default_rng()
    d = len(specs)
    try:
        from pyDOE import lhs
        u = lhs(d, samples=n, criterion="maximin",
                random_state=rng.integers(2**31 - 1))
    except Exception:
        u = (rng.random((n, d)) + rng.permutation(n)[:, None]) / n
    return denormalize(u, specs)


def manuscript_anchored(specs: List[ParameterSpec]) -> List[ParameterSpec]:
    """All 42 are Note-S1-anchored; kept for API symmetry."""
    return list(specs)


def notes1_values(specs: List[ParameterSpec]) -> Dict[str, float]:
    """``{name: notes1_value}`` for every spec — usable as a starting
    point or as the "manuscript-anchored" parameter dict for diagnostics."""
    return {p.name: float(p.notes1_value) for p in specs}


def mos2_ref_values(specs: List[ParameterSpec]) -> Dict[str, float]:
    """``{name: mos2_ref_value}`` — usable for an Ostadhossein-2017 baseline
    evaluation against this codebase's loss."""
    return {p.name: float(p.mos2_ref_value) for p in specs}


def default_general_params() -> Dict[str, float]:
    """ReaxFF general parameters that the template carries as constants.

    Kept for compatibility with ``lammps_runner.LAMMPSRunner.__init__``;
    the template itself already holds the Note-S1 numerical values for
    every general / atom / bond / off-diag / angle / torsion entry that
    isn't a ``<<placeholder>>``, so this dict is informational only.
    """
    return {}
