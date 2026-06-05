"""
Export the best BO parameter set as a LAMMPS-format ffield.

The export path goes through the *same* substitution function the runner
uses during BO (``lammps_runner.generate_forcefield``, which is what
``ezff_io.generate_forcefield`` delegates to). Identity to existing
pibo_reaxff outputs is therefore a property of the substitution, not of
this module — see ``scripts/check_ezff_identity.py`` for the
byte-for-byte check.

Public API
----------
``find_best_record(suite, *, optimizer=None, physics_informed=True)``
    Pick the lowest-``best_loss`` ``RunRecord`` from a BenchmarkSuite. By
    default scans only ``physics_informed=True`` records (the manuscript
    column); set ``physics_informed=None`` to scan every record.

``best_params_dict(suite, record)``
    Zip ``record.best_x`` with the suite's parameter names so the result
    is a ``{name: value}`` dict shaped exactly like the runner expects.

``export_lammps_ffield(params, template_path, out_path)``
    Render and write the LAMMPS ffield. Returns the absolute output
    path. Raises if any ``<<NAME>>`` placeholder survives substitution.

``write_best_summary(record, names, out_path)``
    JSON sidecar with metrics + final parameters. Useful for
    reproducibility (the JSON can be fed back into
    ``--ezff-params <path>`` for a re-render).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .ezff_io import generate_forcefield, read_forcefield_template


def find_best_record(suite,
                     *,
                     optimizer: Optional[str] = None,
                     physics_informed: Optional[bool] = True):
    """Lowest-``best_loss`` record matching the filters.

    Filters
    -------
    optimizer
        e.g. ``"pibo"``. ``None`` accepts any.
    physics_informed
        ``True`` (default) restricts to physics-informed runs — those
        are the manuscript-reportable rows. ``False`` restricts to the
        ablation runs. ``None`` accepts both.
    """
    candidates = [r for r in getattr(suite, "results", [])
                  if (optimizer is None or r.optimizer == optimizer)
                  and (physics_informed is None
                       or r.physics_informed == physics_informed)
                  and r.best_x is not None
                  and len(r.best_x) > 0
                  and np.isfinite(r.best_loss)]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r.best_loss)


def parameter_names(suite) -> List[str]:
    """``[spec.name]`` in the same order BO produced ``best_x``."""
    specs = getattr(suite, "specs", None)
    if not specs:
        raise RuntimeError(
            "BenchmarkSuite.specs is empty; parameter names are unknown.")
    return [s.name for s in specs]


def best_params_dict(suite, record) -> Dict[str, float]:
    names = parameter_names(suite)
    if len(names) != len(record.best_x):
        raise RuntimeError(
            f"record.best_x length {len(record.best_x)} != "
            f"len(suite.specs) {len(names)} — cannot align names.")
    return {n: float(v) for n, v in zip(names, record.best_x)}


def export_lammps_ffield(params: Dict[str, float],
                         template_path: str,
                         out_path: str) -> str:
    """Render ``template_path`` with ``params`` and write to ``out_path``.

    Mirrors ``LAMMPSRunner.write_ffield`` (same width-preserving
    substitution via ``ezff_io.generate_forcefield``) so the rendered
    file is byte-for-byte what the BO runner would have written for
    those parameters.
    """
    template = read_forcefield_template(template_path)
    rendered = generate_forcefield(
        template, params, FFtype="REAXFF", MD="LAMMPS")
    leftover = re.findall(r"<<\w+>>", rendered)
    if leftover:
        raise RuntimeError(
            f"export_lammps_ffield: {len(leftover)} placeholders survived "
            f"substitution; first few: {leftover[:5]}. "
            "Either params is missing keys or the template has placeholders "
            "outside the BO parameter set.")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(rendered)
    return out_path


def write_best_summary(record, names: List[str], out_path: str,
                       extra: Optional[Dict[str, Any]] = None) -> str:
    """JSON sidecar: BO metrics + final {name: value} parameters.

    This file is consumable by ``--ezff-params <path>`` for a verbatim
    re-render (only the parameters section is read by that flag; the
    metrics travel along for provenance).
    """
    params = {n: float(v) for n, v in zip(names, record.best_x)}
    payload: Dict[str, Any] = {
        "metrics": {
            "optimizer":              record.optimizer,
            "replicate":              record.replicate,
            "physics_informed":       bool(record.physics_informed),
            "best_loss":              float(record.best_loss),
            "loss_rmse":              float(record.loss_rmse),
            "loss_mae":               float(record.loss_mae),
            "E_total_error":          float(record.E_total_error),
            "val_rmse":               float(record.val_rmse),
            "n_evals":                int(record.n_evals),
            "fevals_to_convergence":  int(record.fevals_to_convergence),
            "wall_clock_s":           float(record.wall_clock_s),
        },
    }
    # Top-level keys mirror the {name: value} payload accepted by
    # --ezff-params <path>, so this JSON is both a summary AND a valid
    # re-render input.
    payload.update(params)
    if extra:
        payload["_extra"] = extra
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return out_path


# ---------------------------------------------------------------------------
# Convergence detection helpers (used by run.py's retry loop)
# ---------------------------------------------------------------------------

def has_nonfinite_result(suite) -> Tuple[bool, str]:
    """Return ``(True, reason)`` if any RunRecord has a non-finite metric.

    Triggers for: best_loss inf/NaN, val_rmse inf/NaN, OR a validation
    log row with worst==inf (stored in ``record.extras['validation_log']``).
    """
    for r in getattr(suite, "results", []):
        for field_name in ("best_loss", "loss_rmse", "val_rmse",
                           "E_total_error"):
            v = getattr(r, field_name, None)
            if v is not None and not np.isfinite(v):
                return True, (
                    f"{r.optimizer} rep={r.replicate} phys={r.physics_informed}: "
                    f"{field_name}={v}")
        vlog = (r.extras or {}).get("validation_log") if r.extras else None
        if vlog:
            for entry in vlog:
                worst = entry.get("worst")
                if worst is not None and not np.isfinite(worst):
                    return True, (
                        f"{r.optimizer} rep={r.replicate}: "
                        f"validation gate worst={worst} at "
                        f"step={entry.get('step', '?')}")
    return False, ""
