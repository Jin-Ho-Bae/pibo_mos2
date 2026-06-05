"""
EZFF-shape ``error_function`` adapter.

Upstream EZFF's optimizer driver calls a user-supplied callable with
signature::

    errors = error_function(variable_dict, template_string) -> list[float]

where ``len(errors) == num_errors`` and each component is one objective
in the multi-objective optimisation. PIBO remains the optimizer in this
project (per user direction); this module exposes the same callable
shape so an EZFF-style optimiser *could* be plugged in later without
touching the runner/reader code.

The objective vector here is the 3-element ``[Energy, Forces, Geometry]``
chosen up-front:

    Energy RMSE :   weighted by ``DFTFrame.category`` (Manuscript Table 1
                    weights), in eV. Uses ``LAMMPSRunner.evaluate`` for
                    every frame, same as the BO loss.

    Forces RMSE :   sqrt(mean (F_LAMMPS - F_DFT)^2) over every atom and
                    every frame that carries a DFT force block. Frames
                    without forces are skipped silently (the OUTCARs in
                    ``vasp_calculations/`` aren't guaranteed to keep
                    forces). Currently the LAMMPS path does not return
                    per-atom forces (single-point evaluation produces
                    only energies + lattice in ``LAMMPSRunner``), so
                    this term is reported as ``np.nan`` unless the
                    runner is extended; the structure of the return
                    value is correct so a downstream caller can wire it.

    Geometry RMSE : sqrt(mean (a_LAMMPS - a_DFT)^2 + ...) over the
                    lattice triplet (a, b, c) for every frame, using
                    the ``lat_a/lat_b/lat_c`` values the existing
                    runner already parses. Falls back to per-frame
                    mean-cell-vector norm if the runner doesn't print
                    those (older builds).

A handful of usability concessions vs upstream EZFF:

    * ``variable_dict`` may be either a ``dict`` (EZFF standard) or any
      numpy array / list — when array-like, the caller must also pass
      ``variable_names`` (a list of names lined up to the positions).
    * The ``template_string`` argument is honoured (EZFF reads it once
      and passes it in on every call). If ``template_string is None``,
      we read ``lammps_templates/ffield.reax.MoSH.template`` from this
      repo.
    * Failure modes return ``[np.nan, np.nan, np.nan]`` rather than
      raising, so a multi-objective optimiser's bookkeeping (e.g.
      EZFF's NSGA2 driver) sees the candidate as infeasible without
      tearing the run down.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .lammps_runner import LAMMPSRunner
from .vasp_reader import DFTFrame, load_dataset, _DIR_TO_CATEGORY
from .loss import MANUSCRIPT_CATEGORY_WEIGHTS
from .ezff_io import read_forcefield_template


# ---------------------------------------------------------------------------
# Data caching: avoid re-walking vasp_calculations/ on every BO iteration.
# ---------------------------------------------------------------------------

_FRAME_CACHE: Dict[str, List[DFTFrame]] = {}


def load_frames(dataset_root: str,
                blocks: Optional[Sequence[str]] = None) -> List[DFTFrame]:
    """Flat list of every DFT frame under ``dataset_root``, cached."""
    key = f"{os.path.abspath(dataset_root)}|{','.join(blocks or [])}"
    if key in _FRAME_CACHE:
        return _FRAME_CACHE[key]
    by_block = load_dataset(dataset_root,
                            blocks=list(blocks) if blocks else None)
    flat: List[DFTFrame] = []
    for fs in by_block.values():
        flat.extend(fs)
    _FRAME_CACHE[key] = flat
    return flat


# ---------------------------------------------------------------------------
# Public: error_function(variable_dict, template_string) -> [E, F, geom]
# ---------------------------------------------------------------------------

def error_function(variable_dict,
                   template_string: Optional[str] = None,
                   *,
                   variable_names: Optional[Sequence[str]] = None,
                   dataset_root: Optional[str] = None,
                   elements: Sequence[str] = ("Mo", "S", "H"),
                   base_ffield: Optional[str] = None,
                   subset: int = 0,
                   verbose: bool = False) -> List[float]:
    """Compute ``[E_err, F_err, geom_err]`` for one parameter candidate.

    Parameters
    ----------
    variable_dict : dict | sequence
        EZFF passes a ``{name: value}`` dict; we also accept a flat
        sequence paired with ``variable_names`` for convenience.
    template_string : str, optional
        ffield template (with ``<<NAME>>`` markers). Defaults to
        ``lammps_templates/ffield.reax.MoSH.template`` in the repo.
    variable_names : sequence of str, optional
        Required iff ``variable_dict`` is a flat sequence.
    dataset_root : str, optional
        Root of the VASP dataset. Defaults to ``vasp_calculations``
        next to this file.
    elements : sequence of str
        Atom-type ordering for the LAMMPS data file
        (must match the template's atom block).
    base_ffield : str, optional
        Override the template path. Mutually exclusive with
        ``template_string``.
    subset : int
        If > 0, evaluate only the first ``subset`` frames (per-block,
        round-robin). Useful for fast sanity tests.
    """
    # 1) Resolve {variable_dict: value} pairs.
    if isinstance(variable_dict, dict):
        params: Dict[str, float] = {k: float(v)
                                    for k, v in variable_dict.items()}
    else:
        if variable_names is None:
            raise ValueError(
                "variable_dict is sequence-like; pass variable_names "
                "aligned to its order.")
        params = {name: float(val)
                  for name, val in zip(variable_names, variable_dict)}

    # 2) Resolve template.
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    if template_string is None:
        ffield_template = base_ffield or os.path.join(
            repo_root, "lammps_templates", "ffield.reax.MoSH.template")
        template_string = read_forcefield_template(ffield_template)

    # 3) Materialise the rendered ffield to a temp file LAMMPSRunner
    #    can load via its existing ``base_ffield`` argument. We don't
    #    overwrite the repo template — just write a sibling.
    tmp_ffield = os.path.join(
        repo_root, "lammps_templates",
        ".ezff_rendered_ffield.reax.MoSH")
    from .ezff_io import generate_forcefield as _gen
    _gen(template_string, params, FFtype="REAXFF", outfile=tmp_ffield,
         MD="LAMMPS")

    # 4) Load frames.
    if dataset_root is None:
        dataset_root = os.path.join(repo_root, "vasp_calculations")
    frames = load_frames(dataset_root)
    if subset > 0:
        frames = frames[:subset]

    # 5) Evaluate every frame with LAMMPS.
    try:
        runner = LAMMPSRunner(elements=tuple(elements),
                              base_ffield=tmp_ffield)
    except RuntimeError as exc:
        if verbose:
            print(f"[ezff_error] LAMMPS runner construction failed: {exc}")
        return [float("nan"), float("nan"), float("nan")]

    lammps_E: List[float] = []
    dft_E: List[float] = []
    cat_weights: List[float] = []
    lat_lammps: List[List[float]] = []
    lat_dft: List[List[float]] = []
    forces_lammps: List[np.ndarray] = []
    forces_dft: List[np.ndarray] = []

    try:
        for frame in frames:
            try:
                ev = runner.evaluate(params, frame)
            except Exception as exc:
                if verbose:
                    print(f"[ezff_error] frame {frame.tag!r} failed: {exc}")
                continue
            lammps_E.append(float(ev.energy))
            dft_E.append(float(frame.energy))
            cat_weights.append(MANUSCRIPT_CATEGORY_WEIGHTS.get(
                frame.category, 1.0))

            # Lattice triplet from EZFF/LAMMPS parsed output.
            if ev.raw and all(k in ev.raw for k in ("lat_a", "lat_b", "lat_c")):
                lat_lammps.append([ev.raw["lat_a"], ev.raw["lat_b"],
                                   ev.raw["lat_c"]])
            else:
                # Fallback: column norms of the DFT cell (zero-error
                # contribution rather than dropping the frame).
                lat_lammps.append(
                    list(np.linalg.norm(frame.cell, axis=1)))
            lat_dft.append(list(np.linalg.norm(frame.cell, axis=1)))

            # Forces are not currently returned by the single-point
            # LAMMPSRunner; reserve the slot.
            if frame.forces is not None and ev.forces is not None:
                forces_lammps.append(np.asarray(ev.forces))
                forces_dft.append(np.asarray(frame.forces))
    finally:
        runner.close()

    if not lammps_E:
        return [float("nan"), float("nan"), float("nan")]

    # 6) Compose the three objectives.
    p = np.asarray(lammps_E, float)
    r = np.asarray(dft_E, float)
    w = np.asarray(cat_weights, float)

    # Remove the constant absolute-energy offset (ReaxFF zero != DFT zero).
    p_shift = p - (p.mean() - r.mean())
    e_err = float(np.sqrt(np.sum(w * (p_shift - r) ** 2) / np.sum(w)))

    if forces_lammps:
        flat_p = np.concatenate([f.ravel() for f in forces_lammps])
        flat_r = np.concatenate([f.ravel() for f in forces_dft])
        f_err = float(np.sqrt(np.mean((flat_p - flat_r) ** 2)))
    else:
        f_err = float("nan")

    lat_p = np.asarray(lat_lammps, float)
    lat_r = np.asarray(lat_dft, float)
    geom_err = float(np.sqrt(np.mean((lat_p - lat_r) ** 2)))

    return [e_err, f_err, geom_err]
