"""PES error function for the PIBO optimizer.

Computes the ``[Energy, Forces, Geometry]`` error vector of one ReaxFF
parameter candidate against the bond / angle / torsion / non-bonded DFT
scans, using LAMMPS for every frame:

    Energy RMSE   : category-weighted (manuscript Table 1 weights), in eV,
                    after removing the constant ReaxFF-vs-DFT energy offset.
    Forces RMSE   : over every atom of every frame that carries DFT forces
                    (np.nan when the single-point runner returns no forces).
    Geometry RMSE : over the lattice triplet (a, b, c) of every frame.

``variable_dict`` may be a ``{name: value}`` dict, or a flat sequence paired
with ``variable_names``. On failure the vector falls back to
``[nan, nan, nan]`` so the optimizer sees an infeasible candidate without
tearing the run down.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from pibo_reaxff.lammps_runner import LAMMPSRunner
from pibo_reaxff.vasp_reader import DFTFrame, load_dataset, _DIR_TO_CATEGORY
from pibo_reaxff.loss import MANUSCRIPT_CATEGORY_WEIGHTS
from optimizer_io import read_forcefield_template


# Avoid re-walking the DFT dataset on every optimizer iteration.
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


def error_function(variable_dict,
                   template_string: Optional[str] = None,
                   *,
                   variable_names: Optional[Sequence[str]] = None,
                   dataset_root: Optional[str] = None,
                   elements: Sequence[str] = ("Mo", "S", "H"),
                   base_ffield: Optional[str] = None,
                   subset: int = 0,
                   verbose: bool = False) -> List[float]:
    """Compute ``[E_err, F_err, geom_err]`` for one parameter candidate."""
    # 1) Resolve {name: value} pairs.
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

    # 2) Resolve the template.
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    if template_string is None:
        ffield_template = base_ffield or os.path.join(
            repo_root, "lammps_templates", "ffield.reax.MoSH.template")
        template_string = read_forcefield_template(ffield_template)

    # 3) Render the candidate ffield to a temp file the runner can load.
    tmp_dir = os.path.join(here, "output")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_ffield = os.path.join(tmp_dir, ".optimizer_rendered_ffield.reax.MoSH")
    from optimizer_io import generate_forcefield as _gen
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
            print(f"[optimizer_error] LAMMPS runner construction failed: {exc}")
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
                    print(f"[optimizer_error] frame {frame.tag!r} failed: {exc}")
                continue
            lammps_E.append(float(ev.energy))
            dft_E.append(float(frame.energy))
            cat_weights.append(MANUSCRIPT_CATEGORY_WEIGHTS.get(
                frame.category, 1.0))

            if ev.raw and all(k in ev.raw for k in ("lat_a", "lat_b", "lat_c")):
                lat_lammps.append([ev.raw["lat_a"], ev.raw["lat_b"],
                                   ev.raw["lat_c"]])
            else:
                lat_lammps.append(
                    list(np.linalg.norm(frame.cell, axis=1)))
            lat_dft.append(list(np.linalg.norm(frame.cell, axis=1)))

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
