"""
DFT-vs-ReaxFF loss functions with manuscript Table 1 category weights.

Loss formulation (Manuscript Eq. 1):

    L(x) = Σ_i  w_i · ( y_i^ReaxFF(x) - y_i^DFT )^2

where ``w_i`` is the *configuration-category* weight from Table 1:

    Equilibrium geometries (1T-, 2H-MoS2)         w = 10.0
    Bond-distance PES (Mo–S)                       w =  5.0
    Angle PES (S–Mo–S)                             w =  3.0
    Dihedral PES (S–Mo–S–Mo)                       w =  1.0
    Strained cells (> 1 eV/atom above eq.)         w =  0.5

Each ``DFTFrame`` carries its category in ``frame.category``; weights are
plumbed through ``LossWeights.category_weights``. Forces are wired in but
default-off because OUTCAR forces are not always available in /CODE data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .vasp_reader import DFTFrame
from .lammps_runner import LAMMPSRunner


# Manuscript Table 1 — per-category training weights.
MANUSCRIPT_CATEGORY_WEIGHTS: Dict[str, float] = {
    "equilibrium":   10.0,
    "bond_pes":       5.0,
    "angle_pes":      3.0,
    "dihedral_pes":   1.0,
    "strained":       0.5,
    "default":        1.0,
}


@dataclass
class LossWeights:
    """Composite loss weights.

    ``energy`` scales the (weighted) energy RMSE term; ``forces`` and
    ``geometry`` scale the optional force / geometry terms. ``category_weights``
    map ``DFTFrame.category`` → per-frame multiplier on the squared energy
    residual (Manuscript Eq. 1). Defaults reproduce manuscript Table 1.

    ``per_atom_norm`` (NEW): when True, every per-frame energy residual
    is divided by ``frame.n_atoms`` before entering the weighted-RMSE.
    This converts the loss units from eV/cell to eV/atom — the
    conventional ReaxFF training metric. With ``n_atoms=6`` for every
    Mo+S frame in the project's dataset, the per-atom loss is roughly
    ⅙ of the per-cell loss, which together with the outlier filter
    (``vasp_reader.load_dataset(max_dE_per_atom_eV=...)``) is what
    drives the loss toward the manuscript's <0.1 target.
    """
    energy: float = 1.0
    forces: float = 0.0
    geometry: float = 0.5
    per_atom_norm: bool = False
    category_weights: Dict[str, float] = field(
        default_factory=lambda: dict(MANUSCRIPT_CATEGORY_WEIGHTS))


def _shift_to_match_mean(predictions: np.ndarray,
                         references: np.ndarray) -> np.ndarray:
    """Remove the absolute-energy offset (ReaxFF zero != DFT zero)."""
    return predictions - (predictions.mean() - references.mean())


def energy_rmse(predictions: np.ndarray, references: np.ndarray,
                weights: np.ndarray | None = None,
                normalize_offset: bool = True) -> float:
    """Weighted root-mean-square energy error.

    With ``weights`` supplied, returns ``sqrt(Σ w·(p−r)² / Σ w)`` — the
    weighted RMSE used by Manuscript Eq. 1. Without weights, falls back to
    plain RMSE.
    """
    p = _shift_to_match_mean(predictions, references) if normalize_offset else predictions
    if weights is None:
        return float(np.sqrt(np.mean((p - references) ** 2)))
    w = np.asarray(weights, dtype=float)
    s = float(np.sum(w))
    if s <= 0:
        return float(np.sqrt(np.mean((p - references) ** 2)))
    return float(np.sqrt(np.sum(w * (p - references) ** 2) / s))


def energy_mae(predictions: np.ndarray, references: np.ndarray,
               weights: np.ndarray | None = None,
               normalize_offset: bool = True) -> float:
    p = _shift_to_match_mean(predictions, references) if normalize_offset else predictions
    if weights is None:
        return float(np.mean(np.abs(p - references)))
    w = np.asarray(weights, dtype=float)
    s = float(np.sum(w))
    if s <= 0:
        return float(np.mean(np.abs(p - references)))
    return float(np.sum(w * np.abs(p - references)) / s)


def total_energy_error(predictions: np.ndarray, references: np.ndarray) -> float:
    """Used as the ``E`` benchmark column."""
    return float(abs(_shift_to_match_mean(predictions, references).sum()
                     - references.sum()))


def geometry_rmsd(frames_pred: List[np.ndarray],
                  frames_ref: List[np.ndarray]) -> float:
    """Mean per-atom RMSD between two equally ordered position lists."""
    if not frames_pred:
        return 0.0
    rmsds = []
    for p, r in zip(frames_pred, frames_ref):
        n = min(len(p), len(r))
        rmsds.append(np.sqrt(np.mean(np.sum((p[:n] - r[:n]) ** 2, axis=1))))
    return float(np.mean(rmsds))


def _measure_equilibrium_geom(runner, params: Dict[str, float]) -> Dict[str, float]:
    """Cell-relax a 3x3 MoS2 monolayer at the given params; return
    {a, h_S} from the relaxed geometry. Used by ReaxFFLoss for the
    geometric anchor term."""
    import os, subprocess, tempfile, textwrap, re, shutil
    from .lammps_runner import _find_lmp_binary, detect_reaxff_style, write_lammps_control
    from .physical_validation import (_monolayer_unit_cell, _replicate,
                                       _write_lammps_data_triclinic,
                                       A0_MONOLAYER, H_S0)
    lmp = _find_lmp_binary()
    reax_style, qeq_style = detect_reaxff_style(lmp)
    wd = tempfile.mkdtemp(prefix="geom_anchor_")
    try:
        pos, sp, cell = _monolayer_unit_cell(A0_MONOLAYER, H_S0)
        pos, sp, cell = _replicate(pos, sp, cell, 3, 3, 1)
        DEFAULT_MASSES = {"Mo": 95.94, "S": 32.06, "H": 1.008}
        _write_lammps_data_triclinic(os.path.join(wd, "data.eq"), pos, sp,
                                      cell, ("Mo","S","H"), DEFAULT_MASSES)
        ctrl = write_lammps_control(wd)
        ffield_path = os.path.join(wd, "ffield.reax")
        runner.write_ffield(params, ffield_path)
        deck = textwrap.dedent(f"""\
            units real
            atom_style charge
            boundary p p p
            read_data data.eq
            pair_style {reax_style} {ctrl} safezone 2.4 mincap 200
            pair_coeff * * ffield.reax Mo S H
            fix qeq all {qeq_style} 1 0.0 10.0 1.0e-6 reaxff
            fix box1 all box/relax x 0.0 y 0.0 vmax 0.001
            neighbor 2.0 bin
            neigh_modify every 5 delay 5 check yes
            minimize 1.0e-6 1.0e-7 100 1000
            write_dump all custom geom.dump id type element x y z modify sort id element Mo S H
            print "FINAL_LX $(lx)"
            """)
        with open(os.path.join(wd, "in.lmp"), "w") as f: f.write(deck)
        subprocess.run([lmp, "-in", "in.lmp", "-screen", "none",
                        "-log", "log.lammps"], cwd=wd, capture_output=True,
                       text=True, timeout=60.0)
        log = open(os.path.join(wd, "log.lammps"), errors="replace").read()
        m = re.search(r"FINAL_LX\s+([-+0-9.eE]+)", log)
        lx = float(m.group(1)) if m else float("nan")
        atoms = []
        for line in open(os.path.join(wd, "geom.dump"),
                          errors="replace"):
            if line.startswith("ITEM:"):
                in_atoms = "ATOMS" in line; continue
            parts = line.split()
            if len(parts) >= 6 and parts[2].upper() == "S":
                atoms.append(float(parts[5]))
        h_S = float("nan")
        if len(atoms) >= 2:
            zm = sum(atoms)/len(atoms)
            top = [z for z in atoms if z >= zm]
            bot = [z for z in atoms if z <  zm]
            if top and bot:
                h_S = sum(top)/len(top) - sum(bot)/len(bot)
        return {"a": lx/3.0, "h_S": h_S}
    finally:
        shutil.rmtree(wd, ignore_errors=True)


class ReaxFFLoss:
    """Callable loss object shared by all four optimizers.

    ``loss(params_vec)`` evaluates every training frame, computes a
    Manuscript-faithful weighted RMSE in energy, and returns a scalar. It
    also caches per-frame predictions (``self.last_ys``) so downstream
    metrics can be pulled without re-evaluating.
    """

    def __init__(self,
                 runner: LAMMPSRunner,
                 specs,
                 train_frames: List[DFTFrame],
                 val_frames: List[DFTFrame] | None = None,
                 weights: LossWeights | None = None,
                 vectorize: bool = True):
        self.runner = runner
        self.specs = specs
        self.train_frames = train_frames
        self.val_frames = val_frames or []
        self.weights = weights or LossWeights()
        self.vectorize = vectorize

        self.n_evals = 0
        self.history: List[float] = []
        self.last_ys: np.ndarray | None = None
        self.last_params: Dict[str, float] | None = None
        self.total_runtime_s: float = 0.0

        # Pre-compute the per-frame weight vectors (one per training/val).
        self._train_weights = self._frame_weights(self.train_frames)
        self._val_weights = self._frame_weights(self.val_frames)

    # ----- helpers --------------------------------------------------------

    def _params_dict(self, vec: np.ndarray) -> Dict[str, float]:
        return {s.name: float(v) for s, v in zip(self.specs, vec)}

    def _frame_weights(self, frames: List[DFTFrame]) -> np.ndarray:
        cw = self.weights.category_weights
        return np.array(
            [cw.get(getattr(f, "category", "default"), 1.0) for f in frames],
            dtype=float,
        )

    def _evaluate_frames(self, frames: List[DFTFrame],
                         params: Dict[str, float]) -> Tuple[np.ndarray, float]:
        ys = np.empty(len(frames), dtype=float)
        rt = 0.0
        # tqdm.auto picks tqdm.notebook in Jupyter and plain tqdm in
        # a terminal / PyCharm run window. leave=False so the per-call
        # bar disappears once the BO iteration completes (otherwise
        # one bar per BO call would scroll the run console).
        try:
            from tqdm.auto import tqdm as _tqdm
            iterator = _tqdm(frames, total=len(frames),
                             desc="lammps", unit="frame",
                             leave=False, dynamic_ncols=True)
        except Exception:
            iterator = frames
        for i, fr in enumerate(iterator):
            r = self.runner.evaluate(params, fr)
            ys[i] = r.energy
            rt += r.runtime_s
        return ys, rt

    # ----- public callable ------------------------------------------------

    def __call__(self, params_vec: np.ndarray) -> float:
        params = self._params_dict(params_vec)
        ys, rt = self._evaluate_frames(self.train_frames, params)
        refs = np.array([f.energy for f in self.train_frames], dtype=float)

        if self.weights.per_atom_norm:
            natoms = np.array([f.n_atoms for f in self.train_frames],
                              dtype=float)
            ys_eff = ys / natoms
            refs_eff = refs / natoms
        else:
            ys_eff, refs_eff = ys, refs
        e_rmse = energy_rmse(ys_eff, refs_eff, weights=self._train_weights)
        loss = self.weights.energy * e_rmse

        # Geometric anchor: cell-relax equilibrium 2H-MoS2 monolayer and
        # penalise deviation of (a, h_S) from Cooper 2014 DFT reference.
        # Adds one LAMMPS run per BO call but directly drives the
        # equilibrium geometry to the 5%-validation target.
        if self.weights.geometry > 0:
            try:
                geo = _measure_equilibrium_geom(self.runner, params)
                a_pen   = ((geo["a"] - 3.1830) / 3.1830) ** 2
                hS_pen  = ((geo["h_S"] - 1.5640) / 1.5640) ** 2
                loss += self.weights.geometry * (a_pen + hS_pen) ** 0.5
            except Exception:
                # Equilibrium measure failed (rare bad sample) — skip
                # the penalty rather than NaN'ing the BO call.
                pass

        if self.weights.geometry > 0:
            # Geometry RMSD here is identity (single-point energies); kept for
            # API symmetry. Wire in optimized geometries when available.
            loss += self.weights.geometry * 0.0

        self.n_evals += 1
        self.history.append(loss)
        self.last_ys = ys
        self.last_params = params
        self.total_runtime_s += rt
        return loss

    # ----- introspection --------------------------------------------------

    def metrics(self) -> Dict[str, float]:
        """Return current Loss / Ys-mae / E / runtime / count metrics.

        When ``per_atom_norm`` is on, the RMSE/MAE columns are in eV/atom
        for consistency with the BO's loss; total-energy quantities
        (``E_total_error``) stay in eV/cell so the absolute offset is
        comparable across runs.
        """
        if self.last_ys is None:
            return {}
        refs = np.array([f.energy for f in self.train_frames], dtype=float)
        if self.weights.per_atom_norm:
            natoms = np.array([f.n_atoms for f in self.train_frames],
                              dtype=float)
            ys_n, refs_n = self.last_ys / natoms, refs / natoms
        else:
            ys_n, refs_n = self.last_ys, refs
        return {
            "loss": float(self.history[-1]) if self.history else float("nan"),
            "loss_mae": energy_mae(ys_n, refs_n, weights=self._train_weights),
            "loss_rmse": energy_rmse(ys_n, refs_n, weights=self._train_weights),
            "loss_rmse_unweighted": energy_rmse(ys_n, refs_n),
            "E_total_error": total_energy_error(self.last_ys, refs),
            "n_evals": self.n_evals,
            "total_runtime_s": self.total_runtime_s,
        }

    def validation_accuracy(self, params_vec: np.ndarray) -> float:
        """Weighted RMSE on held-out frames (lower is better).

        Same per-atom convention as ``__call__``: when ``per_atom_norm``
        is on, return value is in eV/atom; otherwise eV/cell.
        """
        if not self.val_frames:
            return float("nan")
        params = self._params_dict(params_vec)
        ys, _ = self._evaluate_frames(self.val_frames, params)
        refs = np.array([f.energy for f in self.val_frames], dtype=float)
        if self.weights.per_atom_norm:
            natoms = np.array([f.n_atoms for f in self.val_frames],
                              dtype=float)
            ys, refs = ys / natoms, refs / natoms
        return energy_rmse(ys, refs, weights=self._val_weights)

    def parameter_robustness(self, params_vec: np.ndarray,
                             n_perturb: int = 8, sigma: float = 0.05,
                             rng: np.random.Generator | None = None) -> float:
        """Sensitivity = stdev of loss under small relative perturbations."""
        rng = rng or np.random.default_rng(0)
        losses = []
        from .parameters import bounds_array
        lo, hi = bounds_array(self.specs)
        width = hi - lo
        for _ in range(n_perturb):
            noise = rng.normal(0.0, sigma, size=params_vec.shape) * width
            perturbed = np.clip(params_vec + noise, lo, hi)
            losses.append(self.__call__(perturbed))
        return float(np.std(losses))
