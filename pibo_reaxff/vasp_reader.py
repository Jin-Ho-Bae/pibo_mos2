"""
Lightweight VASP OUTCAR/POSCAR reader for DFT reference data.

This is intentionally dependency-light (no pymatgen required) so it works in
a clean Colab kernel before heavy packages finish installing. Use pymatgen
or ASE for production-grade parsing.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class DFTFrame:
    """One DFT calculation: energy, geometry, optional forces, plus a tag.

    ``category`` labels the training-data category for the manuscript's
    weighted loss (Table 1):

        equilibrium    → w = 10.0  (1T-, 2H-MoS2 ground states)
        bond_pes       → w =  5.0  (Mo–S bond-distance scans)
        angle_pes      → w =  3.0  (S–Mo–S valence-angle scans)
        dihedral_pes   → w =  1.0  (S–Mo–S–Mo torsion scans)
        strained       → w =  0.5  (cells > 1 eV/atom above equilibrium)
        default        → w =  1.0  (uncategorized, legacy fallback)
    """
    tag: str
    energy: float
    positions: np.ndarray  # (N, 3) in Angstrom
    cell: np.ndarray       # (3, 3) in Angstrom
    species: List[str]
    forces: np.ndarray | None = None
    category: str = "default"

    @property
    def n_atoms(self) -> int:
        return len(self.species)


# Mapping from on-disk subdirectory names (legacy /CODE/data convention) to
# Manuscript Table 1 category labels. ``load_directory`` consults this when
# tagging frames so that ReaxFFLoss can apply the right weight per frame.
_DIR_TO_CATEGORY: Dict[str, str] = {
    "bond":        "bond_pes",
    "angle":       "angle_pes",
    "torsion":     "dihedral_pes",
    "dihedral":    "dihedral_pes",
    "nonbond":     "strained",
    "vdw_coulomb": "strained",
    "equilibrium": "equilibrium",
    "strained":    "strained",
}


_E0_PATTERN = re.compile(r"energy\s+without\s+entropy\s*=\s*([-\d.E+]+)")
_E_SIGMA_PATTERN = re.compile(r"energy\(sigma->0\)\s*=\s*([-\d.E+]+)")


def parse_outcar(path: str) -> Tuple[float, np.ndarray | None]:
    """Return the final SCF energy(sigma->0) and last force block from an OUTCAR.

    Forces parsing is best-effort: returns None when the block is absent.
    """
    with open(path, "r", errors="ignore") as f:
        text = f.read()

    sigma_matches = _E_SIGMA_PATTERN.findall(text)
    if sigma_matches:
        energy = float(sigma_matches[-1])
    else:
        e0_matches = _E0_PATTERN.findall(text)
        if not e0_matches:
            raise ValueError(f"No energy line found in {path}")
        energy = float(e0_matches[-1])

    forces = None
    blocks = re.findall(
        r"POSITION\s+TOTAL-FORCE.*?\n\s*-+\n(.*?)\n\s*-+",
        text,
        flags=re.DOTALL,
    )
    if blocks:
        rows = []
        for line in blocks[-1].strip().splitlines():
            parts = line.split()
            if len(parts) >= 6:
                rows.append([float(p) for p in parts[3:6]])
        if rows:
            forces = np.asarray(rows, dtype=float)

    return energy, forces


def parse_poscar(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (cell[3x3], positions[Nx3], species_list) from POSCAR/CONTCAR."""
    with open(path, "r") as f:
        lines = [ln.rstrip() for ln in f.readlines()]

    scale = float(lines[1])
    cell = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)])
    cell *= scale

    elements = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    species: List[str] = []
    for el, n in zip(elements, counts):
        species.extend([el] * n)

    coord_type = lines[7].strip().lower()
    start = 8
    if coord_type.startswith("s"):  # Selective dynamics line
        coord_type = lines[8].strip().lower()
        start = 9

    n = sum(counts)
    pos = np.array(
        [[float(x) for x in lines[start + i].split()[:3]] for i in range(n)]
    )
    if coord_type.startswith("d"):
        pos = pos @ cell  # Direct -> Cartesian
    return cell, pos, species


def parse_frame(outcar_path: str, contcar_path: str, tag: str,
                category: str = "default") -> DFTFrame:
    """Combine OUTCAR energy/forces with CONTCAR geometry into a DFTFrame."""
    energy, forces = parse_outcar(outcar_path)
    cell, positions, species = parse_poscar(contcar_path)
    return DFTFrame(
        tag=tag,
        energy=energy,
        positions=positions,
        cell=cell,
        species=species,
        forces=forces,
        category=category,
    )


def load_directory(directory: str,
                   category: str | None = None) -> List[DFTFrame]:
    """Pair OUTCAR_<tag> and CONTCAR_<tag> files in `directory` into frames.

    If ``category`` is ``None``, infer it from the directory's basename via
    ``_DIR_TO_CATEGORY`` (e.g. ``bond/`` → ``bond_pes``).
    """
    if not os.path.isdir(directory):
        return []
    if category is None:
        category = _DIR_TO_CATEGORY.get(
            os.path.basename(directory.rstrip("/\\")).lower(), "default")
    outcars = sorted(glob.glob(os.path.join(directory, "OUTCAR_*")))
    frames: List[DFTFrame] = []
    for o in outcars:
        tag = os.path.basename(o)[len("OUTCAR_"):]
        c = os.path.join(directory, f"CONTCAR_{tag}")
        if not os.path.exists(c):
            continue
        try:
            frames.append(parse_frame(o, c, tag, category=category))
        except Exception as e:  # noqa: BLE001 - keep loading other frames
            print(f"[vasp_reader] skipped {tag}: {e}")
    return frames


def load_dataset(root: str, blocks: List[str] | None = None,
                 max_dE_per_atom_eV: float | None = None,
                 ) -> Dict[str, List[DFTFrame]]:
    """Load ``{block: [frames]}`` from ``root/{bond,angle,torsion,nonbond}/``.

    Each frame is tagged with the manuscript Table 1 category corresponding
    to its on-disk subdirectory.

    Outlier filter (physics-grounded)
    ---------------------------------
    If ``max_dE_per_atom_eV`` is set, frames whose DFT energy is more than
    ``max_dE_per_atom_eV`` above the *per-category* minimum-energy frame
    are dropped. ReaxFF's bond-order formulation saturates beyond
    ~3-5 eV/atom above the local equilibrium — energies above that cap
    are physically outside the model's domain of validity and dominate
    the squared-error loss without informing parameter values. Dropping
    them keeps the loss reflective of fittable chemistry. Standard
    practice in ReaxFF training literature (see Aktulga et al. 2012,
    Senftle et al. 2016).
    """
    blocks = blocks or ["bond", "angle", "torsion", "nonbond"]
    dataset: Dict[str, List[DFTFrame]] = {}
    for b in blocks:
        # Disk layout uses 'nonbond' for the strained-cells set.
        disk_name = "nonbond" if b in ("vdw_coulomb", "offdiag") else b
        frames = load_directory(os.path.join(root, disk_name))
        if max_dE_per_atom_eV is not None and frames:
            # Per-category minimum-energy reference.
            by_cat: Dict[str, List[DFTFrame]] = {}
            for f in frames:
                by_cat.setdefault(f.category, []).append(f)
            kept: List[DFTFrame] = []
            for cat, cat_frames in by_cat.items():
                emin = min(fr.energy for fr in cat_frames)
                for fr in cat_frames:
                    dE_pa = (fr.energy - emin) / max(1, fr.n_atoms)
                    if dE_pa <= max_dE_per_atom_eV:
                        kept.append(fr)
            n_drop = len(frames) - len(kept)
            if n_drop:
                print(f"[vasp_reader] {b}: dropping {n_drop}/{len(frames)} "
                      f"frames with dE/atom > {max_dE_per_atom_eV:.2f} eV/atom")
            frames = kept
        dataset[b] = frames
        cat = frames[0].category if frames else "(none)"
        print(f"[vasp_reader] {b}: {len(frames)} frames  [category={cat}]")
    return dataset


def train_validation_split(frames: List[DFTFrame], val_ratio: float = 0.2,
                           seed: int = 0) -> Tuple[List[DFTFrame], List[DFTFrame]]:
    """Deterministic train/val split for held-out validation accuracy."""
    rng = np.random.default_rng(seed)
    n = len(frames)
    if n < 2:
        return frames, []
    idx = rng.permutation(n)
    cut = max(1, int(round(n * (1.0 - val_ratio))))
    train_idx, val_idx = idx[:cut], idx[cut:]
    return [frames[i] for i in train_idx], [frames[i] for i in val_idx]
