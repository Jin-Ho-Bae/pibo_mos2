"""
EZFF-format VASP interface port.

Mirrors ``ezff/interfaces/vasp.py`` from arvk/EZFF: thin readers for
POSCAR/CONTCAR, vasprun.xml, and a Phonopy band.dat-style phonon
dispersion file. Where upstream EZFF uses ``xtal.AtTraj``, we return
this project's lighter ``ezff_lammps.Structure`` so the EZFF-style
``job`` can consume it without a hard ``xtal`` dependency.

The OUTCAR branch isn't a one-to-one port (upstream EZFF reads energies
only from ``vasprun.xml``). We expose the project's existing
``vasp_reader.parse_outcar`` under an EZFF-shaped helper so callers
that have OUTCAR-only data — which is your ``vasp_calculations/``
layout — still get a uniform reader API.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from .ezff_lammps import Atom, Snapshot, Structure
from . import vasp_reader as _vr


# ---------------------------------------------------------------------------
# read_atomic_structure
# ---------------------------------------------------------------------------

def read_atomic_structure(structure_file: str) -> Structure:
    """Read atomic structure from POSCAR/CONTCAR/vasprun.xml/XDATCAR.

    Returns a Structure with one snapshot (or many, for XDATCAR /
    vasprun.xml trajectories) in the same shape the EZFF-style ``job``
    consumes.

    Upstream EZFF returns ``xtal.AtTraj``; we return ``Structure`` from
    ``ezff_lammps``. The field names line up (``snaplist``, ``box``,
    ``abc``, ``ang``, ``atomlist``) so downstream EZFF code patterns
    work unchanged.
    """
    fname = structure_file.lower()
    if "xdatcar" in fname or fname.endswith(".xml") or os.path.isdir(
            structure_file):
        return _read_trajectory_vasp(structure_file)
    if "poscar" in fname or "contcar" in fname:
        return _read_snapshot_vasp(structure_file)
    raise ValueError(
        f"unrecognised VASP structure file {structure_file!r}; "
        "expected POSCAR/CONTCAR/XDATCAR/vasprun.xml")


def _read_snapshot_vasp(poscar_path: str) -> Structure:
    cell, positions, species = _vr.parse_poscar(poscar_path)
    snap = Snapshot()
    snap.box = np.asarray(cell, dtype=float)
    snap.abc = np.linalg.norm(snap.box, axis=1)
    # Angles from the cell vectors (radians).
    a, b, c = snap.box
    cos_alpha = float(np.dot(b, c) / (np.linalg.norm(b) * np.linalg.norm(c)))
    cos_beta = float(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c)))
    cos_gamma = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    snap.ang = np.arccos(np.clip([cos_alpha, cos_beta, cos_gamma], -1.0, 1.0))
    for sp, p in zip(species, positions):
        snap.atomlist.append(Atom(element=sp, cart=np.asarray(p, float)))
    s = Structure()
    s.snaplist.append(snap)
    return s


def _read_trajectory_vasp(path: str) -> Structure:
    """Best-effort vasprun.xml / XDATCAR / directory reader.

    Upstream EZFF defers to ``xtal.AtTraj.read_trajectory_vasp``. We
    don't vendor xtal; instead we parse vasprun.xml via stdlib
    ``xml.etree.ElementTree``, capturing every ``<calculation>`` block.
    Directory inputs walk all immediate ``vasprun.xml`` files in
    alphabetical order — same convention as EZFF for ensemble runs.
    """
    if os.path.isdir(path):
        xmls = sorted(p for p in os.listdir(path) if p.endswith(".xml"))
        out = Structure()
        for xml in xmls:
            sub = _read_trajectory_vasp(os.path.join(path, xml))
            out.snaplist.extend(sub.snaplist)
        return out

    if not path.endswith(".xml"):
        raise NotImplementedError(
            "XDATCAR parsing requires xtal.AtTraj; "
            "pass a vasprun.xml or directory of vasprun.xml files.")

    import xml.etree.ElementTree as ET
    tree = ET.parse(path)
    root = tree.getroot()
    # Element list (atominfo/array[@name='atoms']/set/rc/c[0]).
    species: List[str] = []
    atominfo = root.find(".//atominfo")
    if atominfo is not None:
        for rc in atominfo.findall(".//array[@name='atoms']/set/rc"):
            cs = rc.findall("c")
            if cs:
                species.append(cs[0].text.strip())

    out = Structure()
    for calc in root.findall(".//calculation"):
        # cell basis
        basis = calc.find(".//structure/crystal/varray[@name='basis']")
        if basis is None:
            continue
        cell = np.array([[float(x) for x in v.text.split()]
                         for v in basis.findall("v")])
        # positions (direct)
        positions_node = calc.find(
            ".//structure/varray[@name='positions']")
        if positions_node is None:
            continue
        frac = np.array([[float(x) for x in v.text.split()]
                         for v in positions_node.findall("v")])
        cart = frac @ cell

        snap = Snapshot()
        snap.box = cell
        snap.abc = np.linalg.norm(cell, axis=1)
        a, b, c = cell
        cos_alpha = float(np.dot(b, c) / (snap.abc[1] * snap.abc[2]))
        cos_beta = float(np.dot(a, c) / (snap.abc[0] * snap.abc[2]))
        cos_gamma = float(np.dot(a, b) / (snap.abc[0] * snap.abc[1]))
        snap.ang = np.arccos(np.clip([cos_alpha, cos_beta, cos_gamma],
                                     -1.0, 1.0))

        for sp, p in zip(species, cart):
            snap.atomlist.append(Atom(element=sp, cart=p))

        # energy_sigma->0 if present
        e_node = calc.find(
            ".//energy/i[@name='e_0_energy']")
        if e_node is not None:
            try:
                snap.energy = float(e_node.text)
            except (ValueError, TypeError):
                pass

        # forces if present
        forces_node = calc.find(".//varray[@name='forces']")
        if forces_node is not None:
            forces = np.array([[float(x) for x in v.text.split()]
                               for v in forces_node.findall("v")])
            for atom, f in zip(snap.atomlist, forces):
                atom.force = f
        out.snaplist.append(snap)
    return out


# ---------------------------------------------------------------------------
# read_energy (vasprun.xml + OUTCAR convenience)
# ---------------------------------------------------------------------------

def read_energy(source: str) -> np.ndarray:
    """Read energies (eV) from a VASP output.

    EZFF-API parity: pass a ``vasprun.xml`` path to get every
    ``<calculation>``'s ``e_0_energy`` (the energy(sigma->0) value)
    as a 1-D array. As a non-EZFF convenience, an OUTCAR path is also
    accepted and produces a 1-element array via this project's
    ``vasp_reader.parse_outcar``.
    """
    lower = source.lower()
    if lower.endswith(".xml"):
        traj = _read_trajectory_vasp(source)
        return np.array([s.energy for s in traj.snaplist
                         if s.energy is not None], dtype=float)
    if "outcar" in os.path.basename(lower):
        energy, _ = _vr.parse_outcar(source)
        return np.array([energy], dtype=float)
    raise ValueError(
        f"read_energy: unsupported source {source!r}; "
        "pass *.xml or OUTCAR")


# ---------------------------------------------------------------------------
# read_phonon_dispersion (Phonopy band.dat)
# ---------------------------------------------------------------------------

def read_phonon_dispersion(phonon_dispersion_file: str) -> np.ndarray:
    """Read a Phonopy/VASP band.dat-style dispersion as a 2-D array.

    Mirrors EZFF's behaviour: skip three header/comment lines, then
    accumulate frequency columns into ``(n_bands, n_qpoints)`` shape.
    Empty lines separate q-segments; double-empty lines separate bands.
    """
    with open(phonon_dispersion_file, "r", encoding="utf-8") as fh:
        # Skip three header lines (EZFF convention).
        fh.readline(); fh.readline(); fh.readline()
        segment: List[float] = []
        band: List[List[float]] = []
        full_dispersion: List[List[List[float]]] = []
        prev = "NOT EMPTY"
        for line in fh:
            data = line.strip()
            if data == "" and prev == "":
                full_dispersion.append(band)
                band = []
            elif data == "":
                band.append(segment)
                segment = []
            else:
                segment.append(float(data.split()[-1]))
            prev = data

    if not full_dispersion:
        return np.array([])
    g = np.ravel(full_dispersion[0])
    for i in range(1, len(full_dispersion)):
        fd = np.ravel(full_dispersion[i])
        if fd.size:
            g = np.vstack((g, fd))
    return g


# ---------------------------------------------------------------------------
# OUTCAR/CONTCAR convenience — bridges your existing dataset layout
# ---------------------------------------------------------------------------

def read_outcar_frame(outcar_path: str, contcar_path: str,
                      tag: str = "", category: str = "default"
                      ) -> _vr.DFTFrame:
    """Shortcut to ``vasp_reader.parse_frame`` for the OUTCAR + CONTCAR
    pair used everywhere in ``vasp_calculations/``. Not in upstream EZFF —
    added so a single EZFF-style import gives access to your data."""
    return _vr.parse_frame(outcar_path, contcar_path, tag, category=category)
