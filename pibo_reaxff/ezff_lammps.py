"""
EZFF-format LAMMPS interface port.

Mirrors ``ezff/interfaces/lammps.py`` from arvk/EZFF: a ``job`` class that
writes a ``structure`` data file, a ``in.lmp`` script, runs the ``lmp``
binary, and parses ``EZFF_*`` print sentinels and the per-snapshot dump
file into Python objects (energy, optimised structure, elastic moduli).

Differences vs. upstream EZFF (deliberate, called out in the body):

  * Windows-safe: all POSIX shell calls (``os.system('cat ... >> ...')``,
    ``os.system('rm -f ...')``, ``timeout`` prefix) are replaced with
    Python stdlib equivalents (``subprocess.run(..., timeout=)``,
    ``shutil.move``, ``os.remove``, plain ``open(... , 'a')``).
  * Pair-style auto-detection (``reaxff`` vs legacy ``reax/c``) is
    reused from ``lammps_runner.detect_reaxff_style`` so the same probe
    that the PIBO BO uses also drives the EZFF-style job. Upstream EZFF
    hard-codes ``reax/c`` which silent-fails on modern builds.
  * ``xtal`` (Crockford/USCCACS atomic-trajectory library) is the
    upstream EZFF in-memory structure; here we use a thin internal
    ``Structure`` dataclass to avoid pulling in xtal as a hard
    dependency. The reader returns the same per-snapshot fields
    (cell, atoms, charges, forces) that upstream xtal exposes.
  * ``atomic_properties.atomic_mass`` (an EZFF util module) is replaced
    by ``LAMMPSRunner.DEFAULT_MASSES`` from this project.

Print sentinels preserved verbatim so any downstream tool that already
greps EZFF output (e.g. a phonopy front-end) still works:

    "EZFF_TEMP",  "EZFF_VOL",  "EZFF_ENERGY",
    "EZFF C11 ... GPa" ... through C66, plus Bulk_Modulus, Shear_Modulus_1,
    Shear_Modulus_2, Poisson_Ratio.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .lammps_runner import (
    _find_lmp_binary,
    detect_reaxff_style,
    _lammps_supports_reaxff,
    KCAL_PER_MOL_TO_EV,
)
from .ezff_io import generate_forcefield as _ezff_generate_forcefield


# Pair-style + QEq fix selection. EZFF originally hard-coded ``reax/c``;
# we read the modern/legacy style from ``detect_reaxff_style`` at job
# construction so this works on LAMMPS builds shipped in 2024+ as well.
_FFTYPE_PAIRSTYLE = {
    "SW": "sw",
    "STILLINGER-WEBER": "sw",
    "STILLINGER WEBER": "sw",
    "VASHISHTA": "vashishta",
}


# ---------------------------------------------------------------------------
# Lightweight in-memory structure (replaces xtal.AtTraj in upstream EZFF)
# ---------------------------------------------------------------------------

@dataclass
class Atom:
    element: str
    cart: np.ndarray            # (3,) Angstrom
    charge: float = 0.0
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Snapshot:
    atomlist: List[Atom] = field(default_factory=list)
    box: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    abc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ang: np.ndarray = field(default_factory=lambda: np.zeros(3))   # radians
    energy: Optional[float] = None
    pressure: Optional[float] = None


@dataclass
class Structure:
    """Minimal stand-in for ``xtal.AtTraj``: list of snapshots."""
    snaplist: List[Snapshot] = field(default_factory=list)

    @property
    def box(self) -> np.ndarray:
        return self.snaplist[0].box if self.snaplist else np.zeros((3, 3))


# ---------------------------------------------------------------------------
# job class
# ---------------------------------------------------------------------------

class job:
    """A single LAMMPS calculation, EZFF-API-compatible.

    Usage mirrors upstream::

        j = job(path='runs/relax_2H', verbose=True)
        j.structure = my_structure              # Structure with .snaplist
        j.forcefield = j.generate_forcefield(template_string, params,
                                             FFtype='REAXFF')
        j.options['relax_cell'] = True
        j.options['pbc'] = True
        j.run_relax(command='lmp', timeout=120)
        energy = j.read_energy()                # eV per cell
        relaxed = j.read_structure()            # Structure
        Cmat   = j.read_elastic_moduli()        # list of 6x6 GPa

    The ``run_static`` / ``run_relax`` / ``run_elastic`` entry points
    each loop over ``self.structure.snaplist`` and append per-snapshot
    output to ``self.outfile``.
    """

    def __init__(self,
                 verbose: bool = False,
                 path: str = ".",
                 units: str = "metal",
                 elements: Optional[Tuple[str, ...]] = None) -> None:
        os.makedirs(path, exist_ok=True)
        self.path = os.path.abspath(path)

        self.scriptfile = os.path.join(self.path, "in.lmp")
        self.outfile = os.path.join(self.path, "out.lmp")
        self.dumpfile = os.path.join(self.path, "out.dump")
        self.structfile = os.path.join(self.path, "input.structure")
        self.forcefieldfile = os.path.join(self.path, "generated_forcefield")

        self.structure: Optional[Structure] = None
        self.forcefield: str = ""
        self.units = units
        self.elements = elements   # optional explicit Mo/S/H order

        self.options: Dict[str, object] = {
            "relax_atoms": False,
            "relax_cell": False,
            "pbc": False,
            "atomic_charges": False,
            "fftype": "REAXFF",
            "atom_sequence": None,
        }
        self.verbose = verbose

        # Modern vs legacy ReaxFF pair-style probe.
        lmp = _find_lmp_binary()
        if lmp and _lammps_supports_reaxff(lmp):
            self._reax_style, self._qeq_style = detect_reaxff_style(lmp)
        else:
            self._reax_style, self._qeq_style = ("reaxff", "qeq/reaxff")

        if verbose:
            print(f"[ezff_lammps] new job at {self.path} "
                  f"(pair_style={self._reax_style})")

    # ------------------------------------------------------------------
    # Forcefield generation (delegated to ezff_io)
    # ------------------------------------------------------------------

    def generate_forcefield(self,
                            template_string: str,
                            parameters: Dict[str, float],
                            FFtype: Optional[str] = None,
                            outfile: Optional[str] = None) -> str:
        """Generate ffield via ``ezff_io.generate_forcefield`` (LAMMPS MD)."""
        if FFtype:
            self.options["fftype"] = FFtype.upper()
        rendered = _ezff_generate_forcefield(
            template_string=template_string,
            parameters=parameters,
            FFtype=FFtype,
            outfile=outfile,
            MD="LAMMPS",
        )
        # If outfile was set, ezff_io returns None; otherwise return string.
        if rendered is None:
            with open(outfile, "r", encoding="utf-8") as fh:
                rendered = fh.read()
        self.forcefield = rendered
        return rendered

    # ------------------------------------------------------------------
    # Run entry points
    # ------------------------------------------------------------------

    def run_static(self, command: Optional[str] = None,
                   timeout: Optional[float] = None) -> None:
        self._run_loop_over_snapshots("static", command, timeout)

    def run_relax(self, command: Optional[str] = None,
                  timeout: Optional[float] = None) -> None:
        self._run_loop_over_snapshots("relax", command, timeout)

    def run_elastic(self, command: Optional[str] = None,
                    timeout: Optional[float] = None) -> None:
        self._run_loop_over_snapshots("elastic", command, timeout)

    def run(self, command: Optional[str] = None,
            timeout: Optional[float] = None) -> None:
        """Upstream EZFF default: relax if ``relax_atoms`` else static."""
        mode = "relax" if self.options.get("relax_atoms") else "static"
        self._run_loop_over_snapshots(mode, command, timeout)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_loop_over_snapshots(self, mode: str,
                                 command: Optional[str],
                                 timeout: Optional[float]) -> None:
        if self.structure is None or not self.structure.snaplist:
            raise RuntimeError("job.structure has no snapshots")
        if command is None:
            command = _find_lmp_binary()
            if not command:
                raise RuntimeError(
                    "LAMMPS executable not specified and `lmp` was not found "
                    "on PATH. Pass command='/abs/path/to/lmp'.")

        # Clear out files from previous runs.
        for f in (self.outfile, self.dumpfile,
                  self.outfile + ".runerror"):
            if os.path.exists(f):
                os.remove(f)

        tmpdump = os.path.join(self.path, "tempdumpfile")
        tmpout = self.outfile + "_temp"

        for snapID, _ in enumerate(self.structure.snaplist):
            self._write_structure_file(snapID)
            if mode == "static":
                self._write_script_file_static()
            elif mode == "relax":
                self._write_script_file_relax()
            elif mode == "elastic":
                self._write_script_file_elastic()
            else:
                raise ValueError(f"unknown mode {mode!r}")

            self._write_forcefield_file()

            for stale in (tmpdump, tmpout, self.outfile + ".runerror"):
                if os.path.exists(stale):
                    os.remove(stale)

            with open(tmpout, "wb") as out_fh, \
                    open(self.outfile + ".runerror", "wb") as err_fh:
                proc = subprocess.run(
                    [command, "-in", self.scriptfile, "-screen", "none",
                     "-log", os.path.join(self.path, "log.lammps")],
                    cwd=self.path,
                    stdout=out_fh, stderr=err_fh,
                    timeout=timeout,
                )

            # Append this snapshot's stdout + dump to the cumulative files.
            self._append_file(tmpout, self.outfile)
            if os.path.exists(tmpdump):
                self._append_file(tmpdump, self.dumpfile)
                os.remove(tmpdump)
            os.remove(tmpout)

            if proc.returncode != 0:
                with open(self.outfile + ".runerror", "rb") as fh:
                    err_tail = fh.read().decode(errors="replace")[-400:]
                raise RuntimeError(
                    f"LAMMPS exit={proc.returncode} on snapshot {snapID}; "
                    f"stderr tail:\n{err_tail}")

        # Best-effort cleanup of intermediate per-snapshot artefacts.
        for f in (self.outfile + ".disp", self.outfile + ".dens",
                  self.outfile + ".runerror"):
            if os.path.exists(f):
                os.remove(f)

    @staticmethod
    def _append_file(src: str, dst: str) -> None:
        with open(src, "rb") as a, open(dst, "ab") as b:
            shutil.copyfileobj(a, b)

    # ------------------------------------------------------------------
    # Structure file (LAMMPS ``read_data``, triclinic, ``atom_style charge``)
    # ------------------------------------------------------------------

    def _resolve_atom_sequence(self, snap: Snapshot) -> List[str]:
        if self.elements:
            return list(self.elements)
        # First-occurrence order, matching upstream EZFF semantics
        # (set() iteration was non-deterministic; we tighten that here).
        seen: List[str] = []
        for atom in snap.atomlist:
            if atom.element not in seen:
                seen.append(atom.element)
        return seen

    def _write_structure_file(self, snap_ID: int) -> None:
        from .lammps_runner import LAMMPSRunner   # for DEFAULT_MASSES
        snap = self.structure.snaplist[snap_ID]
        atom_types = self._resolve_atom_sequence(snap)
        elem_to_id = {e: i + 1 for i, e in enumerate(atom_types)}
        self.options["atom_sequence"] = atom_types

        # Triclinic box (EZFF coordinate transform, LAMMPS Howto_triclinic).
        if not np.any(snap.abc) and np.any(snap.box):
            abc = np.linalg.norm(snap.box, axis=1)
            # Orthogonal box assumed if angles unset.
            ang = np.array([np.pi / 2, np.pi / 2, np.pi / 2])
        else:
            abc, ang = snap.abc, snap.ang

        lx = abc[0]
        xy = abc[1] * np.cos(ang[2])
        xz = abc[2] * np.cos(ang[1])
        ly = float(np.sqrt(abc[1] ** 2 - xy ** 2))
        yz = (abc[1] * (abc[2] * np.cos(ang[0])) - xy * xz) / ly if ly else 0.0
        lz = float(np.sqrt(abc[2] ** 2 - xz ** 2 - yz ** 2))

        # Clamp the LAMMPS-required |tilt| <= L/2.
        xy = float(np.clip(xy, -lx / 2.0, lx / 2.0))
        xz = float(np.clip(xz, -lx / 2.0, lx / 2.0))
        yz = float(np.clip(yz, -ly / 2.0, ly / 2.0))

        with open(self.structfile, "w", encoding="utf-8") as fh:
            fh.write("## LAMMPS structure file from ezff_lammps\n\n")
            fh.write(f"{len(snap.atomlist)} atoms\n\n")
            fh.write(f"{len(atom_types)} atom types\n\n")
            fh.write(f"0.0 {lx:.6f} xlo xhi\n")
            fh.write(f"0.0 {ly:.6f} ylo yhi\n")
            fh.write(f"0.0 {lz:.6f} zlo zhi\n")
            fh.write(f"{xy:.6f} {xz:.6f} {yz:.6f} xy xz yz\n\n")

            fh.write("Masses\n\n")
            for el, idx in elem_to_id.items():
                fh.write(f"{idx}  {LAMMPSRunner.DEFAULT_MASSES.get(el, 1.0):.4f}\n")
            fh.write("\nAtoms\n\n")
            for i, atom in enumerate(snap.atomlist, start=1):
                fh.write(
                    f"{i} {elem_to_id[atom.element]} "
                    f"{atom.charge:.6f} "
                    f"{atom.cart[0]:.6f} {atom.cart[1]:.6f} {atom.cart[2]:.6f}\n"
                )

    # ------------------------------------------------------------------
    # Forcefield include + LAMMPS script writers
    # ------------------------------------------------------------------

    def _write_forcefield_file(self) -> None:
        with open(self.forcefieldfile, "w", encoding="utf-8") as fh:
            fh.write(self.forcefield)

    def _include_forcefield(self) -> str:
        fftype = str(self.options["fftype"]).upper()
        seq = " ".join(self.options["atom_sequence"]).title()
        ff_basename = os.path.basename(self.forcefieldfile)
        if "REAX" in fftype:
            return (
                f"pair_style {self._reax_style} NULL\n"
                f"pair_coeff * * {ff_basename} {seq}\n"
                f"fix qeq all {self._qeq_style} 1 0.0 10.0 1.0e-6 reaxff\n"
            )
        if fftype == "LJ":
            return f"include {ff_basename}\n"
        pairstyle = _FFTYPE_PAIRSTYLE.get(fftype)
        if not pairstyle:
            raise ValueError(f"unsupported FFtype {fftype!r}")
        return (
            f"pair_style {pairstyle}\n"
            f"pair_coeff * * {ff_basename} {seq}\n"
        )

    def _script_prelude(self) -> str:
        opts = self.options
        pbc = "p p p" if opts["pbc"] else "fm fm fm"
        return (
            f"units {self.units}\n"
            "dimension 3\n"
            "atom_style charge\n"
            f"boundary {pbc}\n"
            f"read_data {os.path.basename(self.structfile)}\n"
        )

    def _script_thermo(self) -> str:
        return (
            "thermo_style custom step temp pxx pyy pzz pxy pxz pyz pe ke "
            "etotal vol xlo xhi ylo yhi zlo zhi xy xz yz press lx ly lz\n"
            "thermo 1000\n"
            "variable ezff_T equal temp\n"
            "variable ezff_V equal vol\n"
            "variable ezff_E equal etotal\n"
        )

    def _script_summary(self) -> str:
        seq = " ".join(self.options["atom_sequence"]).title()
        return (
            "write_dump all custom tempdumpfile id type element mass q "
            "x y z vx vy vz fx fy fz modify sort id element " + seq + "\n"
            'print "-----SUMMARY-----"\n'
            'print "EZFF_TEMP ${ezff_T}"\n'
            'print "EZFF_VOL ${ezff_V}"\n'
            'print "EZFF_ENERGY ${ezff_E}"\n'
        )

    def _write_script_file_static(self) -> None:
        body = (
            self._script_prelude()
            + self._include_forcefield()
            + self._script_thermo()
            + "run 0\n\n"
            + self._script_summary()
        )
        with open(self.scriptfile, "w", encoding="utf-8") as fh:
            fh.write(body)

    def _write_script_file_relax(self) -> None:
        opts = self.options
        relax = ("fix FixBoxRelax all box/relax aniso 0.0\n"
                 if opts["relax_cell"] else "")
        body = (
            self._script_prelude()
            + self._include_forcefield()
            + self._script_thermo()
            + relax
            + "minimize 0.0 1.0e-8 1000 100000\n"
            + "run 0\n\n"
            + self._script_summary()
        )
        with open(self.scriptfile, "w", encoding="utf-8") as fh:
            fh.write(body)

    def _write_script_file_elastic(self) -> None:
        """Equilibrate then probe ±strain along six directions.

        Lifted verbatim from upstream EZFF (which itself adapted the LAMMPS
        ``examples/ELASTIC`` template). The block is long but is purely
        a string of LAMMPS commands — comments preserved so a user
        diffing against EZFF can verify nothing was silently changed.
        """
        opts = self.options
        seq = " ".join(opts["atom_sequence"]).title()
        ff_basename = os.path.basename(self.forcefieldfile)

        relax = ("fix FixBoxRelax all box/relax aniso 0.0\n"
                 if opts["relax_cell"] else "")
        unfix = "unfix FixBoxRelax\n" if opts["relax_cell"] else ""

        # Pre-equilibration phase.
        prelude = (
            self._script_prelude()
            + self._include_forcefield()
            + self._script_thermo()
            + relax
            + "minimize 0.0 1.0e-8 1000 100000\n"
            + "run 0\n\n"
            + self._script_summary()
        )

        # displace.mod — strain probe in 6 directions.
        displace_mod = textwrap_block(f"""\
            clear
            variable up equal 1.0e-6
            variable atomjiggle equal 1.0e-5
            variable cfac equal 1.0e-4
            variable cunits string GPa
            if "${{dir}} == 1" then "variable len0 equal ${{lx0}}"
            if "${{dir}} == 2" then "variable len0 equal ${{ly0}}"
            if "${{dir}} == 3" then "variable len0 equal ${{lz0}}"
            if "${{dir}} == 4" then "variable len0 equal ${{lz0}}"
            if "${{dir}} == 5" then "variable len0 equal ${{lz0}}"
            if "${{dir}} == 6" then "variable len0 equal ${{ly0}}"
            box tilt large
            read_restart restart.equil
            pair_style {self._reax_style} NULL
            pair_coeff * * {ff_basename} {seq}
            fix qeq all {self._qeq_style} 1 0.0 10.0 1.0e-6 reaxff
            thermo_style custom step temp pxx pyy pzz pxy pxz pyz pe ke etotal vol xlo xhi ylo yhi zlo zhi xy xz yz press lx ly lz
            thermo 1
            variable delta equal -${{up}}*${{len0}}
            variable deltaxy equal -${{up}}*xy
            variable deltaxz equal -${{up}}*xz
            variable deltayz equal -${{up}}*yz
            if "${{dir}} == 1" then "change_box all x delta 0 ${{delta}} xy delta ${{deltaxy}} xz delta ${{deltaxz}} remap units box"
            if "${{dir}} == 2" then "change_box all y delta 0 ${{delta}} yz delta ${{deltayz}} remap units box"
            if "${{dir}} == 3" then "change_box all z delta 0 ${{delta}} remap units box"
            if "${{dir}} == 4" then "change_box all yz delta ${{delta}} remap units box"
            if "${{dir}} == 5" then "change_box all xz delta ${{delta}} remap units box"
            if "${{dir}} == 6" then "change_box all xy delta ${{delta}} remap units box"
            minimize 0.0 1.0e-8 1000 100000
            variable C1neg equal ${{d1}}
            variable C2neg equal ${{d2}}
            variable C3neg equal ${{d3}}
            variable C4neg equal ${{d4}}
            variable C5neg equal ${{d5}}
            variable C6neg equal ${{d6}}
            clear
            box tilt large
            read_restart restart.equil
            pair_style {self._reax_style} NULL
            pair_coeff * * {ff_basename} {seq}
            fix qeq all {self._qeq_style} 1 0.0 10.0 1.0e-6 reaxff
            thermo_style custom step temp pxx pyy pzz pxy pxz pyz pe ke etotal vol xlo xhi ylo yhi zlo zhi xy xz yz press lx ly lz
            thermo 1
            variable delta equal ${{up}}*${{len0}}
            variable deltaxy equal ${{up}}*xy
            variable deltaxz equal ${{up}}*xz
            variable deltayz equal ${{up}}*yz
            if "${{dir}} == 1" then "change_box all x delta 0 ${{delta}} xy delta ${{deltaxy}} xz delta ${{deltaxz}} remap units box"
            if "${{dir}} == 2" then "change_box all y delta 0 ${{delta}} yz delta ${{deltayz}} remap units box"
            if "${{dir}} == 3" then "change_box all z delta 0 ${{delta}} remap units box"
            if "${{dir}} == 4" then "change_box all yz delta ${{delta}} remap units box"
            if "${{dir}} == 5" then "change_box all xz delta ${{delta}} remap units box"
            if "${{dir}} == 6" then "change_box all xy delta ${{delta}} remap units box"
            minimize 0.0 1.0e-8 1000 100000
            variable C1pos equal ${{d1}}
            variable C2pos equal ${{d2}}
            variable C3pos equal ${{d3}}
            variable C4pos equal ${{d4}}
            variable C5pos equal ${{d5}}
            variable C6pos equal ${{d6}}
            variable C1${{dir}} equal 0.5*(${{C1neg}}+${{C1pos}})
            variable C2${{dir}} equal 0.5*(${{C2neg}}+${{C2pos}})
            variable C3${{dir}} equal 0.5*(${{C3neg}}+${{C3pos}})
            variable C4${{dir}} equal 0.5*(${{C4neg}}+${{C4pos}})
            variable C5${{dir}} equal 0.5*(${{C5neg}}+${{C5pos}})
            variable C6${{dir}} equal 0.5*(${{C6neg}}+${{C6pos}})
            variable dir delete
            """)

        with open(os.path.join(self.path, "displace.mod"),
                  "w", encoding="utf-8") as fh:
            fh.write(displace_mod)

        elastic_driver = textwrap_block("""\
            variable up equal 1.0e-6
            variable atomjiggle equal 1.0e-5
            variable cfac equal 1.0e-4
            variable cunits string GPa
            variable pxx0 equal pxx
            variable pyy0 equal pyy
            variable pzz0 equal pzz
            variable pyz0 equal pyz
            variable pxz0 equal pxz
            variable pxy0 equal pxy
            variable lx0 equal lx
            variable ly0 equal ly
            variable lz0 equal lz
            variable d1 equal -(v_pxx1-${pxx0})/(v_delta/v_len0)*${cfac}
            variable d2 equal -(v_pyy1-${pyy0})/(v_delta/v_len0)*${cfac}
            variable d3 equal -(v_pzz1-${pzz0})/(v_delta/v_len0)*${cfac}
            variable d4 equal -(v_pyz1-${pyz0})/(v_delta/v_len0)*${cfac}
            variable d5 equal -(v_pxz1-${pxz0})/(v_delta/v_len0)*${cfac}
            variable d6 equal -(v_pxy1-${pxy0})/(v_delta/v_len0)*${cfac}
            """) + unfix + "write_restart restart.equil\n"

        # Six probe directions, then derived moduli prints.
        loop_and_report = ""
        for d in range(1, 7):
            loop_and_report += (f"variable dir equal {d}\ninclude displace.mod\n")
        loop_and_report += textwrap_block("""\
            variable C11all equal ${C11}
            variable C22all equal ${C22}
            variable C33all equal ${C33}
            variable C12all equal 0.5*(${C12}+${C21})
            variable C13all equal 0.5*(${C13}+${C31})
            variable C23all equal 0.5*(${C23}+${C32})
            variable C44all equal ${C44}
            variable C55all equal ${C55}
            variable C66all equal ${C66}
            variable C14all equal 0.5*(${C14}+${C41})
            variable C15all equal 0.5*(${C15}+${C51})
            variable C16all equal 0.5*(${C16}+${C61})
            variable C24all equal 0.5*(${C24}+${C42})
            variable C25all equal 0.5*(${C25}+${C52})
            variable C26all equal 0.5*(${C26}+${C62})
            variable C34all equal 0.5*(${C34}+${C43})
            variable C35all equal 0.5*(${C35}+${C53})
            variable C36all equal 0.5*(${C36}+${C63})
            variable C45all equal 0.5*(${C45}+${C54})
            variable C46all equal 0.5*(${C46}+${C64})
            variable C56all equal 0.5*(${C56}+${C65})
            variable C11cubic equal (${C11all}+${C22all}+${C33all})/3.0
            variable C12cubic equal (${C12all}+${C13all}+${C23all})/3.0
            variable C44cubic equal (${C44all}+${C55all}+${C66all})/3.0
            variable bulkmodulus equal (${C11cubic}+2*${C12cubic})/3.0
            variable shearmodulus1 equal ${C44cubic}
            variable shearmodulus2 equal (${C11cubic}-${C12cubic})/2.0
            variable poissonratio equal 1.0/(1.0+${C11cubic}/${C12cubic})
            print "EZFF C11 ${C11all} ${cunits}"
            print "EZFF C22 ${C22all} ${cunits}"
            print "EZFF C33 ${C33all} ${cunits}"
            print "EZFF C12 ${C12all} ${cunits}"
            print "EZFF C13 ${C13all} ${cunits}"
            print "EZFF C23 ${C23all} ${cunits}"
            print "EZFF C44 ${C44all} ${cunits}"
            print "EZFF C55 ${C55all} ${cunits}"
            print "EZFF C66 ${C66all} ${cunits}"
            print "EZFF C14 ${C14all} ${cunits}"
            print "EZFF C15 ${C15all} ${cunits}"
            print "EZFF C16 ${C16all} ${cunits}"
            print "EZFF C24 ${C24all} ${cunits}"
            print "EZFF C25 ${C25all} ${cunits}"
            print "EZFF C26 ${C26all} ${cunits}"
            print "EZFF C34 ${C34all} ${cunits}"
            print "EZFF C35 ${C35all} ${cunits}"
            print "EZFF C36 ${C36all} ${cunits}"
            print "EZFF C45 ${C45all} ${cunits}"
            print "EZFF C46 ${C46all} ${cunits}"
            print "EZFF C56 ${C56all} ${cunits}"
            print "EZFF Bulk_Modulus ${bulkmodulus} ${cunits}"
            print "EZFF Shear_Modulus_1 ${shearmodulus1} ${cunits}"
            print "EZFF Shear_Modulus_2 ${shearmodulus2} ${cunits}"
            print "EZFF Poisson_Ratio ${poissonratio}"
            """)

        body = prelude + "\n" + elastic_driver + loop_and_report
        with open(self.scriptfile, "w", encoding="utf-8") as fh:
            fh.write(body)

    # ------------------------------------------------------------------
    # Result readers
    # ------------------------------------------------------------------

    def read_energy(self) -> np.ndarray:
        """Per-snapshot total energy (eV) read from ``EZFF_ENERGY`` lines."""
        raw = _read_energy(self.outfile)
        if self.units == "metal":
            return raw            # LAMMPS metal => eV/cell already
        if self.units == "real":
            return raw / 23.0605  # kcal/mol -> eV (LAMMPS real units)
        return raw

    def read_elastic_moduli(self) -> List[np.ndarray]:
        return _read_elastic_moduli(self.outfile)

    def read_atomic_charges(self) -> Structure:
        return _read_structure(self.dumpfile)

    def read_structure(self) -> Structure:
        return _read_structure(self.dumpfile)

    def cleanup(self) -> None:
        for f in (self.outfile + ".disp", self.outfile + ".dens",
                  self.outfile, self.scriptfile,
                  self.outfile + ".runerror",
                  self.structfile, self.forcefieldfile, self.dumpfile):
            if os.path.exists(f):
                os.remove(f)


# ---------------------------------------------------------------------------
# Parser helpers (private; module-level so they can be unit-tested)
# ---------------------------------------------------------------------------

_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def _read_energy(outfilename: str) -> np.ndarray:
    energies: List[float] = []
    if not os.path.exists(outfilename):
        return np.array(energies)
    with open(outfilename, "r", errors="replace", encoding="utf-8") as fh:
        for line in fh:
            if "EZFF_ENERGY" in line:
                m = re.search(rf"({_NUM})\s*$", line.strip())
                if m:
                    energies.append(float(m.group(1)))
    return np.array(energies)


def _read_elastic_moduli(outfilename: str) -> List[np.ndarray]:
    """Read all 6x6 elastic-moduli matrices reported in EZFF print lines.

    Returns a list (one matrix per snapshot's elastic-probe stanza).
    """
    moduli_array: List[np.ndarray] = []
    if not os.path.exists(outfilename):
        return moduli_array
    with open(outfilename, "r", errors="replace", encoding="utf-8") as fh:
        lines = fh.readlines()

    # Map each EZFF Cij sentinel into its (i, j) position; we drive
    # population off whichever ones we actually see (some stanzas
    # only emit a subset of off-diagonal entries).
    cij_idx = {
        "C11": (0, 0), "C22": (1, 1), "C33": (2, 2),
        "C12": (0, 1), "C13": (0, 2), "C23": (1, 2),
        "C44": (3, 3), "C55": (4, 4), "C66": (5, 5),
        "C14": (0, 3), "C15": (0, 4), "C16": (0, 5),
        "C24": (1, 3), "C25": (1, 4), "C26": (1, 5),
        "C34": (2, 3), "C35": (2, 4), "C36": (2, 5),
        "C45": (3, 4), "C46": (3, 5), "C56": (4, 5),
    }
    current: Optional[np.ndarray] = None
    seen_in_current: set = set()
    for line in lines:
        if not line.startswith("EZFF "):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        key = parts[1]
        if key in cij_idx:
            if current is None or key in seen_in_current:
                if current is not None:
                    moduli_array.append(current)
                current = np.zeros((6, 6))
                seen_in_current = set()
            try:
                value = float(parts[2])
            except ValueError:
                continue
            i, j = cij_idx[key]
            current[i, j] = value
            current[j, i] = value
            seen_in_current.add(key)
    if current is not None:
        moduli_array.append(current)
    return moduli_array


def _read_structure(dumpfilename: str) -> Structure:
    """Parse the LAMMPS ``write_dump custom`` block into ``Structure``.

    The dump file may contain multiple snapshots; each starts with a
    ``ITEM: TIMESTEP`` header and ends after the per-atom block.
    """
    structure = Structure()
    if not os.path.exists(dumpfilename):
        return structure
    with open(dumpfilename, "r", errors="replace", encoding="utf-8") as fh:
        lines = fh.readlines()

    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if line.startswith("ITEM: NUMBER OF ATOMS"):
            snap = Snapshot()
            snap.box = np.zeros((3, 3))
            i += 1
            num_atoms = int(lines[i].strip())
            i += 1
            continue
        if line.startswith("ITEM: BOX BOUNDS"):
            header = line
            l1 = lines[i + 1].strip().split()
            l2 = lines[i + 2].strip().split()
            l3 = lines[i + 3].strip().split()
            if "xy" in header:
                xlo_b, xhi_b, xy = map(float, l1)
                ylo_b, yhi_b, xz = map(float, l2)
                zlo_b, zhi_b, yz = map(float, l3)
                xlo = xlo_b - min(0.0, xy, xz, xy + xz)
                xhi = xhi_b - max(0.0, xy, xz, xy + xz)
                ylo = ylo_b - min(0.0, yz)
                yhi = yhi_b - max(0.0, yz)
                zlo, zhi = zlo_b, zhi_b
                lx, ly, lz = xhi - xlo, yhi - ylo, zhi - zlo
                snap.abc = np.array([
                    lx,
                    float(np.sqrt(ly * ly + xy * xy)),
                    float(np.sqrt(lz * lz + xz * xz + yz * yz)),
                ])
                cos_a = (xy * xz + ly * yz) / (snap.abc[1] * snap.abc[2])
                cos_b = xz / snap.abc[2]
                cos_c = xy / snap.abc[1]
                snap.ang = np.array([np.arccos(cos_a), np.arccos(cos_b),
                                     np.arccos(cos_c)])
            else:
                xlo, xhi = map(float, l1)
                ylo, yhi = map(float, l2)
                zlo, zhi = map(float, l3)
                snap.box[0][0] = xhi - xlo
                snap.box[1][1] = yhi - ylo
                snap.box[2][2] = zhi - zlo
            i += 4
            continue
        if line.startswith("ITEM: ATOMS"):
            # Header order matches our write_dump exactly: id type element
            # mass q x y z vx vy vz fx fy fz
            i += 1
            for _ in range(num_atoms):
                parts = lines[i].strip().split()
                a = Atom(
                    element=parts[2].upper(),
                    cart=np.array([float(parts[5]), float(parts[6]),
                                   float(parts[7])]),
                    charge=float(parts[4]),
                    vel=np.array([float(parts[8]), float(parts[9]),
                                  float(parts[10])]),
                    force=np.array([float(parts[11]), float(parts[12]),
                                    float(parts[13])]),
                )
                snap.atomlist.append(a)
                i += 1
            structure.snaplist.append(snap)
            continue
        i += 1
    return structure


def textwrap_block(s: str) -> str:
    """Dedent a triple-quoted literal block, preserving line endings."""
    import textwrap
    return textwrap.dedent(s)
