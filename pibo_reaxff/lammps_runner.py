"""
LAMMPS ReaxFF runner — cement-notebook conventions for the H/Mo/S system.

Driving model
-------------
LAMMPS is invoked via the **`lmp` CLI binary** (typically `/usr/local/bin/lmp`
or the conda-forge prefix at `/opt/lmp_env/bin/lmp`). The Python module
`from lammps import lammps` is *not* used — it lives inside the isolated
conda env that `pibo_reaxff.colab_setup` installs and shadowing it into
Colab's system kernel is brittle.

LAMMPS-only — no surrogate fallback
-----------------------------------
Every energy evaluation goes through the ``lmp`` CLI. If LAMMPS is unavailable
or rejects REAXFF, the runner raises immediately rather than substituting a
cheap analytic model: the project policy is that **all energies must come
from a LAMMPS MD/static evaluation**.

ReaxFF/LAMMPS conventions adopted from ``ReaxFF_Cement_Optimization.ipynb``
Cells 4.1 / 4.3:

1. **Pair style auto-detection** via ``detect_reaxff_style()`` — picks
   ``reaxff`` on modern LAMMPS builds and falls back to legacy ``reax/c``.
   Hard-coding ``reaxff`` would silent-fail on older builds.

2. **`lmp_control` companion file** with ReaxFF cutoffs (``nbrhood_cutoff``,
   ``hbond_cutoff``, ``thb_cutoff``, etc.) so the pair style sees the same
   tabulation/H-bond/three-body cutoffs the manuscript used.

3. **`FINAL_EPA_EV` sentinels + kcal→eV unit conversion** (×0.0433641),
   plus full lattice (a, b, c, lx, ly, lz) and pressure prints. The
   parser uses a strict number regex (rejects bare ``-`` or partial
   captures), discards non-finite or sanity-violating values
   (|val| > 1e5 eV anywhere, |epa| > 50 eV/atom), and falls back to
   thermo ``PotEng`` if ``FINAL_EPA_EV`` was not printed.

4. **Width-preserving ``<<placeholder>>`` substitution** in the ffield
   template (matches cement Cell 4.1 ``generate_forcefield``) so the
   strict van Duin column layout never shifts under value changes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .parameters import default_general_params
from .vasp_reader import DFTFrame


# ---------------------------------------------------------------------------
# LAMMPS binary discovery + REAXFF capability probe
# ---------------------------------------------------------------------------

_DEFAULT_LMP_PATHS = ("/usr/local/bin/lmp", "/opt/lmp_env/bin/lmp")


def _find_lmp_binary() -> Optional[str]:
    p = shutil.which("lmp")
    if p:
        return p
    for cand in _DEFAULT_LMP_PATHS:
        if os.path.exists(cand):
            return cand
    return None


def detect_reaxff_style(lmp_path: Optional[str] = None
                        ) -> Tuple[str, str]:
    """Return ``(pair_style, qeq_fix)`` matching this LAMMPS build.

    Modern LAMMPS (≥ 2021) ships ``reaxff`` + ``qeq/reaxff``. Older builds
    only have legacy ``reax/c`` + ``qeq/reax/c``. We probe by sending a
    minimal script through the binary; if the modern style is unrecognized
    we fall back to legacy. Mirrors cement Cell 4.3 helper of the same name.
    """
    lmp_path = lmp_path or _find_lmp_binary()
    if not lmp_path or not os.path.exists(lmp_path):
        raise RuntimeError(
            "LAMMPS binary not found on PATH or at /usr/local/bin/lmp / "
            "/opt/lmp_env/bin/lmp. All energies must be computed by LAMMPS — "
            "install via pibo_reaxff.colab_setup.install_lammps_reaxff().")
    probe = (
        "units real\natom_style charge\n"
        "region box block 0 10 0 10 0 10\n"
        "create_box 1 box\nmass 1 16.0\n"
        "pair_style reaxff NULL\n"
    )
    try:
        proc = subprocess.run([lmp_path], input=probe, capture_output=True,
                              text=True, timeout=15)
        out = (proc.stdout or "") + (proc.stderr or "")
    except Exception:
        return ("reaxff", "qeq/reaxff")
    bad = ("Unrecognized pair style", "Unknown pair style",
           "is part of the REAXFF package which is not enabled")
    if any(b in out for b in bad):
        return ("reax/c", "qeq/reax/c")
    return ("reaxff", "qeq/reaxff")


def _lammps_supports_reaxff(lmp_path: Optional[str]) -> bool:
    if not lmp_path or not os.path.exists(lmp_path):
        return False
    pair, _ = detect_reaxff_style(lmp_path)
    # detect_reaxff_style falls back to ('reaxff', ...) when the binary
    # rejects modern style; verify legacy works too by re-probing.
    if pair == "reaxff":
        return True
    probe = "units real\natom_style charge\npair_style reax/c NULL\n"
    try:
        proc = subprocess.run([lmp_path], input=probe, capture_output=True,
                              text=True, timeout=15)
        out = (proc.stdout or "") + (proc.stderr or "")
    except Exception:
        return False
    return all(b not in out for b in
               ("Unrecognized pair style", "Unknown pair style",
                "is part of the REAXFF package which is not enabled"))


# ---------------------------------------------------------------------------
# Width-preserving placeholder substitution (cement Cell 4.1 generate_forcefield)
# ---------------------------------------------------------------------------

def generate_forcefield(template: str, parameters: Dict[str, float]) -> str:
    """Replace ``<<NAME>>`` placeholders preserving the placeholder's width.

    The ReaxFF ffield format is column-sensitive: a value that overflows its
    placeholder slot would shift every subsequent column and corrupt the
    file. We size the substituted number to the placeholder's literal width;
    if the value is too wide for that slot at 4 decimals, we progressively
    drop precision and finally fall back to scientific notation.
    """
    result = template
    for name, value in parameters.items():
        ph = f"<<{name}>>"
        if ph not in result:
            continue
        width = len(ph)
        formatted = f"{value:{width}.4f}"
        if len(formatted) > width:
            for d in (3, 2, 1, 0):
                formatted = f"{value:{width}.{d}f}"
                if len(formatted) <= width:
                    break
            else:
                formatted = f"{value:{width}.2e}"
                if len(formatted) > width:
                    formatted = formatted[:width]
        result = result.replace(ph, formatted)
    return result


# ---------------------------------------------------------------------------
# LAMMPS output parsing — strict regex + sanity clamps + thermo fallback
# ---------------------------------------------------------------------------

# Strict number regex — never matches a bare '-' or trailing 'e'. Accepts
# IEEE specials (nan, inf) so LAMMPS divergence is detected instead of
# silently miscaptured.
_LMP_NUMBER = (
    r"[+-]?"
    r"(?:nan|inf|"
    r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|"
    r"\.\d+(?:[eE][+-]?\d+)?)"
)

_LMP_FIELDS = (
    ("energy_per_atom_eV", "FINAL_EPA_EV"),
    ("pressure_atm",       "FINAL_PRESS"),
    ("lat_a",              "FINAL_LAT_A"),
    ("lat_b",              "FINAL_LAT_B"),
    ("lat_c",              "FINAL_LAT_C"),
    ("lx",                 "FINAL_LX"),
    ("ly",                 "FINAL_LY"),
    ("lz",                 "FINAL_LZ"),
)

_LMP_PATTERNS = {
    key: re.compile(rf"{tag}\s+({_LMP_NUMBER})\b", re.IGNORECASE)
    for key, tag in _LMP_FIELDS
}

KCAL_PER_MOL_TO_EV = 0.0433641


def parse_lammps(stdout: str) -> Dict[str, float]:
    """Extract structured outputs from a LAMMPS stdout buffer.

    Returns at least ``energy_per_atom_eV`` on success (possibly via thermo
    fallback). ``_lmp_parse_diag`` always populated for offline inspection.
    Garbage / divergent values are dropped silently; caller treats missing
    keys as evaluation failure.
    """
    if not stdout:
        return {}
    result: Dict[str, float] = {}
    diag: Dict[str, list] = {"finite": [], "nonfinite": [],
                             "malformed": [], "blowup": []}

    for key, pat in _LMP_PATTERNS.items():
        m = pat.search(stdout)
        if not m:
            continue
        raw = m.group(1)
        try:
            val = float(raw)
        except (ValueError, TypeError):
            diag["malformed"].append((key, raw))
            continue
        if not np.isfinite(val):
            diag["nonfinite"].append((key, raw))
            continue
        if abs(val) > 1e5 or (key == "energy_per_atom_eV" and abs(val) > 50.0):
            diag["blowup"].append((key, val))
            continue
        result[key] = val
        diag["finite"].append((key, val))

    # Thermo fallback for energy_per_atom_eV
    if "energy_per_atom_eV" not in result:
        for i, line in enumerate(stdout.split("\n")):
            if line.strip().startswith("Step") and "PotEng" in line:
                headers = line.strip().split()
                pe_col = next((c for c, h in enumerate(headers)
                               if h == "PotEng"), None)
                if pe_col is None:
                    break
                for j in range(i + 1, min(i + 5, len(stdout.split("\n")))):
                    parts = stdout.split("\n")[j].strip().split()
                    if (len(parts) > pe_col
                            and parts[0].lstrip("-").isdigit()):
                        try:
                            pe_kcal = float(parts[pe_col])
                        except ValueError:
                            break
                        m_atoms = re.search(r"(\d+)\s+atoms", stdout)
                        if m_atoms:
                            n_at = int(m_atoms.group(1))
                            epa_ev = pe_kcal * KCAL_PER_MOL_TO_EV / n_at
                            if np.isfinite(epa_ev) and abs(epa_ev) <= 50.0:
                                result["energy_per_atom_eV"] = epa_ev
                                diag["finite"].append(
                                    ("energy_per_atom_eV (thermo fallback)",
                                     epa_ev))
                        break
                break

    if all(k in result for k in ("lat_a", "lat_b", "lat_c")):
        result["lattice"] = {"a": result["lat_a"], "b": result["lat_b"],
                             "c": result["lat_c"]}
    result["_lmp_parse_diag"] = diag
    return result


# ---------------------------------------------------------------------------
# `lmp_control` and LAMMPS input script writers (cement Cell 4.3 mirror)
# ---------------------------------------------------------------------------

def write_lammps_control(work_dir: str) -> str:
    """Write the ReaxFF control file used by ``pair_style reaxff lmp_control``.

    Cutoffs match the manuscript's MD parameterization
    (§ "Computational Methods", QEq tolerance 1e-6, nbrhood/H-bond cutoffs
    matching Ostadhossein 2017). Returns the basename ``"lmp_control"``.
    """
    body = (
        "tabulate_long_range 10000\n"
        "nbrhood_cutoff 5.0\n"
        "hbond_cutoff 6.0\n"
        "bond_graph_cutoff 0.3\n"
        "thb_cutoff 0.001\n"
        "thb_cutoff_sq 0.00001\n"
        "write_freq 0\n"
    )
    path = os.path.join(work_dir, "lmp_control")
    with open(path, "w") as f:
        f.write(body)
    return "lmp_control"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    energy: float           # eV (per cell — kcal→eV converted; per-atom × natoms)
    forces: Optional[np.ndarray]
    converged: bool
    runtime_s: float
    raw: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# LAMMPSRunner
# ---------------------------------------------------------------------------

class LAMMPSRunner:
    """Thin wrapper around the ``lmp`` CLI binary. LAMMPS-only — raises if absent."""

    DEFAULT_MASSES: Dict[str, float] = {
        "Mo": 95.94, "S": 32.06, "H": 1.008,
        "Ca": 40.078, "Si": 28.0855, "O": 15.9994, "Al": 26.9815,
    }

    def __init__(self,
                 mode: str = "lammps",
                 elements: Tuple[str, ...] = ("Mo", "S", "H"),
                 base_ffield: Optional[str] = None,
                 lmp_path: Optional[str] = None,
                 cleanup_tmp: bool = True,
                 timeout_s: float = 120.0):
        """
        Parameters
        ----------
        mode : str
            Accepted for backwards compatibility but ignored — the runner
            always uses real LAMMPS. A missing or REAXFF-less binary raises
            immediately; there is no surrogate fallback.
        elements : tuple of element symbols, in atom-type order (1, 2, 3, ...)
            For H/Mo/S the default ``("Mo", "S", "H")`` matches the
            ``ffield.reax.MoSH.template`` atom-block ordering.
        base_ffield : path to ``ffield.reax.<system>.template`` with
            ``<<PARAM>>`` placeholders. Required.
        lmp_path : override LAMMPS binary autodetection.
        cleanup_tmp : delete the per-instance scratch dir in ``close()``.
        timeout_s : per-evaluation timeout.
        """
        del mode  # accepted-but-ignored; LAMMPS is the only path
        self.elements = tuple(elements)
        self.base_ffield = base_ffield
        self.lmp_path = lmp_path or _find_lmp_binary()
        if not self.lmp_path:
            raise RuntimeError(
                "LAMMPS binary not found. Install via "
                "pibo_reaxff.colab_setup.install_lammps_reaxff() — all "
                "energies must be computed by LAMMPS.")
        if not _lammps_supports_reaxff(self.lmp_path):
            raise RuntimeError(
                f"LAMMPS at {self.lmp_path} does not support REAXFF. "
                "Reinstall with a REAXFF-enabled build (conda-forge::lammps "
                "or source build with PKG_REAXFF=ON).")
        self.cleanup_tmp = cleanup_tmp
        self.timeout_s = timeout_s
        self.mode = "lammps"
        self._tmpdir = tempfile.mkdtemp(prefix="pibo_lmp_")
        self.general = default_general_params()
        self._reax_style, self._qeq_style = detect_reaxff_style(self.lmp_path)

    def close(self) -> None:
        if self.cleanup_tmp and self._tmpdir and os.path.isdir(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ---- ffield writer ---------------------------------------------------

    def write_ffield(self, params: Dict[str, float], path: str) -> None:
        """Render the ffield template with optimizer-supplied parameters.

        Uses ``generate_forcefield`` (width-preserving). Raises if the
        template path was not configured or if any ``<<>>`` placeholder
        remains unresolved after substitution.
        """
        if not self.base_ffield:
            raise RuntimeError(
                "LAMMPSRunner.base_ffield is unset. Provide a "
                "ffield.reax.<sys>.template with <<PARAM>> placeholders.")
        # Force utf-8 — Windows defaults to cp949/cp1252 which trips on
        # any non-ASCII char in the template (em-dash, en-dash, accents).
        with open(self.base_ffield, "r", encoding="utf-8") as f:
            template = f.read()
        rendered = generate_forcefield(template, params)
        if "<<" in rendered:
            unresolved = re.findall(r"<<\w+>>", rendered)
            raise RuntimeError(
                f"Unresolved placeholders remain in ffield: {unresolved[:5]}")
        with open(path, "w") as f:
            f.write(rendered)

    # ---- evaluation entry point -----------------------------------------

    def evaluate(self, params: Dict[str, float], frame: DFTFrame
                 ) -> EvaluationResult:
        t0 = time.time()
        e, fc, raw = self._evaluate_lammps(params, frame)
        return EvaluationResult(e, fc, True, time.time() - t0, raw)

    # ---- LAMMPS CLI invocation ------------------------------------------

    def _evaluate_lammps(self, params: Dict[str, float], frame: DFTFrame
                         ) -> Tuple[float, Optional[np.ndarray],
                                    Dict[str, float]]:
        ffield_path = os.path.join(self._tmpdir, "ffield.reax")
        data_path = os.path.join(self._tmpdir, "structure.data")
        in_path = os.path.join(self._tmpdir, "in.lmp")

        self.write_ffield(params, ffield_path)
        self._write_lammps_data(frame, data_path)
        self._write_input_script(in_path, data_path, ffield_path)

        log_path = os.path.join(self._tmpdir, "log.lammps")
        proc = subprocess.run(
            [self.lmp_path, "-in", in_path, "-screen", "none",
             "-log", log_path],
            cwd=self._tmpdir,
            capture_output=True, text=True, timeout=self.timeout_s,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"lmp exit={proc.returncode}; stderr tail: "
                f"{(proc.stderr or '')[-300:]}")

        # LAMMPS `print` writes to screen+log by default; `-screen none`
        # silences stdout. Always parse the log file — it is the canonical
        # destination across every LAMMPS version and -screen setting.
        try:
            with open(log_path, "r", errors="replace") as f:
                lammps_text = f.read()
        except OSError as exc:
            raise RuntimeError(
                f"LAMMPS log {log_path} unreadable: {exc}; "
                f"stdout tail: {(proc.stdout or '')[-300:]}")

        parsed = parse_lammps(lammps_text)
        if "energy_per_atom_eV" not in parsed:
            tail_lines = [l for l in lammps_text.split("\n") if "FINAL_" in l]
            err_lines  = [l for l in lammps_text.split("\n")
                          if "ERROR" in l or "WARNING" in l]
            tail = "\n".join(tail_lines[:8]) or "(no FINAL_* lines)"
            errs = "\n".join(err_lines[-6:]) or "(no ERROR/WARNING lines)"
            raise RuntimeError(
                "Could not parse energy from LAMMPS log "
                f"{log_path}.\n"
                f"FINAL_* lines:\n{tail}\n"
                f"ERROR/WARNING tail:\n{errs}\n"
                f"Log tail (last 600 chars):\n{lammps_text[-600:]}")

        # Convert per-atom eV back to per-cell eV (loss compares totals).
        epa_ev = parsed["energy_per_atom_eV"]
        e_total = epa_ev * float(frame.n_atoms)
        return e_total, None, parsed

    def _write_input_script(self, path: str, data_path: str,
                            ffield_path: str) -> None:
        ctrl_name = write_lammps_control(self._tmpdir)
        elem_str = " ".join(self.elements)
        mass_lines = "\n".join(
            f"mass {i+1} {self.DEFAULT_MASSES.get(e, 1.0):.4f}"
            for i, e in enumerate(self.elements))

        ff_basename = os.path.basename(ffield_path)
        data_basename = os.path.basename(data_path)

        # Cement-style single-point evaluation script. ${...} pieces inside
        # the script are LAMMPS variables — we double-brace them in the
        # f-string ($${{var}} → ${var}) so Python doesn't try to interpolate.
        script = textwrap.dedent(f"""\
            units real
            atom_style charge
            boundary p p p
            read_data {data_basename}
            {mass_lines}
            pair_style {self._reax_style} {ctrl_name} safezone 2.4 mincap 200
            pair_coeff * * {ff_basename} {elem_str}
            fix qeq all {self._qeq_style} 1 0.0 10.0 1.0e-6 reaxff
            neighbor 2.0 bin
            neigh_modify every 1 delay 0 check yes
            thermo 1
            thermo_style custom step pe press vol lx ly lz xy xz yz
            run 0
            variable natoms equal count(all)
            variable pe_ev equal pe*{KCAL_PER_MOL_TO_EV}
            variable epa_ev equal v_pe_ev/v_natoms
            variable my_lx equal lx
            variable my_ly equal ly
            variable my_lz equal lz
            variable my_xy equal xy
            variable my_xz equal xz
            variable my_yz equal yz
            variable lat_a equal v_my_lx
            variable lat_b equal sqrt(v_my_ly*v_my_ly+v_my_xy*v_my_xy)
            variable lat_c equal sqrt(v_my_lz*v_my_lz+v_my_xz*v_my_xz+v_my_yz*v_my_yz)
            print "FINAL_EPA_EV ${{epa_ev}}"
            print "FINAL_PRESS $(press)"
            print "FINAL_LAT_A ${{lat_a}}"
            print "FINAL_LAT_B ${{lat_b}}"
            print "FINAL_LAT_C ${{lat_c}}"
            print "FINAL_LX $(lx)"
            print "FINAL_LY $(ly)"
            print "FINAL_LZ $(lz)"
            """)
        with open(path, "w") as f:
            f.write(script)

    # ---- data file writer (LAMMPS ``charge`` atom style) ----------------

    def _write_lammps_data(self, frame: DFTFrame, path: str) -> None:
        elem_to_id = {e: i + 1 for i, e in enumerate(self.elements)}
        with open(path, "w") as f:
            f.write("ReaxFF data (PIBO H/Mo/S)\n\n")
            f.write(f"{frame.n_atoms} atoms\n")
            f.write(f"{len(self.elements)} atom types\n\n")
            a, b, c = frame.cell
            xhi = float(np.linalg.norm(a))
            yhi = float(np.linalg.norm(b))
            zhi = float(np.linalg.norm(c))
            f.write(f"0.0 {xhi:.6f} xlo xhi\n")
            f.write(f"0.0 {yhi:.6f} ylo yhi\n")
            f.write(f"0.0 {zhi:.6f} zlo zhi\n\n")
            f.write("Masses\n\n")
            for el, idx in elem_to_id.items():
                f.write(f"{idx} {self.DEFAULT_MASSES.get(el, 1.0):.4f}\n")
            f.write("\nAtoms\n\n")
            for i, (sp, p) in enumerate(zip(frame.species, frame.positions),
                                        start=1):
                t = elem_to_id.get(sp, 1)
                f.write(f"{i} {t} 0.0 {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
