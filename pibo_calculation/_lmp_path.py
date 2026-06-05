"""Locate a ReaxFF-enabled LAMMPS executable for the recalibration drivers.

Replaces the per-machine hardcoded ``lmp.exe`` paths that the original
recalibration scripts carried. Resolution order:

1. ``$PIBO_LMP`` environment variable (an explicit path to the binary).
2. A binary named ``lmp`` / ``lmp.exe`` / ``lmp_serial`` / ``lammps`` on PATH.

Raises ``FileNotFoundError`` with a clear message if none is found, rather
than silently falling back — the recalibration cannot run without LAMMPS.
"""
from __future__ import annotations
import os
import shutil


def find_lmp() -> str:
    env = os.environ.get("PIBO_LMP")
    if env:
        if os.path.isfile(env):
            return env
        raise FileNotFoundError(
            f"$PIBO_LMP is set to {env!r} but that file does not exist.")
    for name in ("lmp", "lmp.exe", "lmp_serial", "lmp_mpi", "lammps"):
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError(
        "No LAMMPS executable found. Set the PIBO_LMP environment variable "
        "to your ReaxFF-enabled LAMMPS binary, e.g.\n"
        "    Windows:  set PIBO_LMP=C:\\path\\to\\lmp.exe\n"
        "    Linux/Mac: export PIBO_LMP=/path/to/lmp\n"
        "or put `lmp` on your PATH (conda install -c conda-forge lammps).")
