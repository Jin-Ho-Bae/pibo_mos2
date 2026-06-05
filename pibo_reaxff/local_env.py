"""
Local-environment LAMMPS detection for PIBO-ReaxFF.

PyCharm Pro / Windows + conda use case: the user has installed LAMMPS via
`conda install -c conda-forge lammps` (or equivalent) and just needs the
package to confirm the binary is on PATH and supports REAXFF.

Differences from `colab_setup.py`:
  - No Drive mount, no micromamba bootstrap, no apt install attempts.
  - No `_install_python_packages()` — pip / conda is the user's job.
  - No fallback "FATAL" raise that kills the kernel; this module raises
    a clear RuntimeError with install instructions instead.

Use from `run.py` (or interactively) as:

    from pibo_reaxff.local_env import ensure_lammps, repo_root
    lmp_path = ensure_lammps()
    print(f"LAMMPS ready at {lmp_path}")
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# Canonical LAMMPS binary names across the conda-forge / system / Windows
# packagings we expect to see. Order matters: prefer plain `lmp`.
_LMP_CANDIDATES = (
    "lmp", "lmp.exe",
    "lmp_serial", "lmp_serial.exe",
    "lmp_mpi", "lmp_mpi.exe",
    "lammps", "lammps.exe",
)


def repo_root() -> Path:
    """Absolute path to the PIBO_ReaxFF repo root.

    Anchored to this file so it is invariant under cwd changes — works the
    same whether you launched PyCharm from the project folder or ran a
    script from elsewhere on disk.
    """
    return Path(__file__).resolve().parent.parent


# Windows native installer (https://packages.lammps.org/windows.html) drops
# `lmp.exe` into a versioned subfolder of Program Files. The conda env's PATH
# does not include that location, so a user with a Windows-native LAMMPS
# install needs us to scan these directories explicitly.
_WIN_LAMMPS_SCAN_ROOTS = (
    r"C:\Program Files",
    r"C:\Program Files (x86)",
    r"C:\LAMMPS",
    r"C:\lammps",
)


def _scan_windows_install_dirs() -> Optional[Path]:
    """Find a Windows-native LAMMPS install under Program Files / C:/LAMMPS."""
    if platform.system() != "Windows":
        return None
    for root in _WIN_LAMMPS_SCAN_ROOTS:
        root_p = Path(root)
        if not root_p.exists():
            continue
        try:
            for sub in root_p.iterdir():
                if not sub.is_dir():
                    continue
                if "lammps" not in sub.name.lower():
                    continue
                # Typical layout: <root>/LAMMPS 64-bit 2024-06-27/bin/lmp.exe
                for cand_name in ("lmp.exe", "lmp_serial.exe",
                                  "lmp_mpi.exe", "lammps.exe"):
                    cand = sub / "bin" / cand_name
                    if cand.exists():
                        return cand
                # Some packagings drop the binary directly into the version dir.
                for cand_name in ("lmp.exe", "lmp_serial.exe"):
                    cand = sub / cand_name
                    if cand.exists():
                        return cand
        except OSError:
            continue
    return None


def find_lammps_binary() -> Optional[Path]:
    """Return a LAMMPS binary by trying, in order:

    1. ``$LAMMPS_BIN`` env-var override.
    2. ``shutil.which`` over the standard binary names (covers conda envs,
       apt installs, anything else on PATH).
    3. Windows native installer locations under ``Program Files`` etc.
    """
    env_path = os.environ.get("LAMMPS_BIN")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    for name in _LMP_CANDIDATES:
        which = shutil.which(name)
        if which:
            return Path(which)

    return _scan_windows_install_dirs()


def lammps_supports_reaxff(lmp_path: Path) -> bool:
    """Probe the binary with a minimal `pair_style reaxff` deck.

    Returns True iff LAMMPS accepts the modern `reaxff` pair style (which
    requires the binary to be built with PKG_REAXFF=ON). Legacy `reax/c`
    only builds are *not* treated as supported here — REAXFF is the path
    PIBO actually uses.
    """
    script = (
        "units real\n"
        "atom_style charge\n"
        "region box block 0 10 0 10 0 10\n"
        "create_box 1 box\n"
        "mass 1 16.0\n"
        "pair_style reaxff NULL\n"
    )
    try:
        proc = subprocess.run(
            [str(lmp_path), "-screen", "none"],
            input=script,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
    bad = (
        "is part of the REAXFF package which is not enabled",
        "Unrecognized pair style",
        "Unknown pair style",
    )
    return not any(b in blob for b in bad)


def _find_conda() -> Optional[str]:
    """Return the conda executable, preferring the env-tied one."""
    # CONDA_EXE is set inside any conda-activated shell; trust it first.
    cexe = os.environ.get("CONDA_EXE")
    if cexe and Path(cexe).exists():
        return cexe
    for name in ("conda.exe", "conda", "mamba.exe", "mamba"):
        w = shutil.which(name)
        if w:
            return w
    return None


def install_lammps_conda(channel: str = "conda-forge",
                         env_name: Optional[str] = None,
                         dry_run: bool = False) -> Path:
    """Install / update LAMMPS into a conda environment via ``conda install``.

    Parameters
    ----------
    channel
        Conda channel to install from. ``conda-forge`` ships PKG_REAXFF=ON
        for every supported platform (win-64, linux-64, osx-64, osx-arm64).
    env_name
        Target conda env. ``None`` (default) installs into the *currently
        active* env, which is the right thing 95 % of the time — that env is
        whatever PyCharm picked as the project interpreter.
    dry_run
        Print the command but do not execute. Useful for inspecting what
        would happen without committing to a long-running install.

    Returns
    -------
    Path
        Absolute path to the freshly-installed LAMMPS binary. Raises
        RuntimeError if conda is unavailable or the install fails.
    """
    conda = _find_conda()
    if conda is None:
        raise RuntimeError(
            "No `conda` (or `mamba`) executable found. Install Miniconda first:\n"
            "    https://docs.anaconda.com/miniconda/\n"
            "Then re-open PyCharm so its terminal picks conda up.")

    cmd = [conda, "install", "-y", "-c", channel, "lammps"]
    if env_name:
        cmd[2:2] = ["-n", env_name]   # `conda install -n <env> -y -c ...`

    print(f"[local_env] running: {' '.join(cmd)}")
    if dry_run:
        return Path("(dry run)")

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-800:].strip()
        raise RuntimeError(
            f"conda install lammps failed (exit={proc.returncode}):\n{tail}\n"
            "Try running the command in your PyCharm terminal manually to see\n"
            "the full output, and check that the env name / channel are correct.")

    # Freshly-installed binary location depends on platform:
    #   * Linux / macOS  : <env>/bin/lmp
    #   * Windows        : <env>/Library/bin/lmp.exe   (conda MSYS layout)
    lmp = find_lammps_binary()
    if lmp is None:
        raise RuntimeError(
            "conda install succeeded but no lmp binary was found on PATH "
            "afterwards. Did the install go into a different env? "
            "Try setting $LAMMPS_BIN explicitly or restart PyCharm so the env "
            "is re-activated.")
    if not lammps_supports_reaxff(lmp):
        raise RuntimeError(
            f"conda install completed and {lmp} exists, but the REAXFF probe "
            "failed. The conda-forge build SHOULD have PKG_REAXFF=ON — please "
            "report this with the output of `conda list lammps`.")
    print(f"[local_env] conda install OK: {lmp}")
    return lmp


def _install_hint() -> str:
    sysname = platform.system()
    if sysname == "Windows":
        return (
            "Install via conda-forge in your PyCharm interpreter env:\n"
            "    conda install -c conda-forge lammps\n"
            "Then restart the PyCharm interpreter so the new PATH is picked up,\n"
            "or set LAMMPS_BIN explicitly:\n"
            "    setx LAMMPS_BIN \"C:\\Users\\<you>\\miniconda3\\envs\\pibo\\Library\\bin\\lmp.exe\""
        )
    if sysname == "Linux":
        return (
            "Install via conda-forge OR apt:\n"
            "    conda install -c conda-forge lammps     # cross-distro\n"
            "    sudo apt-get install -y lammps          # Ubuntu 22.04+"
        )
    if sysname == "Darwin":
        return (
            "Install via conda-forge (works on both Intel and Apple Silicon):\n"
            "    conda install -c conda-forge lammps"
        )
    return "Install LAMMPS with PKG_REAXFF=ON and put `lmp` on PATH."


def ensure_lammps(strict: bool = True,
                  auto_install: bool = False,
                  channel: str = "conda-forge") -> Optional[Path]:
    """Return the LAMMPS binary path, optionally installing it on miss.

    Parameters
    ----------
    strict
        If True (default), raise RuntimeError when LAMMPS is missing or
        REAXFF-less. If False, return None instead.
    auto_install
        If True and no usable LAMMPS is found, attempt
        ``install_lammps_conda(channel)`` and re-check. Off by default — the
        user should opt in via `run.py --install-lammps` or pass it
        explicitly.
    channel
        Conda channel for auto-install.
    """
    lmp = find_lammps_binary()
    if lmp is not None and lammps_supports_reaxff(lmp):
        return lmp

    # Either missing or no REAXFF — same recovery path.
    reason = ("not found on PATH" if lmp is None
              else f"found at {lmp} but lacks REAXFF support")

    if auto_install:
        print(f"[local_env] LAMMPS {reason}; attempting conda install ...")
        try:
            return install_lammps_conda(channel=channel)
        except Exception as exc:
            msg = f"auto_install failed: {exc}"
            if strict:
                raise RuntimeError(msg) from exc
            print(f"[local_env] {msg}", file=sys.stderr)
            return None

    msg = (
        f"LAMMPS {reason}. PIBO-ReaxFF requires LAMMPS for every energy "
        "evaluation (no surrogate fallback).\n\n"
        + _install_hint() +
        "\n\nAuto-install: rerun with `python run.py --install-lammps` or "
        "call `ensure_lammps(auto_install=True)` from Python."
    )
    if strict:
        raise RuntimeError(msg)
    print(f"[local_env] {msg}", file=sys.stderr)
    return None


def sanity_check() -> bool:
    """Print a short status block. Returns True iff LAMMPS+REAXFF is usable."""
    print(f"[local_env] OS: {platform.system()} {platform.release()}")
    print(f"[local_env] Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"[local_env] Repo root: {repo_root()}")

    lmp = find_lammps_binary()
    if lmp is None:
        print("[local_env] LAMMPS: NOT FOUND")
        print(_install_hint())
        return False
    print(f"[local_env] LAMMPS binary: {lmp}")

    ok = lammps_supports_reaxff(lmp)
    print(f"[local_env] REAXFF support: {'OK' if ok else 'MISSING'}")
    if not ok:
        print(_install_hint())
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if sanity_check() else 1)