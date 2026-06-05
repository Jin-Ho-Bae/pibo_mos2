"""
EZFF-format file I/O port.

Mirrors the public surface of ``ezff/ffio.py`` from arvk/EZFF so this
project can read EZFF-style ``variable_bounds`` text files and ffield
templates without depending on the EZFF package. PIBO remains the
optimizer; this module is purely format-compatible plumbing.

Mapping to the upstream module:

    EZFF                                ->  here
    -----------------------------------     ------------------------------
    read_variable_bounds(filename)      ->  read_variable_bounds(filename)
    read_forcefield_template(filename)  ->  read_forcefield_template(filename)
    generate_forcefield(template,       ->  generate_forcefield(template,
        parameters, FFtype, outfile,            parameters, FFtype, outfile,
        MD)                                     MD)

The substitution semantics are EZFF's ``<<NAME>>`` placeholders. We keep
the project's *width-preserving* formatter from
``lammps_runner.generate_forcefield`` because the H/Mo/S ffield uses a
fixed column layout — EZFF's plain ``'%12.6f' % value`` substitution
would shift columns on values whose formatted width differs from the
placeholder.

``FFtype='REAXFF'`` + ``MD='LAMMPS'`` is supported as a pass-through:
EZFF's upstream calls ``reax_forcefield(...).write_formatted_forcefields()``
from ``ezff.utils.reaxff``; here our template is already a LAMMPS-formatted
``ffield.reax.*`` so no extra rewrite is needed.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from .lammps_runner import generate_forcefield as _width_preserving_substitute


def read_variable_bounds(filename: str, verbose: bool = False
                         ) -> Dict[str, List[float]]:
    """Read permissible (lo, hi) bounds for each decision variable.

    File format (one variable per line)::

        # comment lines start with '#'
        name           lo            hi
        _int_name      lo            hi

    Names whose first character is ``_`` are read as integers (EZFF
    convention for discrete variables). All other names are floats.

    The retry loop mirrors EZFF's pattern: if multiple MPI ranks open
    the same file simultaneously, an empty read can occur; we re-read
    until non-empty.
    """
    variable_bounds: Dict[str, List[float]] = {}
    while True:
        time.sleep(float(np.random.rand()) * 0.01)
        with open(filename, "r", encoding="utf-8") as fh:
            for line in fh:
                items = line.strip().split()
                if not items or items[0].startswith("#"):
                    continue
                key, values = items[0], items[1:]
                if key.startswith("_"):
                    variable_bounds[key] = list(map(int, values))
                else:
                    variable_bounds[key] = list(map(float, values))
        if variable_bounds:
            break

    if verbose:
        keys = ", ".join(variable_bounds.keys())
        print(f"Keys: {keys} read from {filename}")
    return variable_bounds


def read_forcefield_template(template_filename: str) -> str:
    """Read a ffield template file as a single string.

    Template convention: every numeric parameter that the optimizer is
    allowed to change is replaced by ``<<NAME>>``. The same retry loop
    as EZFF is used for concurrent-access safety.
    """
    while True:
        time.sleep(float(np.random.rand()) * 0.01)
        with open(template_filename, "r", encoding="utf-8") as fh:
            template_string = fh.read()
        if template_string:
            break
    return template_string


def generate_forcefield(template_string: str,
                        parameters: Dict[str, float],
                        FFtype: Optional[str] = None,
                        outfile: Optional[str] = None,
                        MD: str = "LAMMPS") -> Optional[str]:
    """EZFF-compatible signature; width-preserving substitution.

    Parameters
    ----------
    template_string : str
        Body of the ffield template containing ``<<NAME>>`` markers.
    parameters : dict
        ``{name: numeric_value}`` pairs. Every key that appears in
        ``template_string`` is substituted.
    FFtype : str, optional
        Reserved for compatibility with upstream EZFF
        (``'REAXFF'`` / ``'SW'`` / ...). When ``FFtype.upper()`` is
        ``'REAXFF'`` (or ``'REAX'``) and ``MD == 'LAMMPS'`` we treat the
        template as an already-formatted LAMMPS ``ffield.reax.*`` file
        and pass it through unchanged after substitution. This avoids
        EZFF's GULP-library rewrite which would corrupt LAMMPS-format
        files.
    outfile : str, optional
        If given, write the rendered ffield here and return ``None``;
        otherwise return the rendered string.
    MD : str
        ``'LAMMPS'`` (default) or ``'GULP'``. Only ``'LAMMPS'`` is fully
        supported in this project — GULP rewriting requires
        ``ezff.utils.reaxff.reax_forcefield`` which is not vendored here.
    """
    rendered = _width_preserving_substitute(template_string, parameters)

    if FFtype is not None:
        ff = FFtype.strip().upper()
        if ff in ("REAX", "REAXFF") and MD.upper() == "GULP":
            raise NotImplementedError(
                "generate_forcefield(FFtype='REAXFF', MD='GULP') requires "
                "EZFF's reax_forcefield(...).write_gulp_library() helper, "
                "which is not vendored in this project. Use MD='LAMMPS'.")

    if outfile is not None:
        with open(outfile, "w", encoding="utf-8") as fh:
            fh.write(rendered)
        return None
    return rendered
