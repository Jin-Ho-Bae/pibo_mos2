"""Force-field text I/O for the PIBO optimizer.

Reads variable-bounds files and ffield templates, and renders a ReaxFF
``ffield.reax`` from a template by width-preserving substitution of the
optimizer's ``<<NAME>>`` placeholders. The H/Mo/S ffield uses a fixed column
layout, so the substitution preserves column widths.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from pibo_reaxff.lammps_runner import generate_forcefield as _width_preserving_substitute


def read_variable_bounds(filename: str, verbose: bool = False
                         ) -> Dict[str, List[float]]:
    """Read permissible (lo, hi) bounds for each decision variable.

    File format (one variable per line)::

        # comment lines start with '#'
        name           lo            hi
        _int_name      lo            hi

    Names whose first character is ``_`` are read as integers (discrete
    variables); all other names are floats.

    The retry loop guards against an empty read when several processes open
    the same file simultaneously; we re-read until non-empty.
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

    Template convention: every numeric parameter the optimizer is allowed to
    change is replaced by ``<<NAME>>``.
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
    """Render a ffield from a template by width-preserving substitution.

    Parameters
    ----------
    template_string : str
        Body of the ffield template containing ``<<NAME>>`` markers.
    parameters : dict
        ``{name: numeric_value}`` pairs; every key that appears in
        ``template_string`` is substituted.
    FFtype : str, optional
        ``'REAXFF'`` for the LAMMPS-format ReaxFF ffield used here.
    outfile : str, optional
        If given, write the rendered ffield here and return ``None``;
        otherwise return the rendered string.
    MD : str
        ``'LAMMPS'`` (default). GULP rewriting is not supported here.
    """
    rendered = _width_preserving_substitute(template_string, parameters)

    if FFtype is not None:
        ff = FFtype.strip().upper()
        if ff in ("REAX", "REAXFF") and MD.upper() == "GULP":
            raise NotImplementedError(
                "generate_forcefield(FFtype='REAXFF', MD='GULP') is not "
                "supported here. Use MD='LAMMPS'.")

    if outfile is not None:
        with open(outfile, "w", encoding="utf-8") as fh:
            fh.write(rendered)
        return None
    return rendered
