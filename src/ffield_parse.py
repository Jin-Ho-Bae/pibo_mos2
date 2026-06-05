"""Pure-Python ReaxFF ``ffield.reax`` text parsing helpers.

This module contains ONLY the format-parsing functions needed by the
uncertainty-quantification pipeline (``phase01``–``phase07`` and
``build_parameter_bounds``) to read a calibrated MoS-H ReaxFF force
field and recover the 40-parameter calibration spec (name, value,
per-parameter bounds).

It deliberately carries **no** LAMMPS / VASP / molecular-dynamics code:
the functions here only read and rewrite the ReaxFF text format. They
were factored out of the standalone recalibration driver so the
reproducibility repository ships no validation / LAMMPS-execution code.

Functions
---------
read_lines / write_lines
    Trivial UTF-8 line IO on ``pathlib.Path`` objects.
parse_offsets
    Locate the atom / bond / off-diagonal / angle / torsion blocks in a
    ReaxFF ffield by walking the section counts.
build_spec
    Extract the 40 calibrated parameters (Mo-S bond, off-diagonal vdW,
    S-Mo-S / Mo-S-Mo angle, and Mo/S atom-level rows) together with the
    relative bounds used during calibration.
write_ffield
    Width-preserving substitution of new parameter values back into a
    copy of the template lines.
"""
from __future__ import annotations


def read_lines(p):
    return p.read_text(encoding="utf-8").splitlines()


def write_lines(p, lines):
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_offsets(lines):
    o = {}; i = 1
    n_gen = int(lines[i].split()[0]); i += 1 + n_gen
    n_atoms = int(lines[i].split()[0]); i += 1 + 3
    o["atom_start"] = i; o["n_atoms"] = n_atoms; i += n_atoms * 4
    n_bonds = int(lines[i].split()[0]); i += 1 + 1
    o["bond_start"] = i; o["n_bonds"] = n_bonds; i += n_bonds * 2
    n_off = int(lines[i].split()[0]); i += 1
    o["offdiag_start"] = i; o["n_offdiag"] = n_off; i += n_off
    n_ang = int(lines[i].split()[0]); i += 1
    o["angle_start"] = i; o["n_angle"] = n_ang; i += n_ang
    n_tor = int(lines[i].split()[0]); i += 1
    o["tor_start"] = i; o["n_tor"] = n_tor
    return o


# NARROW bounds around v9 (the proven Pareto point for biaxial sigma).
# Bond-energy parameters get moderate spread (+/-30 %) to allow further
# improvement on sigma trade-offs; off-diagonal vdW + angle parameters held
# tight (+/-15 %) to preserve v9's mechanical response; atom-level Mo/S
# parameters allow slight relaxation.
WIDEN = {
    "De_sigma_MoS":  (0.70, 1.30), "De_pi_MoS":     (0.70, 1.30),
    "p_be1_MoS":     (0.50, 1.70), "p_be2_MoS":     (0.50, 1.70),
    "p_bo1_MoS":     (0.70, 1.35), "p_bo2_MoS":     (0.70, 1.35),
    "p_bo3_MoS":     (0.50, 1.70), "p_bo4_MoS":     (0.60, 1.50),
    "p_bo5_MoS":     (0.60, 1.60), "p_bo6_MoS":     (0.50, 1.70),
    "p_ovun1_MoS":   (0.50, 1.70),
    "Dij_MoS":       (0.85, 1.20), "RvdW_MoS":      (0.94, 1.08),
    "Alfa_MoS":      (0.90, 1.15), "ro_sigma_MoS":  (0.95, 1.07),
    "Thetao_SMoS":   (0.95, 1.07), "Thetao_MoSMo":  (0.92, 1.10),
    "p_val1_SMoS":   (0.80, 1.25), "p_val1_MoSMo":  (0.78, 1.28),
    "p_val2_SMoS":   (0.78, 1.28), "p_val7_SMoS":   (0.78, 1.28),
    "p_val4_SMoS":   (0.78, 1.28),
    "Mo_RvdW":       (0.94, 1.08), "Mo_Dij":        (0.75, 1.30),
    "Mo_gamma":      (0.88, 1.15), "Mo_Alfa":       (0.88, 1.15),
    "Mo_gamma_w":    (0.82, 1.22), "Mo_p_boc4":     (0.65, 1.50),
    "Mo_p_boc3":     (0.55, 1.80), "Mo_p_boc5":     (0.65, 1.50),
    "Mo_p_ovun2":    (0.75, 1.30), "Mo_p_val3":     (0.75, 1.30),
    "Mo_p_val5":     (0.82, 1.22),
    "S_RvdW":        (0.94, 1.06), "S_Dij":         (0.78, 1.25),
    "S_gamma":       (0.88, 1.15), "S_Alfa":        (0.88, 1.15),
    "S_gamma_w":     (0.85, 1.18), "S_p_ovun2":     (0.78, 1.25),
    "S_p_val3":      (0.78, 1.25),
}


def build_spec(lines, off):
    spec = []
    bs = off["bond_start"]; mos_l1, mos_l2 = bs + 2, bs + 3
    p1 = lines[mos_l1].split(); p2 = lines[mos_l2].split()
    for n, c in [("De_sigma_MoS",0),("De_pi_MoS",1),("p_be1_MoS",3),
                 ("p_bo5_MoS",4),("p_bo6_MoS",6),("p_ovun1_MoS",7)]:
        spec.append({"name":n,"line":mos_l1,"col_in_floats":c,"init":float(p1[2+c])})
    for n, c in [("p_be2_MoS",0),("p_bo3_MoS",1),("p_bo4_MoS",2),
                 ("p_bo1_MoS",4),("p_bo2_MoS",5)]:
        try:
            v = float(p2[c])
            if abs(v) < 1e-6 and n in ("p_be2_MoS","p_bo3_MoS","p_bo4_MoS"): continue
            spec.append({"name":n,"line":mos_l2,"col_in_floats":c,"init":v})
        except (IndexError, ValueError):
            pass
    od = off["offdiag_start"]; p3 = lines[od].split()
    for n, c in [("Dij_MoS",0),("RvdW_MoS",1),("Alfa_MoS",2),("ro_sigma_MoS",3)]:
        spec.append({"name":n,"line":od,"col_in_floats":c,"init":float(p3[2+c])})
    ang = off["angle_start"]
    for ai in range(off["n_angle"]):
        ln = ang + ai; toks = lines[ln].split()
        if len(toks) >= 3 and toks[0] == "2" and toks[1] == "1" and toks[2] == "2":
            for n, c in [("Thetao_SMoS",0),("p_val1_SMoS",1),("p_val2_SMoS",2),
                         ("p_val7_SMoS",4),("p_val4_SMoS",6)]:
                spec.append({"name":n,"line":ln,"col_in_floats":c,"init":float(toks[3+c])})
            break
    for ai in range(off["n_angle"]):
        ln = ang + ai; toks = lines[ln].split()
        if len(toks) >= 3 and toks[0] == "1" and toks[1] == "2" and toks[2] == "1":
            for n, c in [("Thetao_MoSMo",0),("p_val1_MoSMo",1)]:
                spec.append({"name":n,"line":ln,"col_in_floats":c,"init":float(toks[3+c])})
            break
    atom_start = off["atom_start"]
    def _parse_floats(ln_idx):
        toks = lines[ln_idx].split()
        start = 1 if toks and toks[0] in ("Mo","S","H","C","O","N") else 0
        floats = []
        for t in toks[start:]:
            try: floats.append(float(t))
            except ValueError: break
        return start, floats
    mo_l1 = atom_start; mo_l2 = atom_start + 1
    mo_l3 = atom_start + 2; mo_l4 = atom_start + 3
    s_l1  = atom_start + 4; s_l2  = atom_start + 5
    s_l4  = atom_start + 7
    _, f = _parse_floats(mo_l1)
    for n, c in [("Mo_RvdW", 3), ("Mo_Dij", 4), ("Mo_gamma", 5)]:
        if c < len(f) and abs(f[c]) > 1e-6:
            spec.append({"name": n, "line": mo_l1, "col_in_floats": c, "init": f[c]})
    _, f = _parse_floats(mo_l2)
    for n, c in [("Mo_Alfa", 0), ("Mo_gamma_w", 1)]:
        if c < len(f) and abs(f[c]) > 1e-6:
            spec.append({"name": n, "line": mo_l2, "col_in_floats": c, "init": f[c]})
    _, f = _parse_floats(mo_l3)
    for n, c in [("Mo_p_boc4", 3), ("Mo_p_boc3", 4), ("Mo_p_boc5", 5)]:
        if c < len(f) and abs(f[c]) > 1e-6:
            spec.append({"name": n, "line": mo_l3, "col_in_floats": c, "init": f[c]})
    _, f = _parse_floats(mo_l4)
    for n, c in [("Mo_p_ovun2", 0), ("Mo_p_val3", 1), ("Mo_p_val5", 4)]:
        if c < len(f) and abs(f[c]) > 1e-6:
            spec.append({"name": n, "line": mo_l4, "col_in_floats": c, "init": f[c]})
    _, f = _parse_floats(s_l1)
    for n, c in [("S_RvdW", 3), ("S_Dij", 4), ("S_gamma", 5)]:
        if c < len(f) and abs(f[c]) > 1e-6:
            spec.append({"name": n, "line": s_l1, "col_in_floats": c, "init": f[c]})
    _, f = _parse_floats(s_l2)
    for n, c in [("S_Alfa", 0), ("S_gamma_w", 1)]:
        if c < len(f) and abs(f[c]) > 1e-6:
            spec.append({"name": n, "line": s_l2, "col_in_floats": c, "init": f[c]})
    _, f = _parse_floats(s_l4)
    for n, c in [("S_p_ovun2", 0), ("S_p_val3", 1)]:
        if c < len(f) and abs(f[c]) > 1e-6:
            spec.append({"name": n, "line": s_l4, "col_in_floats": c, "init": f[c]})
    for s in spec:
        v = s["init"]; sign = -1.0 if v < 0 else 1.0; av = abs(v)
        lo_m, hi_m = WIDEN.get(s["name"], (0.70, 1.40))
        lo_abs = max(1e-6, av * lo_m); hi_abs = av * hi_m
        if sign < 0:
            s["lo"] = -hi_abs; s["hi"] = -lo_abs
        else:
            s["lo"] = lo_abs; s["hi"] = hi_abs
    return spec


def write_ffield(src_lines, spec, overrides, dst):
    lines = list(src_lines)
    by_line = {}
    for s, v in zip(spec, overrides):
        by_line.setdefault(s["line"], []).append((s["col_in_floats"], v))
    for ln_idx, mods in by_line.items():
        orig = lines[ln_idx]; parts = orig.split()
        n_lead = 0
        for tok in parts:
            try: int(tok); n_lead += 1
            except ValueError:
                if tok in ("Mo","S","H","C","O","N"): n_lead += 1
                else: break
        floats = []
        for tok in parts[n_lead:]:
            try: floats.append(float(tok))
            except ValueError: break
        for col, new_val in mods:
            if col < len(floats): floats[col] = new_val
        lead = " " + " ".join(parts[:n_lead])
        body = "".join(f"{f:10.4f}" for f in floats)
        lines[ln_idx] = lead + body
    write_lines(dst, lines)
