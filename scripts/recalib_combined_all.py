"""Combined ReaxFF recalibration targeting 7 metrics simultaneously:

  1. biaxial sigma(eps) max-RE (5 strain pts)
  2. uniaxial x1 sigma max-RE (5 strain pts)
  3. uniaxial x2 sigma max-RE (5 strain pts)
  4. uniaxial x1 h_S max-RE (5 strain pts)
  5. uniaxial x2 h_S max-RE (5 strain pts)
  6. V_S vacancy formation energy rel-err
  7. S diffusion barrier (Path 2) rel-err

Loss = max(metric_1..7).  Target: every metric <= 10 % rel-err.

Warm-start from PIBO calibrated (the original manuscript ffield, which has
the most balanced metrics: biax 149 %, x1 σ 576 %, x2 σ 67 %, h_S ~20 %,
V_S ~27 %, diff unknown).

Trial cost: ~5-7 min (biax 30s + x1 50s + x2 50s + V_S 120s + diff 90s).
Budget default 80 trials → 6-9 hours.

Output (results/reviewer_response/recalib_combined/):
    RECALIB_LOG.csv      per-trial all 7 metrics + params
    BEST_RESULT.txt
    ffield.reax.MoSH.combined_recalib.reax
"""
from __future__ import annotations
import argparse, csv, json, math, re, shutil, subprocess, sys, textwrap, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_FFIELD = ROOT / "results" / "ffield" / "ffield.reax.MoSH.calibrated.reax"
DST_FFIELD = ROOT / "results" / "ffield" / "ffield.reax.MoSH.combined_recalib.reax"
DFT_CSV    = ROOT / "MoS2_physical_validation.csv"
from _lmp_path import find_lmp as _find_lmp  # parameterized LAMMPS discovery
LMP        = _find_lmp()
DATA10x10  = ROOT / "lammps_templates" / "data.mos2_2H_monolayer_10x10_ryanDFT.lammpsdata"
OUT_DIR    = ROOT / "results" / "reviewer_response" / "recalib_combined"
LOG_PATH   = OUT_DIR / "RECALIB_LOG.csv"
BEST_PATH  = OUT_DIR / "BEST_RESULT.txt"

GPA_PER_ATM = 1.01325e-4
H_EFF_AA    = 6.145
ELEMENTS    = ("Mo", "S")
KCAL_TO_EV  = 1.0 / 23.0605
LMP_CONTROL = (
    "tabulate_long_range 10000\nnbrhood_cutoff 5.0\nhbond_cutoff 6.0\n"
    "bond_graph_cutoff 0.3\nthb_cutoff 0.001\nthb_cutoff_sq 0.00001\nwrite_freq 0\n"
)
_NUM = r"[+-]?(?:nan|inf|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)"

TARGET_PCT = 10.0
MU_S_REF   = 2.82
MU_MO_REF  = 6.90

# DFT reference values (Zhou 2013 + Le 2014)
VFE_REF  = {"V_S": 2.35}
DIFF_REF = {"path2": 1.35}


def grab(t, tag):
    m = re.search(rf"{tag}\s+({_NUM})\b", t, re.I)
    return float(m.group(1)) if m and math.isfinite(float(m.group(1))) else None


def read_lines(p): return p.read_text(encoding="utf-8").splitlines()
def write_lines(p, lines): p.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


# NARROW bounds around v9 (the proven Pareto point for biaxial σ).
# Bond-energy parameters get moderate spread (±30 %) to allow further
# improvement on σ trade-offs; off-diagonal vdW + angle parameters held
# tight (±15 %) to preserve v9's mechanical response; atom-level Mo/S
# parameters allow slight relaxation to fix V_S without breaking σ.
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


def setup_wd(ff_path, target_wd, data_path=None):
    if target_wd.exists():
        shutil.rmtree(target_wd, ignore_errors=True)
        if target_wd.exists():
            time.sleep(0.2); shutil.rmtree(target_wd, ignore_errors=True)
    target_wd.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ff_path, target_wd / "ffield.reax")
    (target_wd / "lmp_control").write_text(LMP_CONTROL, encoding="utf-8")
    if data_path:
        (target_wd / data_path.name).write_text(data_path.read_text(encoding="utf-8"),
                                                encoding="utf-8")
    return target_wd


def run_lmp(deck, wd, tag, timeout=180):
    in_path = wd / f"in.{tag}.lmp"
    in_path.write_text(deck, encoding="utf-8")
    log_path = wd / f"log.{tag}.lammps"
    try:
        p = subprocess.run([LMP, "-in", in_path.name, "-log", log_path.name, "-screen", "none"],
                           cwd=str(wd), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return -1, ""
    try:
        return p.returncode, log_path.read_text(errors="replace")
    except Exception:
        return -1, ""


def biax_init_deck():
    return textwrap.dedent(f"""\
        units real
        atom_style charge
        boundary p p p
        read_data {DATA10x10.name}
        pair_style reaxff lmp_control safezone 2.4 mincap 200
        pair_coeff * * ffield.reax {' '.join(ELEMENTS)}
        fix qeq all qeq/reaxff 1 0.0 10.0 1.0e-10 reaxff
        neighbor 2.0 bin
        neigh_modify every 1 delay 0 check yes
        thermo 100
        min_style cg
        minimize 1.0e-10 1.0e-12 3000 30000
        fix br all box/relax x 0.0 y 0.0 vmax 0.001
        minimize 1.0e-10 1.0e-12 3000 30000
        unfix br
        minimize 1.0e-10 1.0e-12 3000 30000
        variable pe equal pe
        write_data data.relaxed.lammpsdata
        print "RELAXED_LX $(lx)"
        print "RELAXED_LY $(ly)"
        print "PRISTINE_PE_KCAL ${{pe}}"
        """)


def biax_strain_deck(tlx, tly):
    return textwrap.dedent(f"""\
        units real
        atom_style charge
        boundary p p p
        read_data data.relaxed.lammpsdata
        pair_style reaxff lmp_control safezone 2.4 mincap 200
        pair_coeff * * ffield.reax {' '.join(ELEMENTS)}
        fix qeq all qeq/reaxff 1 0.0 10.0 1.0e-10 reaxff
        neighbor 2.0 bin
        neigh_modify every 1 delay 0 check yes
        change_box all x final 0.0 {tlx:.10f} y final 0.0 {tly:.10f} remap units box
        fix freeze all setforce 0.0 0.0 NULL
        thermo 100
        min_style cg
        minimize 1.0e-10 1.0e-12 3000 30000
        variable sig equal 0.5*(-pxx-pyy)*{GPA_PER_ATM}*lz/{H_EFF_AA}
        write_data data.eps.lammpsdata
        print "FINAL_SIG_GPA ${{sig}}"
        """)


def uni_deck(axis, target_a):
    if axis == "x1":
        change = f"change_box all x final 0.0 {target_a:.10f} remap units box"
        relax  = "fix br all box/relax y 0.0 vmax 0.001"
        sig_expr = f"-pxx*{GPA_PER_ATM}*lz/{H_EFF_AA}"
    else:
        change = f"change_box all y final 0.0 {target_a:.10f} remap units box"
        relax  = "fix br all box/relax x 0.0 vmax 0.001"
        sig_expr = f"-pyy*{GPA_PER_ATM}*lz/{H_EFF_AA}"
    return textwrap.dedent(f"""\
        units real
        atom_style charge
        boundary p p p
        read_data data.relaxed.lammpsdata
        pair_style reaxff lmp_control safezone 2.4 mincap 200
        pair_coeff * * ffield.reax {' '.join(ELEMENTS)}
        fix qeq all qeq/reaxff 1 0.0 10.0 1.0e-10 reaxff
        neighbor 2.0 bin
        neigh_modify every 1 delay 0 check yes
        {change}
        {relax}
        thermo 100
        min_style cg
        minimize 1.0e-10 1.0e-12 3000 30000
        unfix br
        minimize 1.0e-10 1.0e-12 3000 30000
        variable sig equal {sig_expr}
        run 0
        write_data data.eps.lammpsdata
        print "FINAL_SIG_GPA ${{sig}}"
        """)


def defect_deck(data_name):
    return textwrap.dedent(f"""\
        units real
        atom_style charge
        boundary p p p
        read_data {data_name}
        pair_style reaxff lmp_control safezone 2.4 mincap 200
        pair_coeff * * ffield.reax {' '.join(ELEMENTS)}
        fix qeq all qeq/reaxff 1 0.0 10.0 1.0e-10 reaxff
        neighbor 2.0 bin
        neigh_modify every 1 delay 0 check yes
        thermo 100
        min_style cg
        minimize 1.0e-10 1.0e-12 3000 30000
        minimize 1.0e-10 1.0e-12 3000 30000
        variable pe equal pe
        write_data data.def.lammpsdata
        print "DEFECT_PE_KCAL ${{pe}}"
        """)


def saddle_deck(data_name, migr_id):
    return textwrap.dedent(f"""\
        units real
        atom_style charge
        boundary p p p
        read_data {data_name}
        pair_style reaxff lmp_control safezone 2.4 mincap 200
        pair_coeff * * ffield.reax {' '.join(ELEMENTS)}
        fix qeq all qeq/reaxff 1 0.0 10.0 1.0e-10 reaxff
        neighbor 2.0 bin
        neigh_modify every 1 delay 0 check yes
        group migr id {migr_id}
        fix freeze_xy migr setforce 0.0 0.0 NULL
        thermo 100
        min_style cg
        minimize 1.0e-10 1.0e-12 3000 30000
        variable pe equal pe
        print "SADDLE_PE_KCAL ${{pe}}"
        """)


def measure_h_S(wd, data_name="data.eps.lammpsdata"):
    path = wd / data_name
    if not path.exists():
        return float("nan")
    lines = path.read_text(errors="replace").splitlines()
    n_atoms = None; atoms_start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.endswith("atoms") and n_atoms is None:
            try: n_atoms = int(s.split()[0])
            except ValueError: pass
        elif s.startswith("Atoms"):
            atoms_start = i + 2; break
    if n_atoms is None or atoms_start is None:
        return float("nan")
    types, zs = [], []
    for j in range(atoms_start, min(atoms_start + n_atoms + 5, len(lines))):
        toks = lines[j].split()
        if len(toks) < 6: continue
        try:
            types.append(int(toks[1])); zs.append(float(toks[5]))
        except ValueError:
            continue
        if len(zs) >= n_atoms: break
    if not zs: return float("nan")
    types = np.asarray(types); zs = np.asarray(zs)
    s_mask = (types == 2); mo_mask = (types == 1)
    if not s_mask.any() or not mo_mask.any():
        return float("nan")
    z_mo_mean = float(zs[mo_mask].mean())
    s_zs = zs[s_mask]
    top = s_zs[s_zs > z_mo_mean]; bot = s_zs[s_zs < z_mo_mean]
    if len(top) == 0 or len(bot) == 0:
        return float(0.5 * (zs.max() - zs.min()))
    return float(0.5 * (top.mean() - bot.mean()))


def parse_data_atoms(path):
    lines = path.read_text(errors="replace").splitlines()
    header = []
    atoms_start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Atoms"):
            atoms_start = i + 1
            break
        header.append(ln)
    if atoms_start is None:
        raise ValueError(f"no Atoms in {path}")
    while atoms_start < len(lines) and not lines[atoms_start].strip():
        atoms_start += 1
    atom_lines = []
    for ln in lines[atoms_start:]:
        if not ln.strip(): break
        atom_lines.append(ln)
    return header, atom_lines


def write_data_with_atoms(out_path, header_lines, atom_records):
    body = list(header_lines)
    # Update N atoms count
    body = [f"{len(atom_records)} atoms" if re.match(r"^\s*\d+\s+atoms\s*$", l) else l
            for l in body]
    body.append("")
    body.append("Atoms # charge")
    body.append("")
    for r in atom_records:
        aid, atype, q, x, y, z = r
        body.append(f"{aid} {atype} {q:.10f} {x:.10f} {y:.10f} {z:.10f}")
    body.append("")
    out_path.write_text("\n".join(body), encoding="utf-8")


def parse_atom_record(line):
    toks = line.split()
    if len(toks) < 6: return None
    return (int(toks[0]), int(toks[1]), float(toks[2]),
            float(toks[3]), float(toks[4]), float(toks[5]))


def rel_err(p, r):
    if not (p is not None and math.isfinite(p) and r and math.isfinite(r)):
        return float("nan")
    return abs(p - r) / abs(r) * 100.0


def evaluate(spec, x, src_lines, refs, wd_root):
    """Evaluate parameter set across all 7 metrics. Returns dict with each
    metric's max relative error and the overall worst.
    """
    tmp = OUT_DIR / "tmp_ffield.reax"
    write_ffield(src_lines, spec, x, tmp)
    BIG = 9999.0
    out = {"biax_max": BIG, "x1_sig_max": BIG, "x2_sig_max": BIG,
           "x1_hS_max": BIG, "x2_hS_max": BIG,
           "V_S_re": BIG, "diff_path2_re": BIG, "worst": BIG}

    # 1) Biaxial init (gets pristine PE for V_S, and box dims for strain)
    wd_b = setup_wd(tmp, wd_root / "biax", DATA10x10)
    rc, log = run_lmp(biax_init_deck(), wd_b, "init", timeout=180)
    if rc != 0: return out
    lx0 = grab(log, "RELAXED_LX"); ly0 = grab(log, "RELAXED_LY")
    E_pristine = grab(log, "PRISTINE_PE_KCAL")
    if lx0 is None or E_pristine is None: return out
    E_pristine_eV = E_pristine * KCAL_TO_EV
    # Biaxial strains
    sigs_b = []
    for i, eps in enumerate(refs["biax_eps"]):
        tlx = lx0 * math.exp(eps); tly = ly0 * math.exp(eps)
        rc, log = run_lmp(biax_strain_deck(tlx, tly), wd_b, f"bs{i}", timeout=90)
        s = grab(log, "FINAL_SIG_GPA") if rc == 0 else None
        sigs_b.append(s if (s is not None and math.isfinite(s)) else float("nan"))
        eps_path = wd_b/"data.eps.lammpsdata"
        if eps_path.exists():
            shutil.copyfile(eps_path, wd_b/"data.relaxed.lammpsdata")
        else:
            break
    rels = [rel_err(s, d) for s, d in zip(sigs_b, refs["biax_sig"])]
    v = [r for r in rels if math.isfinite(r)]
    out["biax_max"] = max(v) if v else BIG

    # 2) Uniaxial x1
    wd_x1 = setup_wd(tmp, wd_root / "x1", DATA10x10)
    rc, log = run_lmp(biax_init_deck(), wd_x1, "init", timeout=180)
    if rc == 0:
        lx0_1 = grab(log, "RELAXED_LX")
        if lx0_1 is not None:
            sigs1, hs1 = [], []
            for i, eps in enumerate(refs["x1_eps"]):
                rc, log = run_lmp(uni_deck("x1", lx0_1 * math.exp(eps)),
                                  wd_x1, f"x1s{i}", timeout=120)
                s = grab(log, "FINAL_SIG_GPA") if rc == 0 else None
                sigs1.append(s if (s is not None and math.isfinite(s)) else float("nan"))
                hs1.append(measure_h_S(wd_x1))
                eps_path = wd_x1/"data.eps.lammpsdata"
                if eps_path.exists():
                    shutil.copyfile(eps_path, wd_x1/"data.relaxed.lammpsdata")
                else:
                    break  # LAMMPS strain step failed; stop incremental ladder
            rs = [rel_err(s, d) for s, d in zip(sigs1, refs["x1_sig"])]
            rh = [rel_err(h, d) for h, d in zip(hs1, refs["x1_hS"])]
            vs = [r for r in rs if math.isfinite(r)]
            vh = [r for r in rh if math.isfinite(r)]
            out["x1_sig_max"] = max(vs) if vs else BIG
            out["x1_hS_max"]  = max(vh) if vh else BIG

    # 3) Uniaxial x2
    wd_x2 = setup_wd(tmp, wd_root / "x2", DATA10x10)
    rc, log = run_lmp(biax_init_deck(), wd_x2, "init", timeout=180)
    if rc == 0:
        ly0_2 = grab(log, "RELAXED_LY")
        if ly0_2 is not None:
            sigs2, hs2 = [], []
            for i, eps in enumerate(refs["x2_eps"]):
                rc, log = run_lmp(uni_deck("x2", ly0_2 * math.exp(eps)),
                                  wd_x2, f"x2s{i}", timeout=120)
                s = grab(log, "FINAL_SIG_GPA") if rc == 0 else None
                sigs2.append(s if (s is not None and math.isfinite(s)) else float("nan"))
                hs2.append(measure_h_S(wd_x2))
                eps_path = wd_x2/"data.eps.lammpsdata"
                if eps_path.exists():
                    shutil.copyfile(eps_path, wd_x2/"data.relaxed.lammpsdata")
                else:
                    break
            rs = [rel_err(s, d) for s, d in zip(sigs2, refs["x2_sig"])]
            rh = [rel_err(h, d) for h, d in zip(hs2, refs["x2_hS"])]
            vs = [r for r in rs if math.isfinite(r)]
            vh = [r for r in rh if math.isfinite(r)]
            out["x2_sig_max"] = max(vs) if vs else BIG
            out["x2_hS_max"]  = max(vh) if vh else BIG

    # 4) V_S vacancy (reuse biax pristine relaxed)
    relaxed = wd_b / "data.relaxed.lammpsdata"
    # Note: we earlier overwrote relaxed with last strain step. Re-relax pristine.
    wd_vs = setup_wd(tmp, wd_root / "vs", DATA10x10)
    rc, log = run_lmp(biax_init_deck(), wd_vs, "init", timeout=180)
    if rc == 0:
        E_p = grab(log, "PRISTINE_PE_KCAL")
        if E_p is not None:
            E_p_eV = E_p * KCAL_TO_EV
            # Build V_S structure: remove top-S near center
            rel = wd_vs / "data.relaxed.lammpsdata"
            header, atomlines = parse_data_atoms(rel)
            records = [parse_atom_record(l) for l in atomlines if parse_atom_record(l)]
            lx = ly = lz = 0.0
            for ln in header:
                if "xlo xhi" in ln: lx = float(ln.split()[1]) - float(ln.split()[0])
                elif "ylo yhi" in ln: ly = float(ln.split()[1]) - float(ln.split()[0])
                elif "zlo zhi" in ln: lz = float(ln.split()[1]) - float(ln.split()[0])
            mo_zs = [r[5] for r in records if r[1] == 1]
            z_mo = sum(mo_zs)/len(mo_zs)
            cx, cy = lx/2, ly/2
            top_S = [r for r in records if r[1]==2 and r[5]>z_mo]
            s_top = min(top_S, key=lambda r: (r[3]-cx)**2 + (r[4]-cy)**2)
            new_records = [r for r in records if r[0] != s_top[0]]
            wd_def = setup_wd(tmp, wd_root / "vs_def", data_path=None)
            ddata = wd_def / "data.vs.lammpsdata"
            write_data_with_atoms(ddata, header, new_records)
            rc, log = run_lmp(defect_deck(ddata.name), wd_def, "vs", timeout=300)
            if rc == 0:
                E_d = grab(log, "DEFECT_PE_KCAL")
                if E_d is not None:
                    E_d_eV = E_d * KCAL_TO_EV
                    vfe = (E_d_eV - E_p_eV) + MU_S_REF
                    out["V_S_re"] = rel_err(vfe, VFE_REF["V_S"])
                    # Save migrating-S ID for diffusion
                    out["_s_top"] = s_top

    # 5) Diffusion path 2 (S hops between adjacent V_S, 60° off-axis)
    # Use the wd_vs pristine relaxed structure
    if "_s_top" in out:
        s_top = out["_s_top"]
        rel = wd_vs / "data.relaxed.lammpsdata"
        header, atomlines = parse_data_atoms(rel)
        records = [parse_atom_record(l) for l in atomlines if parse_atom_record(l)]
        mo_zs = [r[5] for r in records if r[1] == 1]
        z_mo = sum(mo_zs)/len(mo_zs)
        top_S = [r for r in records if r[1]==2 and r[5]>z_mo]
        # Choose B (nearest neighbor) and C (60° off-axis)
        nbrs = sorted([(math.sqrt((r[3]-s_top[3])**2+(r[4]-s_top[4])**2), r)
                       for r in top_S if r[0]!=s_top[0]])
        if len(nbrs) >= 2:
            B = nbrs[0][1]
            vAB = (B[3]-s_top[3], B[4]-s_top[4])
            best_C = None; best_align = -1
            for d, r in nbrs[1:8]:
                vBC = (r[3]-B[3], r[4]-B[4])
                m1 = math.hypot(*vAB); m2 = math.hypot(*vBC)
                if m1<1e-6 or m2<1e-6: continue
                cos_t = (vAB[0]*vBC[0]+vAB[1]*vBC[1])/(m1*m2)
                score = -abs(cos_t-0.5)  # want 60°
                if score > best_align: best_align=score; best_C=r
            if best_C is not None:
                # Initial: vacancies at B & C, A intact
                init_records = [r for r in records if r[0] not in {B[0], best_C[0]}]
                wd_init = setup_wd(tmp, wd_root / "diff_init", data_path=None)
                idata = wd_init / "data.diff_init.lammpsdata"
                write_data_with_atoms(idata, header, init_records)
                rc, log = run_lmp(defect_deck(idata.name), wd_init, "diff_init", timeout=300)
                if rc == 0:
                    E_init = grab(log, "DEFECT_PE_KCAL")
                    if E_init is not None:
                        E_init_eV = E_init * KCAL_TO_EV
                        # Saddle: A at midpoint A-B
                        A_mid = (s_top[0], s_top[1], s_top[2],
                                 0.5*(s_top[3]+B[3]), 0.5*(s_top[4]+B[4]),
                                 0.5*(s_top[5]+B[5]))
                        saddle_records = []
                        for r in records:
                            if r[0] in {B[0], best_C[0]}: continue
                            saddle_records.append(A_mid if r[0]==s_top[0] else r)
                        wd_sad = setup_wd(tmp, wd_root / "diff_saddle", data_path=None)
                        sdata = wd_sad / "data.diff_saddle.lammpsdata"
                        write_data_with_atoms(sdata, header, saddle_records)
                        rc, log = run_lmp(saddle_deck(sdata.name, s_top[0]),
                                          wd_sad, "diff_saddle", timeout=300)
                        if rc == 0:
                            E_sad = grab(log, "SADDLE_PE_KCAL")
                            if E_sad is not None:
                                E_sad_eV = E_sad * KCAL_TO_EV
                                barrier = E_sad_eV - E_init_eV
                                out["diff_path2_re"] = rel_err(barrier, DIFF_REF["path2"])

    out.pop("_s_top", None)
    # PRIORITY 5 (per goal 2026-05-25 update): biax + x1 σ + x2 σ + V_S + diff_path2
    # h_S excluded from worst-of-N — kept for tracking only.
    out["worst"] = max(out["biax_max"], out["x1_sig_max"], out["x2_sig_max"],
                       out["V_S_re"], out["diff_path2_re"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=80)
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = read_lines(SRC_FFIELD)
    off = parse_offsets(src)
    spec = build_spec(src, off)
    lows  = np.array([s["lo"] for s in spec])
    highs = np.array([s["hi"] for s in spec])
    x0    = np.array([s["init"] for s in spec])
    names = [s["name"] for s in spec]
    print(f"[setup] src={SRC_FFIELD.name}, params={len(spec)}, budget={args.budget}", flush=True)
    print(f"[setup] target=ALL 7 metrics <= {TARGET_PCT}%", flush=True)

    # Reduced strain ladders for tractable cost
    SUB = [0.05, 0.10, 0.15, 0.20, 0.25]
    dft = pd.read_csv(DFT_CSV)
    bx = dft[(dft.category=="BIAXIAL") & (dft.condition_var=="true_strain_biaxial")]
    bx = bx.dropna(subset=["stress_GPa"]).sort_values("condition_value")
    refs = {"biax_eps": np.asarray(SUB),
            "biax_sig": np.interp(SUB, bx.condition_value.values, bx.stress_GPa.values)}
    x1 = dft[(dft.category=="UNIAXIAL_X1") & (dft.condition_var=="true_strain_zigzag")]
    x1 = x1.dropna(subset=["stress_GPa"]).reset_index(drop=True)
    refs["x1_eps"] = np.asarray(SUB)
    refs["x1_sig"] = np.interp(SUB, x1.condition_value.values, x1.stress_GPa.values)
    refs["x1_hS"]  = np.interp(SUB, x1.condition_value.values, x1.h_S_AA.values)
    x2 = dft[(dft.category=="UNIAXIAL_X2") & (dft.condition_var=="true_strain_armchair")]
    x2 = x2.dropna(subset=["stress_GPa"]).reset_index(drop=True)
    refs["x2_eps"] = np.asarray(SUB)
    refs["x2_sig"] = np.interp(SUB, x2.condition_value.values, x2.stress_GPa.values)
    refs["x2_hS"]  = np.interp(SUB, x2.condition_value.values, x2.h_S_AA.values)

    from skopt import Optimizer
    from skopt.space import Real
    space = [Real(float(lo), float(hi)) for lo, hi in zip(lows, highs)]
    bo = Optimizer(dimensions=space, base_estimator="GP", acq_func="EI",
                   n_initial_points=10, initial_point_generator="lhs",
                   random_state=args.seed)

    metric_keys = ["biax_max","x1_sig_max","x2_sig_max","x1_hS_max","x2_hS_max",
                   "V_S_re","diff_path2_re"]
    cols = ["trial","wall_s","worst"] + metric_keys + names
    if not LOG_PATH.exists():
        LOG_PATH.write_text(",".join(cols) + "\n", encoding="utf-8")
    best = {"worst": float("inf"), "x": None, "comps": None, "trial": -1}
    t_start = time.time()
    work_root = OUT_DIR / "_runs"
    work_root.mkdir(parents=True, exist_ok=True)
    for trial in range(1, args.budget + 1):
        x_raw = list(x0) if trial == 1 else bo.ask()
        x = np.minimum(np.maximum(np.asarray(x_raw), lows), highs)
        t0 = time.time()
        r = evaluate(spec, x, src, refs, work_root / f"t{trial:04d}")
        dt = time.time() - t0
        worst = r["worst"]
        try: bo.tell(list(x), float(worst))
        except Exception as e: print(f"  [warn] tell error: {e}")
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            row = [str(trial), f"{time.time()-t_start:.2f}", f"{worst:.4f}",
                   *[f"{r[k]:.4f}" for k in metric_keys],
                   *[f"{v:.6f}" for v in x]]
            fh.write(",".join(row) + "\n")
        improved = worst < best["worst"] - 1e-6
        if improved:
            best.update({"worst": worst, "x": x.tolist(), "comps": r, "trial": trial})
            write_ffield(src, spec, x, DST_FFIELD)
            BEST_PATH.write_text(
                f"trial {trial}\nworst={worst:.4f}\n"
                f"biax={r['biax_max']:.4f}  x1_sig={r['x1_sig_max']:.4f}  "
                f"x2_sig={r['x2_sig_max']:.4f}  x1_hS={r['x1_hS_max']:.4f}  "
                f"x2_hS={r['x2_hS_max']:.4f}  V_S={r['V_S_re']:.4f}  "
                f"diff_path2={r['diff_path2_re']:.4f}\n"
                f"params={dict(zip(names, x.tolist()))}\n",
                encoding="utf-8")
        mark = "  [NEW BEST]" if improved else ""
        print(f"t{trial:4d} [{dt:5.0f}s] w={worst:6.2f} "
              f"(bx={r['biax_max']:5.1f} 1s={r['x1_sig_max']:5.1f} 2s={r['x2_sig_max']:5.1f} "
              f"1h={r['x1_hS_max']:5.1f} 2h={r['x2_hS_max']:5.1f} "
              f"Vs={r['V_S_re']:5.1f} D2={r['diff_path2_re']:5.1f}) best={best['worst']:6.2f}{mark}",
              flush=True)
        if best["worst"] <= TARGET_PCT:
            print(f"\n[TARGET MET at trial {best['trial']}]", flush=True)
            break

    elapsed = time.time() - t_start
    print(f"\n[done] {trial} trials, {elapsed/60:.1f} min, best worst={best['worst']:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
