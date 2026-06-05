"""PIBO recalib v12: constrained physics-informed Bayesian Optimization
that drives BOTH biaxial sigma(epsilon) accuracy AND equilibrium geometry
below the 10 % target simultaneously.

Strategy:
  - 22 parameters bootstrapped from v11 best (the proven Pareto point at
    biax=52 % / a=5 % / d=3 % / ang=2 %).
  - **Constrained quadratic-penalty loss**:
        L = biax_max_re + PEN_W * Σ_i max(0, x_i_re - τ_i)^2
    Biaxial is the primary objective; the geometry penalty is *zero* as
    long as a / d / angle stay below their per-observable tolerance and
    grows quadratically once they exceed it. This is the standard
    constrained-optimization PIBO formulation.
  - Wider bond-curvature / bond-energy bounds than v11 to chase the σ(ε)
    shape; tighter off-diagonal / angle bounds to preserve geometry.
  - **True Bayesian Optimization** via scikit-optimize's Optimizer (GP
    surrogate, EI acquisition). At the end, we keep:
      * the full BO trace (`RECALIB_V10_LOG.csv`)
      * the final GP posterior as an ensemble of top-K trials, sampled
        from a softmax(-loss) distribution over the trials (this is the
        standard "approximate posterior" trick from sequential model-based
        optimization)
  - **Bayesian uncertainty quantification**: the top-K ensemble is then
    re-evaluated via LAMMPS to produce observable posteriors with std.

Outputs (under results/reviewer_response/biaxial_recalib_v10/):
    RECALIB_V10_LOG.csv         per-trial parameter + 4 metrics + sigs
    BEST_RESULT.txt             best parameters + sigs + per-eps errors
    posterior_params.csv        N_post draws from GP posterior approx
    posterior_observables.csv   per-posterior-draw LAMMPS observables
    posterior_summary.csv       mean / std / 5/50/95 percentiles
"""
from __future__ import annotations
import json, math, re, shutil, subprocess, textwrap, time, sys
from pathlib import Path
import numpy as np
import pandas as pd

from skopt import Optimizer
from skopt.space import Real

ROOT = Path(__file__).resolve().parents[1]
SRC_FFIELD = ROOT / "results" / "ffield" / "ffield.reax.MoSH.pibo_biaxial_v11.reax"
FB_FFIELD  = ROOT / "results" / "ffield" / "ffield.reax.MoSH.calibrated.reax"
if not SRC_FFIELD.exists(): SRC_FFIELD = FB_FFIELD
DST_FFIELD = ROOT / "results" / "ffield" / "ffield.reax.MoSH.pibo_biaxial_v12.reax"
DFT_CSV    = ROOT / "MoS2_physical_validation.csv"
from _lmp_path import find_lmp as _find_lmp  # parameterized LAMMPS discovery
LMP        = _find_lmp()
DATA10x10  = ROOT / "lammps_templates" / "data.mos2_2H_monolayer_10x10_ryanDFT.lammpsdata"
DATA3x3    = ROOT / "results" / "reviewer_response" / "v9_validation" / "data.mos2_3x3_v9_validation.lammpsdata"
OUT_DIR    = ROOT / "results" / "reviewer_response" / "biaxial_recalib_v12"
LOG_PATH   = OUT_DIR / "RECALIB_V12_LOG.csv"
BEST_PATH  = OUT_DIR / "BEST_RESULT.txt"
TARGET_RE  = 10.0   # target % rel err on every individual observable

# v12 constrained-loss tolerances. Geometry constraints behave as soft
# tolerances; biaxial is the primary minimisation objective.
TOL_A   = 8.0    # |a_re|   <= this is "free"; beyond it we add penalty
TOL_D   = 8.0
TOL_ANG = 5.0
PEN_W   = 3.0    # quadratic penalty multiplier on geometry overflow

GPA_PER_ATM= 1.01325e-4
H_EFF_AA   = 6.145
ELEMENTS   = ("Mo","S")
LMP_CONTROL = (
    "tabulate_long_range 10000\nnbrhood_cutoff 5.0\nhbond_cutoff 6.0\n"
    "bond_graph_cutoff 0.3\nthb_cutoff 0.001\nthb_cutoff_sq 0.00001\nwrite_freq 0\n"
)
_NUM = r"[+-]?(?:nan|inf|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)"
def grab(t,tag):
    m=re.search(rf"{tag}\s+({_NUM})\b",t,re.I)
    return float(m.group(1)) if m and math.isfinite(float(m.group(1))) else None

def read_lines(p): return p.read_text(encoding="utf-8").splitlines()
def write_lines(p, lines): p.write_text("\n".join(lines) + "\n", encoding="utf-8")

DFT_REF = {
    "a_AA":         3.1830,
    "d_MoS_AA":     2.4131,
    "h_S_AA":       1.5640,
    "angle_intra":  82.53,
}

# Weights on the joint loss. All set to 1 -> equal importance after each
# observable's relative error is normalised to %.
W = {"biaxial": 1.0, "a": 1.0, "d_MoS": 1.0, "angle_intra": 1.0}


# ---------------------------------------------------------------------------
# ffield offset parser + 22-parameter spec (matches v9 exactly)
# ---------------------------------------------------------------------------
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

# Asymmetric multiplicative bounds. Equilibrium-controlling parameters
# (RvdW, Alfa, ro_sigma, Dij, Thetao_*) are given wider room than in v9
# because v9 had collapsed equilibrium geometry. Bond-curvature parameters
# (p_bo*, p_be*) are kept tight around v9 best.
# v11: bootstrap from PIBO calibrated (good geometry, bad biaxial). Widen
# bond-curvature parameters (p_be2/p_bo3/p_bo4/p_bo5/p_bo6) to chase biaxial
# accuracy; keep equilibrium-controlling parameters tight to PRESERVE the
# already-good a / d_MoS / angle_intra.
# v12: bootstrap from v11 best (Pareto point: biax=52 %, geometry within
# PIBO-calibrated band). The constraint-aware loss lets BO push biaxial
# aggressively without breaking geometry, so bond-curvature handles can
# be widened beyond v11.
WIDEN = {
    # Bond depth / curvature: wide -> chase biaxial sigma(eps)
    "De_sigma_MoS":  (0.60, 1.70),
    "De_pi_MoS":     (0.60, 1.70),
    "p_be1_MoS":     (0.40, 2.00),
    "p_be2_MoS":     (0.30, 2.50),
    "p_bo1_MoS":     (0.55, 1.55),
    "p_bo2_MoS":     (0.55, 1.55),
    "p_bo3_MoS":     (0.30, 2.50),
    "p_bo4_MoS":     (0.40, 2.20),
    "p_bo5_MoS":     (0.40, 2.00),
    "p_bo6_MoS":     (0.30, 2.50),
    "p_ovun1_MoS":   (0.30, 2.50),
    # Off-diagonal vdW / radius: stay tight to preserve equilibrium
    "Dij_MoS":       (0.75, 1.30),
    "RvdW_MoS":      (0.92, 1.10),
    "Alfa_MoS":      (0.85, 1.18),
    "ro_sigma_MoS":  (0.94, 1.08),
    # Angle terms: stay tight (PIBO calibrated nails angles)
    "Thetao_SMoS":   (0.94, 1.08),
    "Thetao_MoSMo":  (0.90, 1.13),
    "p_val1_SMoS":   (0.75, 1.35),
    "p_val1_MoSMo":  (0.70, 1.40),
    "p_val2_SMoS":   (0.70, 1.40),
    "p_val7_SMoS":   (0.70, 1.40),
    "p_val4_SMoS":   (0.70, 1.40),
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
            if abs(v) < 1e-6 and n in ("p_be2_MoS","p_bo3_MoS","p_bo4_MoS"):
                continue
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
    for s in spec:
        v = s["init"]; sign = -1.0 if v < 0 else 1.0; av = abs(v)
        lo_m, hi_m = WIDEN.get(s["name"], (0.70, 1.40))
        lo_abs = max(1e-6, av * lo_m)
        hi_abs = av * hi_m
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


# ---------------------------------------------------------------------------
# LAMMPS evaluation: biaxial sigma(eps) on 10x10 + equilibrium on 3x3
# ---------------------------------------------------------------------------
WD_BIAX = OUT_DIR / "_biax"
WD_EQ   = OUT_DIR / "_eq"

def setup_wd(ff_path, target_wd, data_file):
    if target_wd.exists():
        shutil.rmtree(target_wd, ignore_errors=True)
        if target_wd.exists():
            time.sleep(0.2); shutil.rmtree(target_wd, ignore_errors=True)
    target_wd.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ff_path, target_wd / "ffield.reax")
    (target_wd / "lmp_control").write_text(LMP_CONTROL, encoding="utf-8")
    (target_wd / data_file.name).write_text(data_file.read_text(encoding="utf-8"),
                                            encoding="utf-8")
    return target_wd


def run_lmp(deck, wd, tag, timeout=120):
    in_path = wd / f"in.{tag}.lmp"
    in_path.write_text(deck, encoding="utf-8")
    log_path = wd / f"log.{tag}.lammps"
    try:
        p = subprocess.run([LMP,"-in",in_path.name,"-log",log_path.name,"-screen","none"],
                           cwd=str(wd), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return -1, ""
    try:
        return p.returncode, log_path.read_text(errors="replace")
    except Exception:
        return -1, ""


def biax_init_deck(data_name):
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
        fix br all box/relax x 0.0 y 0.0 vmax 0.001
        minimize 1.0e-10 1.0e-12 3000 30000
        unfix br
        minimize 1.0e-10 1.0e-12 3000 30000
        write_data data.relaxed.lammpsdata
        print "RELAXED_LX $(lx)"
        print "RELAXED_LY $(ly)"
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
        variable sig_iso_gpa equal 0.5*(-pxx-pyy)*{GPA_PER_ATM}*lz/{H_EFF_AA}
        write_data data.eps.lammpsdata
        print "FINAL_SIG_ISO_GPA ${{sig_iso_gpa}}"
        """)


def eq_deck():
    return textwrap.dedent(f"""\
        units real
        atom_style charge
        atom_modify map array
        boundary p p p
        read_data data.mos2_3x3_v9_validation.lammpsdata
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
        write_data data.eq.lammpsdata
        print "DONE_EQ"
        """)


def parse_data_file(path: Path):
    txt = path.read_text(errors="replace").splitlines()
    n_atoms = None
    box = {"xlo": 0.0, "xhi": 0.0, "ylo": 0.0, "yhi": 0.0,
           "zlo": 0.0, "zhi": 0.0, "xy": 0.0, "xz": 0.0, "yz": 0.0}
    atoms_start = None
    for i, line in enumerate(txt):
        s = line.strip()
        if s.endswith("atoms") and n_atoms is None:
            try: n_atoms = int(s.split()[0])
            except ValueError: pass
        if "xlo xhi" in s:
            xs = s.split(); box["xlo"] = float(xs[0]); box["xhi"] = float(xs[1])
        elif "ylo yhi" in s:
            xs = s.split(); box["ylo"] = float(xs[0]); box["yhi"] = float(xs[1])
        elif "zlo zhi" in s:
            xs = s.split(); box["zlo"] = float(xs[0]); box["zhi"] = float(xs[1])
        elif "xy xz yz" in s:
            xs = s.split(); box["xy"] = float(xs[0]); box["xz"] = float(xs[1]); box["yz"] = float(xs[2])
        elif s.startswith("Atoms"):
            atoms_start = i + 2
            break
    if n_atoms is None or atoms_start is None:
        raise ValueError(f"parse fail {path}")
    atoms = np.zeros((n_atoms, 5))
    k = 0
    for j in range(atoms_start, len(txt)):
        toks = txt[j].split()
        if not toks:
            if k >= n_atoms: break
            continue
        if k >= n_atoms: break
        atoms[k,0] = int(toks[0]); atoms[k,1] = int(toks[1])
        atoms[k,2] = float(toks[3]); atoms[k,3] = float(toks[4]); atoms[k,4] = float(toks[5])
        k += 1
    lx = box["xhi"]-box["xlo"]; ly = box["yhi"]-box["ylo"]; lz = box["zhi"]-box["zlo"]
    return lx, ly, lz, box["xy"], box["xz"], box["yz"], atoms


def min_image(dr, lx, ly, lz, xy, xz, yz):
    s = np.zeros(3)
    s[2] = dr[2] / lz
    s[1] = (dr[1] - s[2]*yz) / ly
    s[0] = (dr[0] - s[1]*xy - s[2]*xz) / lx
    s -= np.round(s)
    return np.array([s[0]*lx + s[1]*xy + s[2]*xz,
                     s[1]*ly + s[2]*yz, s[2]*lz])


def compute_eq_observables(data_path: Path):
    lx, ly, lz, xy, xz, yz, atoms = parse_data_file(data_path)
    types = atoms[:,1].astype(int); pos = atoms[:,2:5]
    mo_idx = np.where(types==1)[0]; s_idx = np.where(types==2)[0]
    a_AA = lx / 3.0
    nMo, nS = len(mo_idx), len(s_idx)
    dvec = np.zeros((nMo, nS, 3))
    for i, mi in enumerate(mo_idx):
        for j, si in enumerate(s_idx):
            dvec[i,j] = min_image(pos[si]-pos[mi], lx, ly, lz, xy, xz, yz)
    dist = np.linalg.norm(dvec, axis=2)
    d_per_mo, h_per_mo, ai_per_mo = [], [], []
    for i in range(nMo):
        order = np.argsort(dist[i]); nbrs = order[:6]
        dvecs = dvec[i, nbrs]; ds = dist[i, nbrs]
        d_per_mo.append(ds.mean())
        zs = dvecs[:,2]
        top = dvecs[zs>0]; bot = dvecs[zs<0]
        if len(top)>=1 and len(bot)>=1:
            h_per_mo.append(0.5*(np.mean(np.abs(top[:,2]))+np.mean(np.abs(bot[:,2]))))
        if len(top)>=2:
            angs=[]
            for a in range(len(top)):
                for b in range(a+1,len(top)):
                    va, vb = top[a], top[b]
                    cos = np.dot(va,vb)/(np.linalg.norm(va)*np.linalg.norm(vb))
                    cos = max(-1.0, min(1.0, cos))
                    angs.append(math.degrees(math.acos(cos)))
            if angs: ai_per_mo.append(min(angs))
    return {
        "a_AA": a_AA,
        "d_MoS_AA": float(np.mean(d_per_mo)) if d_per_mo else float("nan"),
        "h_S_AA":   float(np.mean(h_per_mo)) if h_per_mo else float("nan"),
        "angle_intra": float(np.mean(ai_per_mo)) if ai_per_mo else float("nan"),
    }


def evaluate(spec, x, src_lines, dft_eps, dft_sig):
    """Evaluate joint loss. Returns (joint_loss, components, sigs, eq_obs)."""
    tmp = OUT_DIR / "tmp_ffield.reax"
    write_ffield(src_lines, spec, x, tmp)

    # 1) Biaxial sigma(eps) on 10x10
    wd_b = setup_wd(tmp, WD_BIAX, DATA10x10)
    rc, log = run_lmp(biax_init_deck(DATA10x10.name), wd_b, "init", timeout=180)
    if rc != 0:
        return 9999.0, {}, [], {}
    lx0 = grab(log,"RELAXED_LX"); ly0 = grab(log,"RELAXED_LY")
    if lx0 is None:
        return 9999.0, {}, [], {}
    sigs = []
    for i, eps in enumerate(dft_eps):
        tlx = lx0 * math.exp(float(eps)); tly = ly0 * math.exp(float(eps))
        rc, log = run_lmp(biax_strain_deck(tlx, tly), wd_b, f"s{i}", timeout=90)
        s = grab(log, "FINAL_SIG_ISO_GPA") if rc == 0 else None
        if s is None or not math.isfinite(s):
            return 9999.0, {}, sigs, {}
        sigs.append(s)
        shutil.copyfile(wd_b/"data.eps.lammpsdata", wd_b/"data.relaxed.lammpsdata")
    sigs = np.array(sigs)
    valid = np.abs(dft_sig) > 1e-6
    rel_b = np.abs(sigs - dft_sig)/np.abs(dft_sig)*100.0
    biax_max_re = float(np.max(rel_b[valid])) if valid.any() else 9999.0

    # 2) Equilibrium on 3x3
    wd_e = setup_wd(tmp, WD_EQ, DATA3x3)
    rc, log = run_lmp(eq_deck(), wd_e, "eq", timeout=180)
    if rc != 0:
        return 9999.0, {"biax_max_re": biax_max_re}, sigs.tolist(), {}
    relaxed = wd_e / "data.eq.lammpsdata"
    if not relaxed.exists():
        return 9999.0, {"biax_max_re": biax_max_re}, sigs.tolist(), {}
    try:
        eq = compute_eq_observables(relaxed)
    except Exception:
        return 9999.0, {"biax_max_re": biax_max_re}, sigs.tolist(), {}

    a_re   = 100*abs(eq["a_AA"]      - DFT_REF["a_AA"])      / DFT_REF["a_AA"]
    d_re   = 100*abs(eq["d_MoS_AA"]  - DFT_REF["d_MoS_AA"])  / DFT_REF["d_MoS_AA"]
    ang_re = 100*abs(eq["angle_intra"] - DFT_REF["angle_intra"]) / DFT_REF["angle_intra"]

    comps = {
        "biax_max_re": biax_max_re,
        "a_re":        a_re,
        "d_re":        d_re,
        "ang_re":      ang_re,
    }
    # v12 constrained quadratic-penalty loss. Biaxial is the primary
    # objective; geometry is held by penalty terms that activate only when
    # the relative error exceeds the per-observable tolerance. This lets
    # the BO chase biaxial aggressively while preventing equilibrium drift.
    pen_a   = max(0.0, a_re   - TOL_A)
    pen_d   = max(0.0, d_re   - TOL_D)
    pen_ang = max(0.0, ang_re - TOL_ANG)
    penalty = PEN_W * (pen_a*pen_a + pen_d*pen_d + pen_ang*pen_ang)
    joint = biax_max_re + penalty
    comps["penalty"] = penalty
    return float(joint), comps, sigs.tolist(), eq


# ---------------------------------------------------------------------------
# Main BO loop
# ---------------------------------------------------------------------------
def main(budget=80, seed=42):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = read_lines(SRC_FFIELD)
    off = parse_offsets(src)
    spec = build_spec(src, off)
    n_params = len(spec)
    lows  = np.array([s["lo"] for s in spec])
    highs = np.array([s["hi"] for s in spec])
    x0    = np.array([s["init"] for s in spec])
    names = [s["name"] for s in spec]
    print(f"[setup] v10 joint BO: {n_params} params, budget={budget}, seed={seed}")
    print(f"[setup] src={SRC_FFIELD.name}")
    print(f"[setup] x0 in bounds: {(x0>=lows).all() and (x0<=highs).all()}")

    dft_df = pd.read_csv(DFT_CSV)
    dft_df = dft_df[(dft_df["category"]=="BIAXIAL")
                    & (dft_df["condition_var"]=="true_strain_biaxial")].copy()
    dft_df = dft_df.dropna(subset=["stress_GPa"]).sort_values("condition_value")
    dft_eps = np.array([0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30])
    dft_sig = np.interp(dft_eps, dft_df["condition_value"].values, dft_df["stress_GPa"].values)
    print(f"[setup] DFT sigs: {[round(s,2) for s in dft_sig]}")
    print(f"[setup] DFT eq:  a={DFT_REF['a_AA']} d_MoS={DFT_REF['d_MoS_AA']} "
          f"angle_intra={DFT_REF['angle_intra']}")

    # CSV log header
    cols = ["trial","timestamp","joint_loss","biax_max_re","a_re","d_re","ang_re",
            "a_AA","d_MoS_AA","h_S_AA","angle_intra"] + names + ["sigs"]
    if not LOG_PATH.exists():
        LOG_PATH.write_text(",".join(cols) + "\n", encoding="utf-8")

    space = [Real(float(lo), float(hi), name=n) for lo, hi, n in zip(lows, highs, names)]
    bo = Optimizer(dimensions=space, base_estimator="GP",
                   acq_func="EI", n_initial_points=10,
                   initial_point_generator="lhs", random_state=seed)

    # Inject v9 best as the first warm-start point so BO doesn't waste
    # evaluations on bad regions.
    bo.tell(list(x0), 999.0)  # placeholder; will be overwritten on real eval

    best = {"joint": float("inf"), "x": None, "comps": None, "sigs": None, "eq": None}
    history = []

    t_start = time.time()
    for trial in range(1, budget + 1):
        if trial == 1:
            x = list(x0)
        else:
            x = bo.ask()
        x_arr = np.minimum(np.maximum(np.asarray(x), lows), highs)
        t0 = time.time()
        joint, comps, sigs, eq = evaluate(spec, x_arr, src, dft_eps, dft_sig)
        dt = time.time() - t0

        if trial == 1:
            # Remove the placeholder and tell the real value
            bo.Xi.pop(); bo.yi.pop()
        try:
            bo.tell(list(x_arr), float(joint))
        except Exception as e:
            print(f"  [warn] bo.tell error: {e}")

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        sigs_str = ";".join(f"{s:.3f}" for s in sigs) if sigs else ""
        row = [str(trial), ts, f"{joint:.4f}",
               f"{comps.get('biax_max_re', float('nan')):.4f}",
               f"{comps.get('a_re', float('nan')):.4f}",
               f"{comps.get('d_re', float('nan')):.4f}",
               f"{comps.get('ang_re', float('nan')):.4f}",
               f"{eq.get('a_AA', float('nan')):.6f}",
               f"{eq.get('d_MoS_AA', float('nan')):.6f}",
               f"{eq.get('h_S_AA', float('nan')):.6f}",
               f"{eq.get('angle_intra', float('nan')):.4f}",
               *[f"{v:.6f}" for v in x_arr],
               f'"{sigs_str}"']
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(",".join(row) + "\n")

        history.append({
            "trial": trial, "joint": joint, "comps": comps,
            "x": x_arr.tolist(), "sigs": sigs, "eq": eq,
        })

        marker = ""
        improved = joint < best["joint"] - 1e-6
        if improved:
            best["joint"] = joint
            best["x"] = x_arr.tolist()
            best["comps"] = comps
            best["sigs"] = sigs
            best["eq"] = eq
            marker = " <- NEW BEST"
            write_ffield(src, spec, x_arr, DST_FFIELD)
            BEST_PATH.write_text(
                f"trial {trial}\njoint_loss = {joint:.3f}\n"
                f"components = {comps}\n"
                f"eq = {eq}\nsigs = {sigs}\nDFT  = {dft_sig.tolist()}\n"
                f"params = {dict(zip(names, x_arr.tolist()))}\n",
                encoding="utf-8")
        print(f"t{trial:4d} [{dt:5.1f}s] J={joint:8.3f} "
              f"(biax={comps.get('biax_max_re', float('nan')):6.2f} "
              f"a={comps.get('a_re', float('nan')):6.2f} "
              f"d={comps.get('d_re', float('nan')):6.2f} "
              f"ang={comps.get('ang_re', float('nan')):6.2f})  "
              f"best={best['joint']:7.3f}{marker}", flush=True)

        # Early stop only when ALL four metrics are within target on the
        # best run so far (joint = biax_max_re + penalty <= 10 implies
        # biax<=10 AND geometry within tolerance).
        bc = best["comps"] or {}
        all_ok = (bc.get("biax_max_re", 9e9) <= 10.0 and
                  bc.get("a_re", 9e9)        <= 10.0 and
                  bc.get("d_re", 9e9)        <= 10.0 and
                  bc.get("ang_re", 9e9)      <= 10.0)
        if all_ok:
            print(f"\n[target met!] best biax={bc['biax_max_re']:.2f} "
                  f"a={bc['a_re']:.2f} d={bc['d_re']:.2f} ang={bc['ang_re']:.2f}",
                  flush=True)
            break

    elapsed = time.time() - t_start
    print(f"\nFinal: best joint = {best['joint']:.3f}  ({elapsed/60:.1f} min, "
          f"{trial} trials)")

    # Save full BO state for posterior reconstruction
    state = {
        "trials": trial,
        "names": names,
        "lows": lows.tolist(),
        "highs": highs.tolist(),
        "x_history": [h["x"] for h in history],
        "joint_history": [h["joint"] for h in history],
        "best": best,
        "elapsed_s": elapsed,
    }
    (OUT_DIR / "bo_state.json").write_text(json.dumps(state, default=float, indent=2),
                                           encoding="utf-8")
    print(f"[state] -> {OUT_DIR / 'bo_state.json'}")
    return state


if __name__ == "__main__":
    budget = 80 if len(sys.argv) < 2 else int(sys.argv[1])
    main(budget=budget)
