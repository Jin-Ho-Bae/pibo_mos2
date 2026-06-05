"""PIBO recalib v4: bootstrap from v3 (33.91%). Add general params (gp_*),
hbond params, ±100% bounds, 800 trials.
"""
from __future__ import annotations
import math, re, shutil, subprocess, sys, textwrap, time
from pathlib import Path
import numpy as np, pandas as pd
from skopt import gp_minimize
from skopt.space import Real

ROOT = Path(__file__).resolve().parents[1]
SRC_FFIELD = ROOT / "results" / "ffield" / "ffield.reax.MoSH.pibo_biaxial_v3.reax"
DST_FFIELD = ROOT / "results" / "ffield" / "ffield.reax.MoSH.pibo_biaxial_v4.reax"
DFT_CSV    = ROOT / "MoS2_physical_validation.csv"
from _lmp_path import find_lmp as _find_lmp  # parameterized LAMMPS discovery
LMP        = _find_lmp()
DATA_FILE  = ROOT / "lammps_templates" / "data.mos2_2H_monolayer_10x10_ryanDFT.lammpsdata"
OUT_DIR    = ROOT / "results" / "reviewer_response" / "biaxial_recalib_v4"
LOG_PATH   = OUT_DIR / "RECALIB_V4_LOG.csv"
BEST_PATH  = OUT_DIR / "BEST_RESULT.txt"

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


def parse_offsets(lines):
    o = {}; i = 1
    n_gen = int(lines[i].split()[0]); i += 1
    o["general_start"] = i; i += n_gen
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


def build_spec(lines, off):
    """v4 spec: includes all v3 params PLUS general params + hbond."""
    spec = []
    # === Same as v3 ===
    bs = off["bond_start"]; mos_l1, mos_l2 = bs + 2, bs + 3
    p1 = lines[mos_l1].split(); p2 = lines[mos_l2].split()
    for n, c in [("De_sigma_MoS",0),("De_pi_MoS",1),("p_be1_MoS",3),
                 ("p_bo5_MoS",4),("p_bo6_MoS",6),("p_ovun1_MoS",7)]:
        spec.append({"name":n,"line":mos_l1,"col_in_floats":c,"init":float(p1[2+c])})
    for n, c in [("p_bo1_MoS",4),("p_bo2_MoS",5)]:
        spec.append({"name":n,"line":mos_l2,"col_in_floats":c,"init":float(p2[c])})
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
    at = off["atom_start"]
    p_mo_l0 = lines[at].split()
    for n, c in [("Mo_RvdW",3),("Mo_Dij",4),("Mo_gamma",5)]:
        spec.append({"name":n,"line":at,"col_in_floats":c,"init":float(p_mo_l0[1+c])})
    p_mo_l1 = lines[at+1].split()
    for n, c in [("Mo_alfa",0),("Mo_gamma_w",1)]:
        spec.append({"name":n,"line":at+1,"col_in_floats":c,"init":float(p_mo_l1[c])})
    p_mo_l2 = lines[at+2].split()
    for n, c in [("Mo_p_boc4",3),("Mo_p_boc3",4),("Mo_p_boc5",5)]:
        spec.append({"name":n,"line":at+2,"col_in_floats":c,"init":float(p_mo_l2[c])})
    p_mo_l3 = lines[at+3].split()
    for n, c in [("Mo_p_val3",1),("Mo_p_val5",4)]:
        spec.append({"name":n,"line":at+3,"col_in_floats":c,"init":float(p_mo_l3[c])})
    p_s_l0 = lines[at+4].split()
    for n, c in [("S_RvdW",3),("S_Dij",4),("S_gamma",5)]:
        spec.append({"name":n,"line":at+4,"col_in_floats":c,"init":float(p_s_l0[1+c])})
    p_s_l1 = lines[at+5].split()
    for n, c in [("S_alfa",0),("S_gamma_w",1)]:
        spec.append({"name":n,"line":at+5,"col_in_floats":c,"init":float(p_s_l1[c])})
    p_s_l2 = lines[at+6].split()
    for n, c in [("S_p_boc4",3),("S_p_boc3",4),("S_p_boc5",5)]:
        spec.append({"name":n,"line":at+6,"col_in_floats":c,"init":float(p_s_l2[c])})
    # Torsion (was added in v3)
    tr = off["tor_start"]
    for ti in range(off["n_tor"]):
        ln = tr + ti; toks = lines[ln].split()
        if len(toks) >= 4 and all(t in ("1","2") for t in toks[:4]):
            for n, c in [("Tor_V1",0),("Tor_V2",1),("Tor_V3",2)]:
                try:
                    spec.append({"name":f"{n}_t{ti}","line":ln,
                                 "col_in_floats":c,"init":float(toks[4+c])})
                except (IndexError, ValueError):
                    pass
            if ti >= 1: break

    # === NEW in v4: General parameters that affect overall energy scale ===
    gs = off["general_start"]
    # General params (single value per line): only vary a few that matter for energy scale
    # gp1 is p_boc1, gp2 is p_boc2, etc.
    # Pick ones likely to matter: p_boc1, p_boc2, p_val6, p_lp1, p_val9, p_val10
    # General params are at gs + idx, single float per line
    gp_indices_to_vary = [0, 1, 14, 15, 16, 17]  # boc1, boc2, val6, lp1, val9, val10
    for gi in gp_indices_to_vary:
        ln = gs + gi
        toks = lines[ln].split()
        if toks:
            try:
                v = float(toks[0])
                spec.append({"name":f"gp_{gi}","line":ln,"col_in_floats":0,"init":v})
            except ValueError:
                pass

    # === NEW in v4: Hydrogen bond (S-H-S, even though no H, the params still exist) ===
    # Skip if not impactful — H is not in the system

    # Bounds: ±100% (allow sign change for params that aren't ~0)
    for s in spec:
        v = s["init"]
        if abs(v) > 0.01:
            margin = 1.0 * abs(v)
        else:
            margin = 0.5
        s["lo"] = v - margin
        s["hi"] = v + margin
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


def setup_eval(ff):
    wd = OUT_DIR / "_active"
    if wd.exists():
        shutil.rmtree(wd, ignore_errors=True)
        if wd.exists(): time.sleep(0.2); shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ff, wd / "ffield.reax")
    (wd / "lmp_control").write_text(LMP_CONTROL, encoding="utf-8")
    (wd / "data.mos2_2H_monolayer_10x10_ryanDFT.lammpsdata").write_text(
        DATA_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    return wd


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


def init_deck():
    return textwrap.dedent(f"""\
        units real
        atom_style charge
        boundary p p p
        read_data data.mos2_2H_monolayer_10x10_ryanDFT.lammpsdata
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


def strain_deck(tlx, tly):
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


def evaluate(spec, x, src_lines, dft_eps, dft_sig):
    tmp = OUT_DIR / "tmp_ffield.reax"
    write_ffield(src_lines, spec, x, tmp)
    wd = setup_eval(tmp)
    rc, log = run_lmp(init_deck(), wd, "init", timeout=180)
    if rc != 0: return 9999.0, []
    lx0 = grab(log,"RELAXED_LX"); ly0 = grab(log,"RELAXED_LY")
    if lx0 is None: return 9999.0, []
    sigs = []
    for i, eps in enumerate(dft_eps):
        tlx = lx0 * math.exp(float(eps)); tly = ly0 * math.exp(float(eps))
        rc, log = run_lmp(strain_deck(tlx,tly), wd, f"s{i}", timeout=90)
        s = grab(log,"FINAL_SIG_ISO_GPA") if rc == 0 else None
        if s is None or not math.isfinite(s): return 9999.0, sigs
        sigs.append(s)
        shutil.copyfile(wd / "data.eps.lammpsdata", wd / "data.relaxed.lammpsdata")
    sigs = np.array(sigs)
    denom = np.where(np.abs(dft_sig) > 1e-6, np.abs(dft_sig), 1.0)
    rel = np.where(np.abs(dft_sig) > 1e-6, np.abs(sigs - dft_sig)/denom*100, np.nan)
    valid = (np.abs(dft_sig) > 1e-6) & np.isfinite(rel)
    if not valid.any(): return 9999.0, sigs
    return float(np.nanmax(rel[valid])), sigs.tolist()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = read_lines(SRC_FFIELD)
    off = parse_offsets(src)
    spec = build_spec(src, off)
    print(f"[setup] v4: {len(spec)} params, ±100% bounds, 800 trials", flush=True)

    dft_df = pd.read_csv(DFT_CSV)
    dft_df = dft_df[(dft_df["category"]=="BIAXIAL")
                    & (dft_df["condition_var"]=="true_strain_biaxial")].copy()
    dft_df = dft_df.dropna(subset=["stress_GPa"]).sort_values("condition_value")
    dft_eps = np.array([0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30])
    dft_sig = np.interp(dft_eps, dft_df["condition_value"].values, dft_df["stress_GPa"].values)

    if not LOG_PATH.exists():
        header = "trial,timestamp,max_re," + ",".join(s["name"] for s in spec) + ",sigs\n"
        LOG_PATH.write_text(header, encoding="utf-8")

    best = {"max_re": float("inf"), "params": None, "sigs": None}
    tc = [0]
    def objective(x):
        tc[0] += 1
        t0 = time.time()
        max_re, sigs = evaluate(spec, x, src, dft_eps, dft_sig)
        rt = time.time() - t0
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        sigs_str = ";".join(f"{s:.3f}" for s in sigs) if sigs else ""
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"{tc[0]},{ts},{max_re:.3f}," +
                     ",".join(f"{v:.6f}" for v in x) + f",\"{sigs_str}\"\n")
        marker = ""
        if max_re < best["max_re"]:
            best["max_re"] = max_re; best["params"] = list(x); best["sigs"] = sigs
            marker = " ← NEW BEST"
            write_ffield(src, spec, x, DST_FFIELD)
            BEST_PATH.write_text(
                f"trial {tc[0]}\nmax_re = {max_re:.2f}%\n"
                f"sigs = {sigs}\nDFT  = {dft_sig.tolist()}\n"
                f"params = {dict(zip([s['name'] for s in spec], list(x)))}\n",
                encoding="utf-8")
        print(f"trial {tc[0]:3d}  [{rt:5.1f}s]  max_re={max_re:7.2f}%  "
              f"best={best['max_re']:7.2f}%{marker}", flush=True)
        return max_re

    bounds = [Real(s["lo"], s["hi"], name=s["name"]) for s in spec]
    x0 = [s["init"] for s in spec]
    gp_minimize(objective, bounds, x0=x0, n_calls=800, n_initial_points=80,
                acq_func="EI", random_state=271, verbose=False)
    print(f"\nFinal best: max_re = {best['max_re']:.2f}%")


if __name__ == "__main__":
    main()
