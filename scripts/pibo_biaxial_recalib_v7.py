"""PIBO recalib v7: pivot from BO to differential evolution + Nelder-Mead.
Bootstrap from v6 best (29.23%) — or fall back to v3 if v6 not yet present.
22 params (adds p_bo3, p_bo4, p_be2). Direct max_re objective (no weighting
mixed in like v6). Tighter bounds (±30%) for cleaner DE convergence.

DE is population-based, robust to discontinuous LAMMPS-crash regions where
GP-EI struggled. NM refines locally after DE.
"""
from __future__ import annotations
import math, re, shutil, subprocess, textwrap, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import differential_evolution, minimize

ROOT = Path(__file__).resolve().parents[1]
FFIELD_V6 = ROOT / "results" / "ffield" / "ffield.reax.MoSH.pibo_biaxial_v6.reax"
FFIELD_V3 = ROOT / "results" / "ffield" / "ffield.reax.MoSH.pibo_biaxial_v3.reax"
SRC_FFIELD = FFIELD_V6 if FFIELD_V6.exists() else FFIELD_V3
DST_FFIELD = ROOT / "results" / "ffield" / "ffield.reax.MoSH.pibo_biaxial_v7.reax"
DFT_CSV    = ROOT / "MoS2_physical_validation.csv"
from _lmp_path import find_lmp as _find_lmp  # parameterized LAMMPS discovery
LMP        = _find_lmp()
DATA_FILE  = ROOT / "lammps_templates" / "data.mos2_2H_monolayer_10x10_ryanDFT.lammpsdata"
OUT_DIR    = ROOT / "results" / "reviewer_response" / "biaxial_recalib_v7"
LOG_PATH   = OUT_DIR / "RECALIB_V7_LOG.csv"
BEST_PATH  = OUT_DIR / "BEST_RESULT.txt"
TARGET_MAX_RE = 10.0

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


def build_spec(lines, off):
    """22 params: same 19 as v6 + p_bo3, p_bo4 from Mo-S bond line 1."""
    spec = []
    bs = off["bond_start"]; mos_l1, mos_l2 = bs + 2, bs + 3
    p1 = lines[mos_l1].split(); p2 = lines[mos_l2].split()
    for n, c in [("De_sigma_MoS",0),("De_pi_MoS",1),("p_be1_MoS",3),
                 ("p_bo5_MoS",4),("p_bo6_MoS",6),("p_ovun1_MoS",7)]:
        spec.append({"name":n,"line":mos_l1,"col_in_floats":c,"init":float(p1[2+c])})
    # bond line 2: tokens c0..c7 → c0=p_be2, c1=p_bo3, c2=p_bo4, c3=?, c4=p_bo1, c5=p_bo2
    for n, c in [("p_be2_MoS",0),("p_bo3_MoS",1),("p_bo4_MoS",2),
                 ("p_bo1_MoS",4),("p_bo2_MoS",5)]:
        try: spec.append({"name":n,"line":mos_l2,"col_in_floats":c,"init":float(p2[c])})
        except (IndexError, ValueError): pass
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
    # Bounds ±30% (symmetric); preserves sign.
    for s in spec:
        v = s["init"]; sign = -1.0 if v < 0 else 1.0; av = abs(v)
        margin = 0.30 * av if av > 0.01 else 0.05
        lo_abs = max(1e-6, av - margin)
        hi_abs = av + margin
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
    valid = np.abs(dft_sig) > 1e-6
    if not valid.any(): return 9999.0, sigs.tolist()
    rel = np.abs(sigs - dft_sig) / np.abs(dft_sig) * 100.0
    return float(np.max(rel[valid])), sigs.tolist()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = read_lines(SRC_FFIELD)
    off = parse_offsets(src)
    spec = build_spec(src, off)
    n_params = len(spec)
    print(f"[setup] v7: src={SRC_FFIELD.name}, {n_params} params, "
          f"DE + NM, target<{TARGET_MAX_RE}%", flush=True)

    dft_df = pd.read_csv(DFT_CSV)
    dft_df = dft_df[(dft_df["category"]=="BIAXIAL")
                    & (dft_df["condition_var"]=="true_strain_biaxial")].copy()
    dft_df = dft_df.dropna(subset=["stress_GPa"]).sort_values("condition_value")
    dft_eps = np.array([0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30])
    dft_sig = np.interp(dft_eps, dft_df["condition_value"].values, dft_df["stress_GPa"].values)
    print(f"[setup] DFT sigs: {[round(s,2) for s in dft_sig]}", flush=True)

    if not LOG_PATH.exists():
        header = ("trial,phase,timestamp,max_re," +
                  ",".join(s["name"] for s in spec) + ",sigs\n")
        LOG_PATH.write_text(header, encoding="utf-8")

    best = {"max_re": float("inf"), "params": None, "sigs": None, "phase": None}
    tc = [0]
    phase = ["DE"]

    def objective(x):
        tc[0] += 1
        t0 = time.time()
        max_re, sigs = evaluate(spec, x, src, dft_eps, dft_sig)
        rt = time.time() - t0
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        sigs_str = ";".join(f"{s:.3f}" for s in sigs) if sigs else ""
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"{tc[0]},{phase[0]},{ts},{max_re:.3f}," +
                     ",".join(f"{v:.6f}" for v in x) + f",\"{sigs_str}\"\n")
        marker = ""
        if max_re < best["max_re"]:
            best["max_re"] = max_re; best["params"] = list(x)
            best["sigs"] = sigs; best["phase"] = phase[0]
            marker = " ← NEW BEST"
            write_ffield(src, spec, x, DST_FFIELD)
            BEST_PATH.write_text(
                f"phase {phase[0]} trial {tc[0]}\n"
                f"max_re = {max_re:.2f}%\n"
                f"sigs = {sigs}\nDFT  = {dft_sig.tolist()}\n"
                f"params = {dict(zip([s['name'] for s in spec], list(x)))}\n",
                encoding="utf-8")
        print(f"{phase[0]} t{tc[0]:4d} [{rt:5.1f}s] max_re={max_re:7.2f}% "
              f"best={best['max_re']:7.2f}%{marker}", flush=True)
        if max_re <= TARGET_MAX_RE:
            raise StopIteration  # break out of optimizer early
        return max_re

    bounds = [(s["lo"], s["hi"]) for s in spec]
    x0 = np.array([s["init"] for s in spec])

    # Phase 1: differential evolution (population-based, robust)
    print("[phase] DE: pop=30, maxiter=80 (~2400 evals)", flush=True)
    try:
        differential_evolution(
            objective, bounds=bounds, maxiter=80, popsize=30,
            mutation=(0.4, 1.2), recombination=0.7, tol=1e-4,
            seed=2027, polish=False, init="sobol", x0=x0, workers=1)
    except StopIteration:
        print("[phase DE] target met — early stop", flush=True)
    except Exception as e:
        print(f"[phase DE] error: {e}", flush=True)

    # Phase 2: Nelder-Mead refinement around best
    if best["max_re"] > TARGET_MAX_RE and best["params"] is not None:
        print(f"[phase] NM refinement from best={best['max_re']:.2f}%", flush=True)
        phase[0] = "NM"
        try:
            minimize(objective, np.array(best["params"]), method="Nelder-Mead",
                     options={"maxiter": 1500, "xatol": 1e-4, "fatol": 1e-3,
                              "adaptive": True})
        except StopIteration:
            print("[phase NM] target met", flush=True)
        except Exception as e:
            print(f"[phase NM] error: {e}", flush=True)

    # Phase 3: DE again, narrower bounds around current best
    if best["max_re"] > TARGET_MAX_RE and best["params"] is not None:
        print(f"[phase] DE-2 narrow around best={best['max_re']:.2f}%", flush=True)
        phase[0] = "DE2"
        narrow_bounds = []
        for s, v in zip(spec, best["params"]):
            half = 0.15 * abs(v) if abs(v) > 0.01 else 0.02
            narrow_bounds.append((v - half, v + half))
        try:
            differential_evolution(
                objective, bounds=narrow_bounds, maxiter=60, popsize=25,
                mutation=(0.3, 1.0), recombination=0.7, tol=1e-4,
                seed=4044, polish=False, init="sobol",
                x0=np.array(best["params"]), workers=1)
        except StopIteration:
            print("[phase DE2] target met", flush=True)
        except Exception as e:
            print(f"[phase DE2] error: {e}", flush=True)

    print(f"\nFinal: max_re={best['max_re']:.2f}%  phase={best['phase']}", flush=True)


if __name__ == "__main__":
    main()
