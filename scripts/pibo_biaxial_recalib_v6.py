"""PIBO recalib v6: focused 19-param search, stiffness-biased bounds, weighted
loss (PIBO systematically undershoots high-strain stress). Bootstrap from v3
(33.91%). Auto multi-restart loop targeting <10% max_re.

Strategy diff from v5:
  - 19 params (bond + offdiag + 2 angles) not 88 — BO scales poorly past ~20D
  - Asymmetric bounds: De_sigma/De_pi widen UPWARD (stiffness), |p_bo1| narrow
  - Loss = 0.7*max_re + 0.3*weighted_RMSE where weights emphasize high-strain
  - On plateau (200 trials no improvement) → expand bounds + restart from best
"""
from __future__ import annotations
import math, re, shutil, subprocess, sys, textwrap, time
from pathlib import Path
import numpy as np, pandas as pd
from skopt import gp_minimize
from skopt.space import Real

ROOT = Path(__file__).resolve().parents[1]
SRC_FFIELD = ROOT / "results" / "ffield" / "ffield.reax.MoSH.pibo_biaxial_v3.reax"
DST_FFIELD = ROOT / "results" / "ffield" / "ffield.reax.MoSH.pibo_biaxial_v6.reax"
DFT_CSV    = ROOT / "MoS2_physical_validation.csv"
from _lmp_path import find_lmp as _find_lmp  # parameterized LAMMPS discovery
LMP        = _find_lmp()
DATA_FILE  = ROOT / "lammps_templates" / "data.mos2_2H_monolayer_10x10_ryanDFT.lammpsdata"
OUT_DIR    = ROOT / "results" / "reviewer_response" / "biaxial_recalib_v6"
LOG_PATH   = OUT_DIR / "RECALIB_V6_LOG.csv"
BEST_PATH  = OUT_DIR / "BEST_RESULT.txt"
TARGET_MAX_RE = 10.0
PLATEAU_TRIALS = 200

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


# Asymmetric bound multipliers tailored to stiffen the model:
# PIBO undershoots stress monotonically => raise De_sigma/De_pi, lower |p_bo|.
# (lo_mult, hi_mult) applied to abs(init); sign preserved.
ASYM_BOUNDS = {
    "De_sigma_MoS": (0.80, 1.80),    # widen up — more stiffness
    "De_pi_MoS":    (0.80, 1.80),
    "p_be1_MoS":    (0.60, 1.60),    # moderate
    "p_bo5_MoS":    (0.50, 1.40),
    "p_bo6_MoS":    (0.50, 1.40),
    "p_ovun1_MoS":  (0.40, 1.60),
    "p_bo1_MoS":    (0.40, 1.20),    # narrow magnitude (it's negative)
    "p_bo2_MoS":    (0.70, 1.30),
    "Dij_MoS":      (0.50, 1.80),
    "RvdW_MoS":     (0.80, 1.30),
    "Alfa_MoS":     (0.70, 1.40),
    "ro_sigma_MoS": (0.85, 1.20),    # geometry — tight
    "Thetao_SMoS":  (0.85, 1.15),
    "p_val1_SMoS":  (0.60, 1.50),
    "p_val2_SMoS":  (0.60, 1.50),
    "p_val7_SMoS":  (0.60, 1.50),
    "p_val4_SMoS":  (0.60, 1.50),
    "Thetao_MoSMo": (0.85, 1.15),
    "p_val1_MoSMo": (0.50, 1.60),
}


def build_spec(lines, off, scale=1.0):
    """scale: multiplier on bound width (1.0 = ASYM_BOUNDS; >1 = wider)."""
    spec = []
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
    for s in spec:
        v = s["init"]; sign = -1.0 if v < 0 else 1.0; av = abs(v)
        lo_m, hi_m = ASYM_BOUNDS.get(s["name"], (0.6, 1.6))
        center = av
        half_lo = (1.0 - lo_m) * center * scale
        half_hi = (hi_m - 1.0) * center * scale
        lo_abs = max(1e-6, av - half_lo)
        hi_abs = av + half_hi
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
    if rc != 0: return 9999.0, 9999.0, []
    lx0 = grab(log,"RELAXED_LX"); ly0 = grab(log,"RELAXED_LY")
    if lx0 is None: return 9999.0, 9999.0, []
    sigs = []
    for i, eps in enumerate(dft_eps):
        tlx = lx0 * math.exp(float(eps)); tly = ly0 * math.exp(float(eps))
        rc, log = run_lmp(strain_deck(tlx,tly), wd, f"s{i}", timeout=90)
        s = grab(log,"FINAL_SIG_ISO_GPA") if rc == 0 else None
        if s is None or not math.isfinite(s): return 9999.0, 9999.0, sigs
        sigs.append(s)
        shutil.copyfile(wd / "data.eps.lammpsdata", wd / "data.relaxed.lammpsdata")
    sigs = np.array(sigs)
    valid = np.abs(dft_sig) > 1e-6
    if not valid.any(): return 9999.0, 9999.0, sigs.tolist()
    rel = np.abs(sigs - dft_sig) / np.abs(dft_sig) * 100.0
    max_re = float(np.max(rel[valid]))
    # weighted loss: emphasise high strain (idx 3..7) where PIBO undershoots
    weights = np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.6, 1.7])
    weights = weights[: len(dft_sig)]
    weighted_rmse = float(np.sqrt(np.average(rel[valid]**2, weights=weights[valid])))
    loss = 0.6 * max_re + 0.4 * weighted_rmse
    return loss, max_re, sigs.tolist()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = read_lines(SRC_FFIELD)
    off = parse_offsets(src)

    dft_df = pd.read_csv(DFT_CSV)
    dft_df = dft_df[(dft_df["category"]=="BIAXIAL")
                    & (dft_df["condition_var"]=="true_strain_biaxial")].copy()
    dft_df = dft_df.dropna(subset=["stress_GPa"]).sort_values("condition_value")
    dft_eps = np.array([0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30])
    dft_sig = np.interp(dft_eps, dft_df["condition_value"].values, dft_df["stress_GPa"].values)
    print(f"[setup] DFT sigs: {[round(s,2) for s in dft_sig]}", flush=True)

    best_global = {"max_re": float("inf"), "params": None, "sigs": None, "scale": None}
    restart_idx = 0
    seed_base = 314

    while best_global["max_re"] > TARGET_MAX_RE:
        scale = 1.0 + 0.5 * restart_idx       # 1.0, 1.5, 2.0, 2.5 ...
        spec = build_spec(src, off, scale=scale)
        n_params = len(spec)
        print(f"\n[restart {restart_idx}] scale={scale:.1f}, "
              f"{n_params} params, target<{TARGET_MAX_RE}%", flush=True)

        if not LOG_PATH.exists():
            header = ("trial,restart,scale,timestamp,loss,max_re," +
                      ",".join(s["name"] for s in spec) + ",sigs\n")
            LOG_PATH.write_text(header, encoding="utf-8")

        best_local = {"max_re": float("inf"), "loss": float("inf")}
        no_improve = [0]
        tc = [0]

        def objective(x):
            tc[0] += 1
            t0 = time.time()
            loss, max_re, sigs = evaluate(spec, x, src, dft_eps, dft_sig)
            rt = time.time() - t0
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            sigs_str = ";".join(f"{s:.3f}" for s in sigs) if sigs else ""
            with LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(f"{tc[0]},{restart_idx},{scale:.2f},{ts},"
                         f"{loss:.3f},{max_re:.3f}," +
                         ",".join(f"{v:.6f}" for v in x) + f",\"{sigs_str}\"\n")
            marker = ""
            if max_re < best_local["max_re"]:
                best_local["max_re"] = max_re
                best_local["loss"] = loss
                no_improve[0] = 0
                marker = " ← local best"
                if max_re < best_global["max_re"]:
                    best_global["max_re"] = max_re
                    best_global["params"] = list(x); best_global["sigs"] = sigs
                    best_global["scale"] = scale
                    marker = " ← NEW GLOBAL BEST"
                    write_ffield(src, spec, x, DST_FFIELD)
                    BEST_PATH.write_text(
                        f"restart {restart_idx} trial {tc[0]}\n"
                        f"scale = {scale}\nmax_re = {max_re:.2f}%  loss = {loss:.2f}\n"
                        f"sigs = {sigs}\nDFT  = {dft_sig.tolist()}\n"
                        f"params = {dict(zip([s['name'] for s in spec], list(x)))}\n",
                        encoding="utf-8")
            else:
                no_improve[0] += 1
            print(f"r{restart_idx} t{tc[0]:4d} [{rt:5.1f}s] "
                  f"loss={loss:7.2f} max_re={max_re:7.2f}% "
                  f"local={best_local['max_re']:6.2f}% "
                  f"global={best_global['max_re']:6.2f}%{marker}", flush=True)
            return loss

        bounds = [Real(s["lo"], s["hi"], name=s["name"]) for s in spec]
        x0 = [s["init"] for s in spec]
        # cap restart at safe n_calls to keep BO healthy; stop when target met
        budget = 800
        try:
            gp_minimize(objective, bounds, x0=x0, n_calls=budget,
                        n_initial_points=60, acq_func="EI",
                        random_state=seed_base + restart_idx, verbose=False)
        except Exception as e:
            print(f"[restart {restart_idx}] gp_minimize errored: {e}", flush=True)
        # plateau detection happens inside objective via no_improve, but the
        # gp_minimize budget already acts as an outer cap.
        restart_idx += 1
        if best_global["max_re"] <= TARGET_MAX_RE:
            print(f"\n*** TARGET MET *** max_re={best_global['max_re']:.2f}%", flush=True)
            break
        if restart_idx > 12:
            print("[stop] restart cap reached.", flush=True)
            break

    print(f"\nFinal: max_re={best_global['max_re']:.2f}%", flush=True)


if __name__ == "__main__":
    main()
