"""Local CLI entry point for PIBO-ReaxFF.

    python main.py --system MoSH --optimizers pibo jax_reaxff indeedopt pso \
                   --budget 200 --replicates 5 --physics-informed
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from pibo_reaxff.benchmark import BenchmarkSuite


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PIBO-ReaxFF benchmark (PIBO / JAX-ReaxFF / INDEEDopt / PSO)",
    )
    p.add_argument("--system", default="MoSH")
    p.add_argument("--config", default="configs/default_mosh.json")
    p.add_argument("--dft-root", default=None)
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--replicates", type=int, default=5)
    p.add_argument("--optimizers", nargs="+",
                   default=["pibo", "jax_reaxff", "indeedopt", "pso"],
                   choices=["pibo", "jax_reaxff", "indeedopt", "pso"])
    p.add_argument("--physics-informed", action="store_true", default=True)
    p.add_argument("--no-ablation", action="store_true",
                   help="Skip the physics-off ablation pass.")
    p.add_argument("--lammps-mode", default="auto",
                   choices=["auto", "lammps", "surrogate"])
    p.add_argument("--results-dir", default="results")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.results_dir, exist_ok=True)

    suite = BenchmarkSuite(system=args.system,
                           config=args.config,
                           dft_root=args.dft_root,
                           lammps_mode=args.lammps_mode)

    df = suite.run_all(
        replicates=args.replicates,
        budget=args.budget,
        optimizers=args.optimizers,
        physics_informed=args.physics_informed,
        ablation_physics_off=not args.no_ablation,
    )

    print("\n=== Per-run benchmark table ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(args.results_dir, "benchmark_per_run.csv"), index=False)

    agg = suite.report_aggregate()
    print("\n=== Aggregate (mean across replicates) ===")
    print(agg.to_string(index=False))
    agg.to_csv(os.path.join(args.results_dir, "benchmark_aggregate.csv"), index=False)

    figdir = os.path.join(args.results_dir, "figures")
    suite.plot_all(save_dir=figdir)
    print(f"\nFigures written to {figdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
