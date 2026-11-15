"""Noise sweep: spline baseline vs 1-state and 2-state PINNs, several seeds.

Example (about 15 min on 2 CPU cores):
    python experiments/benchmark.py --workers 2
"""
from __future__ import annotations

import argparse
import itertools
import os
from multiprocessing import Pool

import pandas as pd
import torch

from common import FIGURES, RESULTS, run_case

from invpinn.plotting import plot_benchmark

POTENTIALS = ["harmonic", "double_well", "trap_with_barrier"]


def _job(args):
    torch.set_num_threads(1)
    pot, noise, seed = args
    rows = run_case(pot, noise, seed)
    for r in rows:
        print(f"{pot:<18} noise {noise:.2f} seed {seed}  {r['method']:<12} core {100 * r['err_core']:6.2f} %",
              flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--noise", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default=os.path.join(RESULTS, "benchmark.csv"))
    p.add_argument("--summarise-only", action="store_true", help="rebuild tables and figures from an existing CSV")
    args = p.parse_args()
    if args.summarise_only:
        summarise(pd.read_csv(args.out))
        return

    jobs = list(itertools.product(POTENTIALS, args.noise, range(args.seeds)))
    with Pool(args.workers) as pool:
        rows = [r for chunk in pool.imap_unordered(_job, jobs) for r in chunk]
    df = pd.DataFrame(rows).sort_values(["potential", "noise", "seed", "method"])
    df.to_csv(args.out, index=False)
    summarise(df)


def summarise(df: pd.DataFrame) -> None:
    plot_benchmark(df, "err_core", os.path.join(FIGURES, "benchmark_core.png"),
                   "Relative L2 error of V where ρ₀ > 10 % of peak (line: mean, dots: individual seeds)")
    plot_benchmark(df, "err_ext", os.path.join(FIGURES, "benchmark_extended.png"),
                   "Relative L2 error of V where ρ₀ > 1 % of peak (line: mean, dots: individual seeds)")
    names = {"spline": "Spline (oracle)", "pinn_1state": "PINN 1-state", "pinn_2state": "PINN 2-state"}
    g = df.groupby(["potential", "noise", "method"])
    mean = (g[["err_core", "err_ext"]].mean() * 100)
    std = (g[["err_core", "err_ext"]].std() * 100)
    rows = []
    for (pot, noise), _ in df.groupby(["potential", "noise"]):
        row = {"potential": pot.replace("_", " "), "noise": f"{100 * noise:.0f} %"}
        for region, col in [("core", "err_core"), ("ext", "err_ext")]:
            for m in ["spline", "pinn_1state", "pinn_2state"]:
                row[f"{names[m]} · {region}"] = f"{mean.loc[(pot, noise, m), col]:.1f} ± {std.loc[(pot, noise, m), col]:.1f}"
        rows.append(row)
    table = pd.DataFrame(rows)
    with open(os.path.join(RESULTS, "benchmark_table.md"), "w") as f:
        f.write(table.to_markdown(index=False))
    gaps = df[df["method"] == "pinn_2state"].copy()
    gaps["rel_err"] = 100 * (gaps["gap"] - gaps["true_gap"]).abs() / gaps["true_gap"]
    gt = gaps.groupby(["potential", "noise"]).agg(true_gap=("true_gap", "first"), gap=("gap", "mean"),
                                                  gap_std=("gap", "std"), rel_err=("rel_err", "mean")).reset_index()
    gap_table = pd.DataFrame({
        "potential": gt["potential"].str.replace("_", " "),
        "noise": (100 * gt["noise"]).round().astype(int).astype(str) + " %",
        "true E1 − E0": gt["true_gap"].map("{:.4f}".format),
        "PINN estimate": [f"{a:.4f} ± {b:.4f}" for a, b in zip(gt["gap"], gt["gap_std"])],
        "mean rel. error": gt["rel_err"].map("{:.1f} %".format),
    })
    with open(os.path.join(RESULTS, "gap_table.md"), "w") as f:
        f.write(gap_table.to_markdown(index=False))
    print(table)
    print(gap_table)


if __name__ == "__main__":
    main()
