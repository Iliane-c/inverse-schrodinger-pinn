"""Ablations on the design choices of the PINN.

1. Residual weight w_res.
2. Density weighting of the residuals on/off.
3. Warm-up and coupling ramp on/off (2-state model).
4. Gradient clipping on/off.

Example:
    python experiments/ablation.py --workers 2
"""
from __future__ import annotations

import argparse
import itertools
import os
from dataclasses import replace
from multiprocessing import Pool

import pandas as pd
import torch

from common import RESULTS, run_case

from invpinn import PINNConfig

VARIANTS = {
    "default": PINNConfig(),
    "w_res=1e-3": replace(PINNConfig(), w_res=1e-3),
    "w_res=3e-3": replace(PINNConfig(), w_res=3e-3),
    "w_res=3e-2": replace(PINNConfig(), w_res=3e-2),
    "no density weighting": replace(PINNConfig(), density_weighting=False),
    "no warm-up / ramp": replace(PINNConfig(), warmup_frac=0.0),
    "no gradient clipping": replace(PINNConfig(), grad_clip=0.0),
}


def _job(args):
    torch.set_num_threads(1)
    variant, pot, seed = args
    methods = ["pinn_2state"] if variant == "no warm-up / ramp" else ["pinn_1state", "pinn_2state"]
    rows = run_case(pot, 0.02, seed, methods=methods, base_cfg=VARIANTS[variant])
    for r in rows:
        r["variant"] = variant
        print(f"{variant:<22} {pot:<18} seed {seed} {r['method']:<12} core {100 * r['err_core']:6.2f} %", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--potentials", nargs="+", default=["double_well", "trap_with_barrier"])
    p.add_argument("--summarise-only", action="store_true", help="rebuild the table from results/ablation.csv")
    args = p.parse_args()
    if args.summarise_only:
        summarise(pd.read_csv(os.path.join(RESULTS, "ablation.csv")))
        return
    jobs = list(itertools.product(VARIANTS, args.potentials, range(args.seeds)))
    with Pool(args.workers) as pool:
        rows = [r for chunk in pool.imap_unordered(_job, jobs) for r in chunk]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "ablation.csv"), index=False)
    summarise(df)


def summarise(df: pd.DataFrame) -> None:
    mean = df.groupby(["variant", "potential", "method"])["err_core"].mean() * 100
    names = {"pinn_1state": "1-state", "pinn_2state": "2-state"}
    rows = []
    for v in VARIANTS:
        row = {"variant": v}
        for pot in ["double_well", "trap_with_barrier"]:
            for m in ["pinn_1state", "pinn_2state"]:
                key = (v, pot, m)
                row[f"{pot.replace('_', ' ')} · {names[m]}"] = f"{mean[key]:.1f}" if key in mean.index else "—"
        rows.append(row)
    table = pd.DataFrame(rows)
    with open(os.path.join(RESULTS, "ablation_table.md"), "w") as f:
        f.write(table.to_markdown(index=False))
    print(table)


if __name__ == "__main__":
    main()
