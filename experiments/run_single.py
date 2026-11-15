"""Reconstruct one potential and save a three-panel figure plus metrics.

Example:
    python experiments/run_single.py --potential trap_with_barrier --noise 0.02
"""
from __future__ import annotations

import argparse
import json
import os

from common import FIGURES, RESULTS, run_case

from invpinn.plotting import plot_reconstruction


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--potential", default="trap_with_barrier",
                   choices=["harmonic", "double_well", "trap_with_barrier"])
    p.add_argument("--noise", type=float, default=0.02, help="noise std as a fraction of peak density")
    p.add_argument("--n-obs", type=int, default=100, help="number of measurement pixels")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows, data, art = run_case(args.potential, args.noise, args.seed, n_obs=args.n_obs, keep=True)
    for r in rows:
        gap = "" if r["gap"] != r["gap"] else f"  gap {r['gap']:.4f} (true {r['true_gap']:.4f})"
        print(f"{r['method']:<12} error core {100 * r['err_core']:6.2f} %   extended {100 * r['err_ext']:6.2f} %{gap}")

    pinn2 = art["pinn_2state"]
    tag = f"{args.potential}_noise{int(round(100 * args.noise))}_seed{args.seed}"
    plot_reconstruction(
        data,
        {k: art[k + "_V"] for k in ["spline", "pinn_1state", "pinn_2state"]},
        fitted_densities={n: pinn2.predict_density(n, data.x_true) for n in range(2)},
        history=pinn2.history,
        errors={r["method"]: r["err_core"] for r in rows},
        path=os.path.join(FIGURES, f"reconstruction_{tag}.png"),
    )
    with open(os.path.join(RESULTS, f"single_{tag}.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved figure to results/figures/reconstruction_{tag}.png")


if __name__ == "__main__":
    main()
