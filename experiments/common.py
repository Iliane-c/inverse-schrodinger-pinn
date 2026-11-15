"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402

from invpinn import (  # noqa: E402
    PINNConfig,
    get_potential,
    identifiable_mask,
    make_dataset,
    max_abs_error,
    potential_error,
    spline_inversion_oracle,
    train_pinn,
)

RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")
os.makedirs(FIGURES, exist_ok=True)


def evaluate(V: np.ndarray, data) -> Dict[str, float]:
    core = identifiable_mask(data.rho_true[0], 0.1)
    ext = identifiable_mask(data.rho_true[0], 0.01)
    return {
        "err_core": potential_error(V, data.V_true, core),
        "err_ext": potential_error(V, data.V_true, ext),
        "maxabs_core": max_abs_error(V, data.V_true, core),
    }


def run_case(
    potential: str,
    noise: float,
    seed: int,
    n_obs: int = 100,
    methods: Optional[List[str]] = None,
    base_cfg: Optional[PINNConfig] = None,
    keep: bool = False,
) -> List[Dict]:
    """Run all methods on one synthetic dataset and return one row per method."""
    methods = methods or ["spline", "pinn_1state", "pinn_2state"]
    base_cfg = base_cfg or PINNConfig()
    data = make_dataset(get_potential(potential), n_states=2, n_obs=n_obs, noise=noise, seed=seed)
    core = identifiable_mask(data.rho_true[0], 0.1)
    true_gap = float(data.E_true[1] - data.E_true[0])
    rows, artefacts = [], {}
    for method in methods:
        row = {"potential": potential, "noise": noise, "seed": seed, "n_obs": n_obs, "method": method,
               "true_gap": true_gap, "gap": np.nan, "seconds": 0.0}
        if method == "spline":
            V = spline_inversion_oracle(data.x_obs, data.rho_obs[0], data.sigma[0], data.x_true, data.V_true, core)
        else:
            ns = 1 if method == "pinn_1state" else 2
            res = train_pinn(data, replace(base_cfg, n_states=ns, seed=seed))
            V = res.predict_V(data.x_true)
            row["seconds"] = res.train_seconds
            if ns == 2:
                row["gap"] = res.gap
            artefacts[method] = res
        row.update(evaluate(V, data))
        artefacts[method + "_V"] = V
        rows.append(row)
    if keep:
        return rows, data, artefacts
    return rows
