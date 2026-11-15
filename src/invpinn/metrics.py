"""Gauge-aware error metrics.

The ground-state density determines V only up to an additive constant (shifting
V shifts every energy by the same amount and leaves all eigenfunctions
unchanged). All potential errors are therefore computed after removing the best
constant offset. V is also only identifiable where the density is not
negligible: in the far tails the measured signal is below the noise and any
potential is compatible with the data. Errors are reported on the region where
the true ground-state density exceeds a fraction of its peak: 10 % ("core")
by default, and 1 % ("extended") as a harder test that reaches into the
low-signal flanks.
"""
from __future__ import annotations

import numpy as np


def identifiable_mask(rho_ground: np.ndarray, frac: float = 0.1) -> np.ndarray:
    return rho_ground > frac * rho_ground.max()


def align_offset(V_pred: np.ndarray, V_true: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return V_pred shifted by the constant that best matches V_true on mask."""
    c = np.mean(V_true[mask] - V_pred[mask])
    return V_pred + c


def potential_error(V_pred: np.ndarray, V_true: np.ndarray, mask: np.ndarray) -> float:
    """Relative L2 error of V on the identifiable region, offset removed.

    Normalised by the variation of the true potential on that region, so the
    metric does not depend on the arbitrary zero of energy.
    """
    Va = align_offset(V_pred, V_true, mask)
    num = np.sqrt(np.mean((Va[mask] - V_true[mask]) ** 2))
    den = np.sqrt(np.mean((V_true[mask] - V_true[mask].mean()) ** 2))
    return float(num / den)


def max_abs_error(V_pred: np.ndarray, V_true: np.ndarray, mask: np.ndarray) -> float:
    Va = align_offset(V_pred, V_true, mask)
    return float(np.max(np.abs(Va[mask] - V_true[mask])))
