"""Classical baselines for density-to-potential inversion.

For a nodeless ground state psi = sqrt(rho), the Schrodinger equation gives the
exact inversion formula

    V(x) - E0 = psi''(x) / (2 psi(x)) = rho''/(4 rho) - (rho')^2 / (8 rho^2).

It requires the second derivative of measured data, which amplifies noise, and
divides by rho, which explodes in the tails. The baselines differ only in how
the derivatives are estimated.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline

RHO_FLOOR = 1e-12


def _invert(rho: np.ndarray, d1: np.ndarray, d2: np.ndarray, floor: float = RHO_FLOOR) -> np.ndarray:
    rho = np.maximum(rho, floor)
    return d2 / (4 * rho) - d1**2 / (8 * rho**2)


def naive_inversion(x_obs: np.ndarray, rho_obs: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Finite differences on the raw measurements, then linear interpolation."""
    d1 = np.gradient(rho_obs, x_obs)
    d2 = np.gradient(d1, x_obs)
    V = _invert(rho_obs, d1, d2)
    return np.interp(x_eval, x_obs, V)


def spline_inversion(
    x_obs: np.ndarray,
    rho_obs: np.ndarray,
    sigma: float,
    x_eval: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Smoothing-spline fit of the density, then analytic derivatives (discrepancy principle).

    The smoothing factor uses the true noise level (s = n sigma^2), the standard
    discrepancy-principle choice. This gives the baseline oracle knowledge of the
    noise, which the PINN does not have.
    """
    return _spline_V(x_obs, rho_obs, sigma, x_eval, k, 1.0)


def _spline_V(x_obs, rho_obs, sigma, x_eval, k, s_factor):
    s = s_factor * len(x_obs) * sigma**2
    spl = UnivariateSpline(x_obs, rho_obs, k=k, s=s)
    rho = spl(x_eval)
    # Densities below the noise level carry no information; flooring them at
    # sigma keeps the inversion finite instead of dividing by ~0.
    return _invert(rho, spl.derivative(1)(x_eval), spl.derivative(2)(x_eval), floor=max(sigma, RHO_FLOOR))


def spline_inversion_oracle(
    x_obs: np.ndarray,
    rho_obs: np.ndarray,
    sigma: float,
    x_eval: np.ndarray,
    V_true: np.ndarray,
    mask: np.ndarray,
    k: int = 5,
    factors: np.ndarray = np.logspace(-1, 1.5, 11),
) -> np.ndarray:
    """Smoothing spline whose smoothing factor is tuned against the ground truth.

    This is an upper bound on what the spline approach can achieve: in a real
    experiment V_true is unknown and this tuning is impossible.
    """
    from .metrics import potential_error

    best, best_err = None, np.inf
    for f in factors:
        try:
            V = _spline_V(x_obs, rho_obs, sigma, x_eval, k, f)
        except Exception:  # pragma: no cover - spline fit failures
            continue
        err = potential_error(V, V_true, mask)
        if err < best_err:
            best, best_err = V, err
    return best
