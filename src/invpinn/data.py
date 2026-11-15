"""Synthetic measurement generation.

We simulate what an experiment would record: the probability density
rho_n(x) = |psi_n(x)|^2 of one or several eigenstates, sampled on a finite set of
"pixels" and corrupted by additive Gaussian noise whose standard deviation is a
fraction of the peak density (a simple model of shot noise / camera noise in
absorption imaging).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .potentials import Potential
from .solver import solve_eigenstates


@dataclass
class Dataset:
    potential: Potential
    x_obs: np.ndarray        # (n_obs,)
    rho_obs: np.ndarray      # (n_states, n_obs) noisy densities
    sigma: np.ndarray        # (n_states,) absolute noise standard deviation per state
    x_true: np.ndarray       # fine grid used by the reference solver
    V_true: np.ndarray       # V on x_true
    rho_true: np.ndarray     # (n_states, n_fine) exact densities
    E_true: np.ndarray       # (n_states,) exact energies
    noise: float             # relative noise level
    seed: int

    @property
    def n_states(self) -> int:
        return self.rho_obs.shape[0]

    @property
    def L(self) -> float:
        return self.potential.L


def make_dataset(
    potential: Potential,
    n_states: int = 2,
    n_obs: int = 100,
    noise: float = 0.02,
    seed: int = 0,
    n_solver: int = 2000,
) -> Dataset:
    ref = solve_eigenstates(potential, potential.L, n=n_solver, k=max(n_states, 2))
    rng = np.random.default_rng(seed)
    x_obs = np.linspace(-potential.L, potential.L, n_obs + 2)[1:-1]
    rho_clean = np.stack([np.interp(x_obs, ref.x, ref.densities[i]) for i in range(n_states)])
    sigma = noise * rho_clean.max(axis=1)
    rho_obs = rho_clean + sigma[:, None] * rng.standard_normal(rho_clean.shape)
    return Dataset(
        potential=potential,
        x_obs=x_obs,
        rho_obs=rho_obs,
        sigma=sigma,
        x_true=ref.x,
        V_true=potential(ref.x),
        rho_true=ref.densities[:n_states],
        E_true=ref.energies[:n_states],
        noise=noise,
        seed=seed,
    )
