"""Reference forward solver: finite-difference eigenstates of H = -1/2 d^2/dx^2 + V(x)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import eigh_tridiagonal


@dataclass
class Eigenstates:
    x: np.ndarray          # interior grid points, shape (n,)
    energies: np.ndarray   # shape (k,)
    psi: np.ndarray        # normalised wavefunctions, shape (k, n)

    @property
    def densities(self) -> np.ndarray:
        return self.psi**2


def solve_eigenstates(V: Callable[[np.ndarray], np.ndarray], L: float, n: int = 2000, k: int = 3) -> Eigenstates:
    """Lowest k eigenstates on [-L, L] with Dirichlet boundary conditions.

    Second-order central differences; the error on eigenvalues scales as h^2.
    Wavefunctions are normalised so that sum(psi^2) * h = 1, and each state is
    given a deterministic sign (positive at its first significant lobe).
    """
    x = np.linspace(-L, L, n + 2)[1:-1]
    h = x[1] - x[0]
    diag = 1.0 / h**2 + V(x)
    off = -0.5 / h**2 * np.ones(n - 1)
    energies, vecs = eigh_tridiagonal(diag, off, select="i", select_range=(0, k - 1))
    psi = vecs.T / np.sqrt(h)
    for i in range(k):
        first = np.argmax(np.abs(psi[i]) > 1e-3 * np.abs(psi[i]).max())
        if psi[i, first] < 0:
            psi[i] *= -1
    return Eigenstates(x=x, energies=energies, psi=psi)
