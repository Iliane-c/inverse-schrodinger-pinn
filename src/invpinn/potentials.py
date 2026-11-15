"""Benchmark potentials (atomic units: hbar = m = 1).

Each entry gives the potential as a NumPy-compatible callable and the half-width
L of the computational domain [-L, L].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


@dataclass(frozen=True)
class Potential:
    name: str
    func: Callable[[np.ndarray], np.ndarray]
    L: float
    description: str

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.func(np.asarray(x, dtype=float))


def _harmonic(x: np.ndarray) -> np.ndarray:
    return 0.5 * x**2


def _double_well(x: np.ndarray, b: float = 1.6, tilt: float = 0.1) -> np.ndarray:
    # Barrier height b^2/4 = 0.64 at x = 0, minima near x = -b and x = +b.
    return (x**2 - b**2) ** 2 / (4 * b**2) + tilt * x


def _trap_with_barrier(x: np.ndarray, height: float = 3.0, width: float = 0.5) -> np.ndarray:
    # Harmonic trap split by a Gaussian barrier: the standard geometry of a
    # cold-atom "double-well" made with a blue-detuned laser sheet.
    return 0.5 * x**2 + height * np.exp(-(x**2) / (2 * width**2))


POTENTIALS: Dict[str, Potential] = {
    "harmonic": Potential("harmonic", _harmonic, 5.0, "V = x^2 / 2"),
    "double_well": Potential(
        "double_well",
        _double_well,
        4.5,
        "Asymmetric double well V = (x^2 - b^2)^2 / (4 b^2) + 0.1 x, b = 1.6",
    ),
    "trap_with_barrier": Potential(
        "trap_with_barrier",
        _trap_with_barrier,
        5.0,
        "Harmonic trap + Gaussian barrier V = x^2/2 + 3 exp(-x^2 / 0.5)",
    ),
}


def get_potential(name: str) -> Potential:
    try:
        return POTENTIALS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown potential '{name}'. Choose from {sorted(POTENTIALS)}") from exc
