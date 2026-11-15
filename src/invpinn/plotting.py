"""Figures for single reconstructions and benchmark sweeps."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .data import Dataset  # noqa: E402
from .metrics import align_offset, identifiable_mask  # noqa: E402

COLORS = {
    "true": "#111111",
    "spline": "#c2410c",
    "pinn_1state": "#2563eb",
    "pinn_2state": "#059669",
}
LABELS = {
    "spline": "Smoothing spline (oracle-tuned)",
    "pinn_1state": "PINN, ground state only",
    "pinn_2state": "PINN, ground + 1st excited",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 110,
})


def _shade_unidentifiable(ax, x, mask, label=True):
    edges = np.flatnonzero(np.diff(mask.astype(int)))
    lo, hi = x[edges[0] + 1], x[edges[-1]]
    ax.axvspan(x[0], lo, color="#9ca3af", alpha=0.15, lw=0, label="ρ₀ < 10 % of peak" if label else None)
    ax.axvspan(hi, x[-1], color="#9ca3af", alpha=0.15, lw=0)


def plot_reconstruction(
    data: Dataset,
    potentials: Dict[str, np.ndarray],
    fitted_densities: Optional[Dict[int, np.ndarray]] = None,
    history: Optional[Sequence[Dict]] = None,
    errors: Optional[Dict[str, float]] = None,
    path: Optional[str] = None,
):
    x = data.x_true
    core = identifiable_mask(data.rho_true[0], 0.1)
    ncols = 3 if history else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.6))

    ax = axes[0]
    state_colors = ["#2563eb", "#9333ea"]
    for n in range(data.n_states):
        ax.plot(data.x_obs, data.rho_obs[n], "o", ms=2.6, alpha=0.55, color=state_colors[n],
                label=f"measured ρ{n}")
        ax.plot(x, data.rho_true[n], "--", lw=1, color="#111111", alpha=0.6, label="exact" if n == 0 else None)
        if fitted_densities and n in fitted_densities:
            ax.plot(x, fitted_densities[n], lw=1.8, color=state_colors[n], label=f"PINN fit ρ{n}")
    ax.set_title("A · Measured densities", loc="left", fontweight="bold")
    ax.set_xlabel("x")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    _shade_unidentifiable(ax, x, core)
    ax.plot(x, data.V_true, color=COLORS["true"], lw=2.2, label="true V")
    vmin, vmax = data.V_true[core].min(), data.V_true[core].max()
    for key, V in potentials.items():
        Va = align_offset(V, data.V_true, core)
        lab = LABELS.get(key, key)
        if errors and key in errors:
            lab += f" ({100 * errors[key]:.1f} %)"
        ax.plot(x, Va, lw=1.6, color=COLORS.get(key), label=lab)
    pad = 0.25 * (vmax - vmin) + 0.5
    ax.set_ylim(vmin - pad, vmax + pad)
    ax.set_title("B · Recovered potential (offset removed)", loc="left", fontweight="bold")
    ax.set_xlabel("x")
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)

    if history:
        ax = axes[2]
        it = [h["iter"] for h in history]
        ax.semilogy(it, [h["data"] for h in history], label="data misfit")
        ax.semilogy(it, [h["res"] for h in history], label="physics residual")
        ax.set_title("C · Training (2-state PINN)", loc="left", fontweight="bold")
        ax.set_xlabel("iteration")
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle(f"{data.potential.description}   ·   noise {100 * data.noise:.0f} % of peak   ·   "
                 f"{len(data.x_obs)} pixels", fontsize=10, color="#374151")
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_benchmark(df, metric: str = "err_core", path: Optional[str] = None, title: str = ""):
    """Error vs noise, one panel per potential, mean ± std over seeds."""
    pots = list(dict.fromkeys(df["potential"]))
    fig, axes = plt.subplots(1, len(pots), figsize=(4.6 * len(pots), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, pot in zip(axes, pots):
        sub = df[df["potential"] == pot]
        for method in ["spline", "pinn_1state", "pinn_2state"]:
            m = sub[sub["method"] == method]
            if m.empty:
                continue
            mean = m.groupby("noise")[metric].mean()
            # Mean as a line, every seed as a faint dot: honest about the spread
            # and readable on a log axis (unlike symmetric error bars).
            ax.scatter(100 * m["noise"], 100 * m[metric], s=14, alpha=0.35, color=COLORS[method], lw=0)
            ax.plot(100 * mean.index, 100 * mean.values, marker="o", ms=5, lw=1.8,
                    color=COLORS[method], label=LABELS[method])
        ax.set_yscale("log")
        ax.set_xlabel("noise (% of peak density)")
        ax.set_title(pot.replace("_", " "), loc="left", fontweight="bold")
    axes[0].set_ylabel("relative error on V (%)")
    axes[0].legend(fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, fontsize=10, color="#374151")
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return fig
