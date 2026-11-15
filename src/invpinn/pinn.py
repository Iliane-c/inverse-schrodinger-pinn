"""Physics-informed neural network for the inverse stationary Schrodinger problem.

Unknowns
--------
* V(x)                 : the potential, represented by an MLP.
* psi_n(x), n < N      : the measured eigenstates (N = 1 or 2).
* E_1 - E_0            : the energy gap (when two states are used), a free scalar.

The zero of energy is a gauge freedom, so we fix E_0 = 0 and recover V up to a
constant.

Wavefunction ansatz (log-amplitude form)
----------------------------------------
* Ground state (nodeless):       psi_0 = exp(h_0(x))
* First excited state (1 node):  psi_1 = (x - a) exp(h_1(x)),  a learnable

with h_n(x) = MLP_n(x / L) - K_n (x / L)^2. The Gaussian envelope gives the
network the right asymptotic decay from the start; the exponential guarantees
psi_0 > 0. Dividing the Schrodinger equation by exp(h_n) gives residuals that
stay O(1) even where psi is exponentially small:

    r_0 = V - E_0 - (h_0'' + h_0'^2) / 2
    r_1 = -(u (h_1'' + h_1'^2) + 2 h_1') / 2 + (V - E_1) u,   u = x - a

Why a second state matters
--------------------------
With the ground state alone, any positive psi_0 is the exact ground state of
V = E_0 + psi_0''/(2 psi_0): the residual can always be zeroed and the physics
only acts as a structured regulariser. With two states sharing the same V, the
system is over-determined: noise that is inconsistent between the two
densities cannot be explained by any single potential, and is rejected.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

from .data import Dataset

DTYPE = torch.float64


@dataclass
class PINNConfig:
    n_states: int = 2            # number of measured eigenstates used (1 or 2)
    width: int = 64
    depth: int = 3
    v_scale: float = 10.0        # output scale of the potential network
    n_collocation: int = 400
    n_quad: int = 801            # quadrature points for normalisation integrals
    w_data: float = 1.0
    w_res: float = 1e-2
    w_smooth: float = 1e-7       # curvature prior on V
    density_weighting: bool = True   # weight residual r_n by the (detached) model density of state n
    weight_floor: float = 1e-3       # minimum residual weight, relative to the peak density
    adam_iters: int = 2000
    warmup_frac: float = 0.4     # 2-state runs: Adam steps (fraction) before the excited state starts shaping V
    ramp_frac: float = 0.2       # then the coupling is ramped linearly from 0 to 1 over this fraction
    grad_clip: float = 1.0       # max gradient norm during Adam (0 disables)
    lr: float = 3e-3
    lbfgs_iters: int = 300
    seed: int = 0
    log_every: int = 250
    verbose: bool = False


class MLP(nn.Module):
    def __init__(self, width: int, depth: int, out_dim: int = 1):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _d(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


class InverseSchrodingerPINN(nn.Module):
    def __init__(self, L: float, cfg: PINNConfig):
        super().__init__()
        if cfg.n_states not in (1, 2):
            raise ValueError("n_states must be 1 or 2")
        self.L = L
        self.cfg = cfg
        self.V_net = MLP(cfg.width, cfg.depth)
        self.h_nets = nn.ModuleList([MLP(cfg.width, cfg.depth) for _ in range(cfg.n_states)])
        # Envelope strength K_n = softplus(raw) * L^2 / 2, initialised to the
        # harmonic-oscillator value (h = -x^2/2) for a unit trap.
        self.K_raw = nn.Parameter(torch.zeros(cfg.n_states, dtype=DTYPE))
        self.node = nn.Parameter(torch.zeros((), dtype=DTYPE))       # position a of the node of psi_1
        # E_1 - E_0, left unconstrained: a softplus here gets stuck at ~0 when the
        # optimiser passes through negative values early in training.
        self.gap_param = nn.Parameter(torch.tensor(0.5, dtype=DTYPE))
        self.to(DTYPE)

    # ---- fields -----------------------------------------------------------------
    def V(self, x: torch.Tensor) -> torch.Tensor:
        return self.cfg.v_scale * self.V_net(x / self.L)

    def h(self, n: int, x: torch.Tensor) -> torch.Tensor:
        xt = x / self.L
        K = nn.functional.softplus(self.K_raw[n]) * self.L**2 / (2 * math.log(2.0))
        return self.h_nets[n](xt) - K * xt**2

    def psi(self, n: int, x: torch.Tensor) -> torch.Tensor:
        amp = torch.exp(self.h(n, x))
        return amp if n == 0 else (x - self.node) * amp

    @property
    def gap(self) -> torch.Tensor:
        return self.gap_param

    def density(self, n: int, x: torch.Tensor, x_quad: torch.Tensor) -> torch.Tensor:
        Z = torch.trapezoid(self.psi(n, x_quad).squeeze(-1) ** 2, x_quad.squeeze(-1))
        return self.psi(n, x).squeeze(-1) ** 2 / Z

    # ---- physics ----------------------------------------------------------------
    def residuals(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        V = self.V(x)
        out = {}
        h0 = self.h(0, x)
        h0x = _d(h0, x)
        h0xx = _d(h0x, x)
        out["r0"] = V - 0.5 * (h0xx + h0x**2)
        if self.cfg.n_states == 2:
            h1 = self.h(1, x)
            h1x = _d(h1, x)
            h1xx = _d(h1x, x)
            u = x - self.node
            A = -0.5 * (u * (h1xx + h1x**2) + 2 * h1x)
            out["r1"] = A + (V - self.gap) * u
            out["r1_gap_only"] = (A + V * u).detach() - self.gap * u.detach()
        Vx = _d(V, x)
        out["Vxx"] = _d(Vx, x)
        return out


@dataclass
class PINNResult:
    model: InverseSchrodingerPINN
    history: List[Dict[str, float]] = field(default_factory=list)
    train_seconds: float = 0.0

    def predict_V(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=DTYPE).reshape(-1, 1)
            return self.model.V(xt).squeeze(-1).numpy()

    def predict_density(self, n: int, x: np.ndarray) -> np.ndarray:
        m = self.model
        with torch.no_grad():
            xq = torch.linspace(-m.L, m.L, m.cfg.n_quad, dtype=DTYPE).reshape(-1, 1)
            xt = torch.as_tensor(x, dtype=DTYPE).reshape(-1, 1)
            return m.density(n, xt, xq).numpy()

    @property
    def gap(self) -> Optional[float]:
        return float(self.model.gap.detach()) if self.model.cfg.n_states == 2 else None


def train_pinn(data: Dataset, cfg: PINNConfig) -> PINNResult:
    if cfg.n_states > data.n_states:
        raise ValueError("Dataset has fewer measured states than requested")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    L = data.L
    model = InverseSchrodingerPINN(L, cfg)

    x_obs = torch.as_tensor(data.x_obs, dtype=DTYPE).reshape(-1, 1)
    rho_obs = torch.as_tensor(data.rho_obs[: cfg.n_states], dtype=DTYPE)
    scale = rho_obs.max(dim=1).values  # per-state normalisation of the data misfit
    x_quad = torch.linspace(-L, L, cfg.n_quad, dtype=DTYPE).reshape(-1, 1)
    x_col = torch.linspace(-L, L, cfg.n_collocation, dtype=DTYPE).reshape(-1, 1).requires_grad_(True)

    coupling = {"c": 1.0 if (cfg.n_states == 1 or cfg.warmup_frac <= 0) else 0.0}
    x_col_plain = x_col.detach()

    def weight(n: int) -> torch.Tensor:
        """Each state constrains V only where it is actually observed.

        r_n is divided by exp(h_n), so unweighted it would carry equal weight in
        the tails or at a node, where psi_n is not constrained by the data. With
        two states this lets the poorly-determined parts of one wavefunction
        corrupt V. Weighting by rho_n restores the natural |psi_n|^2 weighting of
        the undivided equation, with a small floor for conditioning.
        """
        if not cfg.density_weighting:
            return torch.ones_like(x_col_plain)
        with torch.no_grad():
            rho = model.density(n, x_col_plain, x_quad)
            w = rho / rho.max()
        return (w + cfg.weight_floor).reshape(-1, 1)

    def losses() -> Dict[str, torch.Tensor]:
        data_term = sum(
            torch.mean(((model.density(n, x_obs, x_quad) - rho_obs[n]) / scale[n]) ** 2)
            for n in range(cfg.n_states)
        )
        res = model.residuals(x_col)
        res_term = torch.mean(weight(0) * res["r0"] ** 2)
        if cfg.n_states == 2:
            c = coupling["c"]
            w1 = weight(1)
            if c > 0:
                res_term = res_term + c * torch.mean(w1 * res["r1"] ** 2)
            if c < 1:
                # Before full coupling, the excited-state residual only updates
                # E_1, so the gap tracks the current V without disturbing it.
                res_term = res_term + (1 - c) * torch.mean(w1 * res["r1_gap_only"] ** 2)
        smooth = torch.mean(res["Vxx"] ** 2)
        total = cfg.w_data * data_term + cfg.w_res * res_term + cfg.w_smooth * smooth
        return {"total": total, "data": data_term, "res": res_term, "smooth": smooth}

    history: List[Dict[str, float]] = []

    def log(it: int, ls: Dict[str, torch.Tensor], phase: str) -> None:
        row = {"iter": it, "phase": phase, **{k: float(v.detach()) for k, v in ls.items()}}
        if cfg.n_states == 2:
            row["gap"] = float(model.gap.detach())
        history.append(row)
        if cfg.verbose:
            print(f"[{phase} {it:5d}] total={row['total']:.3e} data={row['data']:.3e} res={row['res']:.3e}")

    t0 = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.adam_iters, 1), eta_min=cfg.lr * 0.03)
    warmup = int(cfg.warmup_frac * cfg.adam_iters)
    ramp = max(int(cfg.ramp_frac * cfg.adam_iters), 1)
    for it in range(cfg.adam_iters):
        if cfg.n_states == 2 and cfg.warmup_frac > 0:
            coupling["c"] = float(min(max((it - warmup) / ramp, 0.0), 1.0))
        opt.zero_grad()
        ls = losses()
        ls["total"].backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        if it % cfg.log_every == 0 or it == cfg.adam_iters - 1:
            log(it, ls, "adam")

    coupling["c"] = 1.0
    if cfg.lbfgs_iters > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=cfg.lbfgs_iters, history_size=50,
            line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14,
        )

        def closure():
            lbfgs.zero_grad()
            ls = losses()
            ls["total"].backward()
            return ls["total"]

        lbfgs.step(closure)
        log(cfg.adam_iters + cfg.lbfgs_iters, losses(), "lbfgs")

    return PINNResult(model=model, history=history, train_seconds=time.time() - t0)


def config_dict(cfg: PINNConfig) -> Dict:
    return asdict(cfg)
