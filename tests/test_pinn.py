import numpy as np
import torch

from invpinn import InverseSchrodingerPINN, PINNConfig, get_potential, make_dataset, train_pinn
from invpinn.pinn import DTYPE, _d


def test_autograd_second_derivative():
    x = torch.linspace(-2, 2, 50, dtype=DTYPE).reshape(-1, 1).requires_grad_(True)
    y = torch.sin(3 * x)
    yxx = _d(_d(y, x), x)
    np.testing.assert_allclose(yxx.detach().numpy(), (-9 * torch.sin(3 * x)).detach().numpy(), atol=1e-10)


def test_envelope_initialisation_is_harmonic():
    """At init the log-amplitude carries the h = -x^2/2 envelope of the harmonic oscillator."""
    torch.manual_seed(0)
    m = InverseSchrodingerPINN(L=5.0, cfg=PINNConfig(n_states=2))
    with torch.no_grad():
        K = torch.nn.functional.softplus(m.K_raw[0]) * m.L**2 / (2 * np.log(2.0))
    assert abs(K.item() - 12.5) < 1e-9


def test_residual_vanishes_for_exact_harmonic_solution():
    """Plugging the exact psi_0, psi_1 and V of the harmonic oscillator gives zero residuals."""
    torch.manual_seed(0)
    m = InverseSchrodingerPINN(L=5.0, cfg=PINNConfig(n_states=2))
    for net in m.h_nets:   # MLP outputs 0, so h = -x^2/2 exactly
        for p in net.parameters():
            torch.nn.init.zeros_(p)

    class ExactV(torch.nn.Module):
        def forward(self, xt):
            return (0.5 * (xt * 5.0) ** 2 - 0.5) / m.cfg.v_scale   # V - E0 with E0 = 1/2

    m.V_net = ExactV()
    with torch.no_grad():
        m.gap_param.fill_(1.0)
        m.node.fill_(0.0)
    x = torch.linspace(-4, 4, 101, dtype=DTYPE).reshape(-1, 1).requires_grad_(True)
    res = m.residuals(x)
    assert res["r0"].abs().max().item() < 1e-10
    assert res["r1"].abs().max().item() < 1e-10


def test_training_reduces_loss_and_recovers_rough_shape():
    data = make_dataset(get_potential("trap_with_barrier"), n_states=2, n_obs=60, noise=0.01, seed=0)
    cfg = PINNConfig(n_states=2, adam_iters=300, lbfgs_iters=0, width=32, depth=2, log_every=50)
    res = train_pinn(data, cfg)
    # The total loss is not monotone: the excited-state residual is switched on
    # after the warm-up. The data misfit must go down.
    assert res.history[-1]["data"] < 0.2 * res.history[0]["data"]
    V = res.predict_V(np.array([-3.0, -1.2, 1.2, 3.0]))
    assert V[0] > V[1] and V[3] > V[2], "potential should already be confining"
