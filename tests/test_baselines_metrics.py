import numpy as np

from invpinn import (
    get_potential,
    identifiable_mask,
    make_dataset,
    potential_error,
    spline_inversion,
)


def test_metric_is_gauge_invariant():
    x = np.linspace(-3, 3, 200)
    V = 0.5 * x**2
    mask = np.ones_like(x, dtype=bool)
    assert potential_error(V + 7.3, V, mask) < 1e-12
    assert potential_error(1.1 * V, V, mask) > 0.05


def test_spline_inversion_is_exact_on_clean_dense_data():
    data = make_dataset(get_potential("harmonic"), n_states=1, n_obs=400, noise=0.0, seed=0)
    core = identifiable_mask(data.rho_true[0], 0.1)
    V = spline_inversion(data.x_obs, data.rho_obs[0], 1e-9, data.x_true)
    assert potential_error(V, data.V_true, core) < 0.02


def test_noise_level_matches_request():
    data = make_dataset(get_potential("harmonic"), n_states=2, n_obs=2000, noise=0.05, seed=1)
    clean = np.stack([np.interp(data.x_obs, data.x_true, data.rho_true[n]) for n in range(2)])
    measured_std = (data.rho_obs - clean).std(axis=1)
    np.testing.assert_allclose(measured_std / data.rho_true.max(axis=1), 0.05, rtol=0.1)
