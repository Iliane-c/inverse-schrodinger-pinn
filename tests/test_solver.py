import numpy as np

from invpinn import get_potential, solve_eigenstates


def test_harmonic_spectrum():
    ref = solve_eigenstates(get_potential("harmonic"), L=8.0, n=3000, k=4)
    np.testing.assert_allclose(ref.energies, [0.5, 1.5, 2.5, 3.5], atol=1e-4)


def test_normalisation_and_sign():
    ref = solve_eigenstates(get_potential("double_well"), L=4.5, n=1500, k=2)
    h = ref.x[1] - ref.x[0]
    np.testing.assert_allclose(ref.densities.sum(axis=1) * h, 1.0, atol=1e-10)
    assert (ref.psi[0] > -1e-12).all(), "ground state must be nodeless and positive"
    assert np.sum(np.diff(np.sign(ref.psi[1][np.abs(ref.psi[1]) > 1e-6])) != 0) == 1, "first excited state has one node"


def test_harmonic_ground_state_shape():
    ref = solve_eigenstates(get_potential("harmonic"), L=8.0, n=3000, k=1)
    exact = np.pi ** -0.25 * np.exp(-ref.x**2 / 2)
    np.testing.assert_allclose(ref.psi[0], exact, atol=1e-4)
