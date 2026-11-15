"""invpinn: recovering quantum potentials from measured densities with PINNs."""
from .potentials import POTENTIALS, Potential, get_potential
from .solver import Eigenstates, solve_eigenstates
from .data import Dataset, make_dataset
from .baselines import naive_inversion, spline_inversion, spline_inversion_oracle
from .metrics import identifiable_mask, potential_error, max_abs_error, align_offset
from .pinn import PINNConfig, PINNResult, InverseSchrodingerPINN, train_pinn

__version__ = "0.1.0"
