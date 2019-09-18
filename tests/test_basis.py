"""Tests of many body basis functions."""
import numpy as np
from flare import basis


def test_behler_radial():
    """Test Behler radial derivativer."""

    n_total = np.random.randint(2, 20)
    n = np.random.randint(0, high=n_total)
    sigma = np.random.uniform(0.1, 10)
    r_cut = 5
    r_ij = np.random.uniform(0, r_cut)
    delt = 1e-8

    basis_val, basis_derv = basis.behler_radial(r_ij, n, n_total, sigma, r_cut)
    basis_delt, _ = basis.behler_radial(r_ij + delt, n, n_total, sigma, r_cut)
    derv_delt = (basis_delt - basis_val) / delt

    assert(np.isclose(basis_derv, derv_delt))


def test_legendre():
    """Draw a random integer and do a finite difference test of the \
corresponding Legendre polynomial derivative."""

    n = np.random.randint(0, high=11)
    x = 2 * np.random.uniform() - 1
    delta = 1e-8

    val, derv = basis.legendre(n, x)
    val_delt, _ = basis.legendre(n, x + delta)
    derv_delt = (val_delt - val) / delta

    assert(np.isclose(derv, derv_delt))
