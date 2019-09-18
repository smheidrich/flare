"""Tests of many body basis functions."""
import numpy as np
from flare import basis


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
