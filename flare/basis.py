"""Basis functions for many body kernels."""
import numpy as np
from numba import njit


# -----------------------------------------------------------------------------
#                               radial functions
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
#                              angular functions
# -----------------------------------------------------------------------------

@njit
def legendre(n, x):
    """Return Legendre polynomial and first derivative."""

    # check that the input lies between -1 and 1
    assert(x >= -1 and x <= 1)

    if n == 0:
        val = 1
        derv = 0

    elif n == 1:
        val = x
        derv = 1

    elif n == 2:
        val = (1/2) * (-1 + 3 * x * x)
        derv = 3 * x

    elif n == 3:
        val = (1/2) * (-3 * x + 5 * x**3)
        derv = (1/2) * (-3 + 15 * x * x)

    elif n == 4:
        val = (1/8) * (3 - 30 * x * x + 35 * x**4)
        derv = (1/8) * (-60 * x + 140 * x**3)

    elif n == 5:
        val = (1/8) * (15 * x - 70 * x**3 + 63 * x**5)
        derv = (1/8) * (15 - 210 * x * x + 315 * x**4)

    elif n == 6:
        val = (1/16) * (-5 + 105 * x * x - 315 * x**4 + 231 * x**6)
        derv = (1/16) * (210 * x - 1260 * x**3 + 1386 * x**5)

    elif n == 7:
        val = (1/16) * (-35 * x + 315 * x**3 - 693 * x**5 + 429 * x**7)
        derv = (1/16) * (-35 + 945 * x * x - 3465 * x**4 + 3003 * x**6)

    elif n == 8:
        val = (1/128) * (35 - 1260 * x * x + 6930 * x**4 - 12012 * x**6 +
                         6435 * x**8)
        derv = (1/128) * (-2520 * x + 27720 * x**3 - 72072 * x**5 +
                          51480 * x**7)

    elif n == 9:
        val = (1/128) * (315 * x - 4620 * x**3 + 18018 * x**5 - 25740 * x**7 +
                         12155 * x**9)
        derv = (1/128) * (315 - 13860 * x * x + 90090 * x**4 - 180180 * x**6 +
                          109395 * x**8)

    elif n == 10:
        val = (1/256) * (-63 + 3465 * x * x - 30030 * x**4 + 90090 * x**6 -
                         109395 * x**8 + 46189 * x**10)
        derv = (1/256) * (6930 * x - 120120 * x**3 + 540540 * x**5 -
                          875160 * x**7 + 461890 * x**9)

    else:
        raise Exception('Not implemented for given value of n.')

    return val, derv
