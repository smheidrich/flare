"""Basis functions for many body kernels."""
import numpy as np
from numba import njit
from math import exp, cos, sin, pi


# -----------------------------------------------------------------------------
#                               radial functions
# -----------------------------------------------------------------------------

@njit
def behler_radial(r_ij, n, n_total, sigma, r_cut):
    """Returns a radial symmetry function and its derivative. The basis \
consists of n equispaced Gaussians with standard deviation sigma whose means \
range from 0 to r_cut. A smooth cosine envelope is applied to eliminate \
discontinuous behavior when atoms enter or exit the cutoff sphere.

See Behler, Jörg, and Michele Parrinello. PRL 98.14 (2007): 146401."""

    gauss_mean = (n / (n_total - 1)) * r_cut
    gauss_arg = ((r_ij - gauss_mean)**2) / (2 * sigma * sigma)
    gauss_val = exp(gauss_arg)
    gauss_derv = gauss_val * (r_ij - gauss_mean) / (sigma * sigma)

    cos_val = (1/2) * (cos(pi * r_ij / r_cut) + 1)
    cos_derv = -pi * sin(pi * r_ij / r_cut) / (2 * r_cut)

    basis_val = gauss_val * cos_val
    basis_derv = gauss_derv * cos_val + gauss_val * cos_derv

    return basis_val, basis_derv


# -----------------------------------------------------------------------------
#                              angular functions
# -----------------------------------------------------------------------------

@njit
def cos_grad(cos_theta, bond_vec_j, bond_vec_k):
    """Compute gradient of cos(theta_ijk) with respect to the Cartesian \
coordinates of atoms i, j, and k."""

    grad_vals = np.zeros((3, 3))
    cos_term_j = (cos_theta / bond_vec_j[0]) - (1 / bond_vec_k[0])
    cos_term_k = (cos_theta / bond_vec_k[0]) - (1 / bond_vec_j[0])

    for n in range(3):
        grad_vals[0, n] = \
            bond_vec_j[n+1] * cos_term_j + bond_vec_k[n+1] * cos_term_k

        grad_vals[1, n] = bond_vec_j[n+1] / bond_vec_k[0] - \
            bond_vec_k[n+1] * cos_theta / bond_vec_k[0]

        grad_vals[2, n] = bond_vec_k[n+1] / bond_vec_j[0] - \
            bond_vec_j[n+1] * cos_theta / bond_vec_j[0]

    return grad_vals


@njit
def legendre(n, x):
    """Returns Legendre polynomial and first derivative."""

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

if __name__ == '__main__':
    from flare import struc, env
    from copy import deepcopy

    cell = np.eye(3) * 100
    species = np.array([1, 1, 1])
    positions = np.random.rand(3, 3)
    delt = 1e-8
    cutoffs = np.array([10, 10])
    structure = struc.Structure(cell, species, positions)
    test_env = env.AtomicEnvironment(structure, 0, cutoffs,
                                     compute_angles=True)

    # perturb central atom
    pos_delt_1 = deepcopy(positions)
    coord_1 = np.random.randint(0, 3)
    pos_delt_1[0, coord_1] += delt
    structure_1 = struc.Structure(cell, species, pos_delt_1)
    test_env_1 = env.AtomicEnvironment(structure_1, 0, cutoffs,
                                       compute_angles=True)

    cos_theta = test_env.cos_thetas[0, 1]
    cos_theta_1 = test_env_1.cos_thetas[0, 1]
    bond_vec_j = test_env.bond_array_2[0]
    bond_vec_k = test_env.bond_array_2[1]

    cos_delt_1 = (cos_theta_1 - cos_theta) / delt
    cos_grad_val = cos_grad(cos_theta, bond_vec_j, bond_vec_k)
    assert(np.isclose(cos_delt_1, cos_grad_val[0, coord_1]))

    # # perturb environment atom
    # pos_delt_2 = deepcopy(positions)
    # pert_atom = np.random.randint(1, 3)
    # coord_2 = np.random.randint(0, 3)
    # pos_delt_2[pert_atom, coord_2] += delt
    # structure_2 = struc.Structure(cell, species, pos_delt_2)
    # test_env_2 = env.AtomicEnvironment(structure_2, 0, cutoffs,
    #                                    compute_angles=True)

    # print(cos_delt_1)
    # print(cos_grad(cos_theta, bond_vec_j, bond_vec_k))

