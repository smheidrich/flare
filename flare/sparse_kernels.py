"""Kernels for sparse GPs."""
import numpy as np
from flare import mc_simple, env


# -----------------------------------------------------------------------------
#                            two plus three body
# -----------------------------------------------------------------------------

def two_plus_three_env(env1, env2, hyps, cutoffs):
    """Two plus three body covariance between the local energies of two \
atomic environments."""

    two_term = \
        mc_simple.two_body_mc_en(env1, env2, hyps, cutoffs)
    three_term = \
        mc_simple.three_body_mc_en(env1, env2, hyps, cutoffs)
    return (two_term / 4) + (three_term / 9)


def two_plus_three_struc(env1, struc1, hyps, cutoffs):
    """Two plus three body covariance between the local energy of an atomic \
environment and the energy and force labels on a structure of atoms."""

    noa = struc1.nat
    kernel_vector = np.zeros(len(struc1.labels))
    index = 0

    # if there's an energy label, compute energy kernel
    if struc1.energy is not None:
        en_kern = 0
        for n in range(noa):
            env_curr = env.AtomicEnvironment(struc1, n, cutoffs)
            two_term = \
                mc_simple.two_body_mc_en(env1, env_curr, hyps, cutoffs)
            three_term = \
                mc_simple.three_body_mc_en(env1, env_curr, hyps, cutoffs)

            en_kern += (two_term / 4) + (three_term / 9)

        kernel_vector[index] = en_kern
        index += 1

    # if there are force labels, compute force kernels
    if struc1.forces is not None:
        for n in range(noa):
            env_curr = env.AtomicEnvironment(struc1, n, cutoffs)
            for d in range(3):
                force_kern = \
                    mc_simple.two_plus_three_mc_force_en(env_curr, env1, d+1,
                                                         hyps, cutoffs)
                kernel_vector[index] = force_kern
                index += 1

    return kernel_vector

if __name__ == '__main__':
    from flare import struc

    cell = np.random.rand(3, 3)
    noa = 10
    positions = np.random.rand(noa, 3)
    species = ['Al'] * len(positions)

    energy = np.random.rand()
    forces = np.random.rand(noa, 3)

    test_struc = struc.Structure(cell, species, positions,
                                 energy=energy, forces=forces)

    assert(energy == test_struc.labels[0])
    assert(forces[-1][-2] == test_struc.labels[-2])
