import numpy as np


class SparseGP:
    """Sparse Gaussian process regression model."""

    def __init__(self, environment_kernel, structure_kernel,
                 kernel_hyps, noise_hyps, cutoffs):
        self.environment_kernel = environment_kernel
        self.structure_kernel = structure_kernel
        self.kernel_hyps = kernel_hyps
        self.noise_hyps = noise_hyps
        self.cutoffs = cutoffs
        self.sparse_environments = []
        self.training_structures = []

        self.K_mm = np.zeros(0)
        self.K_nm = np.zeros(0)
        self.lam = np.zeros(0)

    def add_sparse_point(self, atomic_env):
        """Adds a sparse point to the GP model.

        :param atomic_env: Atomic environment of the sparse point.
        :type atomic_env: env.AtomicEnvironment
        """
        # update list of sparse environments
        self.sparse_environments.append(atomic_env)

        # update K_mm
        prev_mm_size = self.K_mm.shape[0]
        K_mm_updated = np.zeros((prev_mm_size+1, prev_mm_size+1))
        K_mm_updated[:prev_mm_size, :prev_mm_size] = self.K_mm

        for count, sparse_env in enumerate(self.sparse_environments):
            energy_kern = self.environment_kernel(atomic_env, sparse_env)
            K_mm_updated[prev_mm_size, count] = energy_kern
            K_mm_updated[count, prev_mm_size] = energy_kern

        self.K_mm = K_mm_updated

        # update K_nm
        prev_nm_size = self.K_nm.shape
        K_nm_updated = np.zeros((prev_nm_size[0], prev_nm_size[1] + 1))

        index = 0
        for train_struc in self.training_structures:
            struc_kern = self.structure_kernel(sparse_env, train_struc)
            kern_len = len(struc_kern)
            K_nm_updated[index:index+kern_len, prev_nm_size[1]+1] = \
                struc_kern
            index += kern_len

        self.K_nm = K_nm_updated

    def add_structure(self, structure):
        """Adds a training structure to the GP database.

        :param structure: Structure of atoms added to the database.
        :type structure: struc.Structure
        """

        # update list of training structures
        self.training_structures.append(structure)

        # update K_nm
        prev_nm_size = self.K_nm.shape

        # TODO: add len method to struc to conveniently get the number of
        # training labels corresponding to that structure

        # K_nm_updated = np.zeros((prev_nm_size[0], prev_nm_size[1]))
