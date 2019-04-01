import numpy as np
import gp
import kernels
import struc
import env
import concurrent.futures

# --------------------------------------------------------------------
#                 -1. define helper functions for prediction
# --------------------------------------------------------------------


def predict_on_structure(structure, gp_model):
    for n in range(structure.nat):
        chemenv = env.AtomicEnvironment(structure, n, gp_model.cutoffs)
        for i in range(3):
            force, var = gp_model.predict(chemenv, i + 1)
            structure.forces[n][i] = float(force)
            structure.stds[n][i] = np.sqrt(np.abs(var))


# --------------------------------------------------------------------
#                     0. set structure information
# --------------------------------------------------------------------

species = ['Al'] * 32
cell = np.eye(3) * 8.092
atom_list = list(range(32))

# --------------------------------------------------------------------
#                     1. set train and test arrays
# --------------------------------------------------------------------

header = '/n/home03/jonpvandermause/Repeat_SCF/'
# header = '/Users/jonpvandermause/Research/GP/otf/paper/Cutoff_Calibration/Repeat_SCF/Repeat_SCF/'

train_positions = ['positions_0.05_0.npy']
train_forces = ['forces_0.05_0.npy']

test_positions = ['positions_0.05_1.npy']
test_forces = ['forces_0.05_1.npy']


# --------------------------------------------------------------------
#                 2. create gp model for cutoff
# --------------------------------------------------------------------

cutoffs = np.array([6, 4])
kernel = kernels.two_plus_three_body
kernel_grad = kernels.two_plus_three_body_grad
hyps = np.array([0.1, 1., 0.1, 1., 0.01])
algo = 'BFGS'
maxiter = 20

noise_pars = np.array([])
std_avgs = np.array([])
test_errs = np.array([])
mse_errs = np.array([])
vars = np.array([])

gp_model = gp.GaussianProcess(kernel, kernel_grad, hyps, cutoffs,
                              opt_algorithm=algo, maxiter=maxiter)

for atom_no in atom_list:
    # add an atom to the training set
    for pos, force in zip(train_positions, train_forces):
        pos_npy = np.load(header+pos)
        frc_npy = np.load(header+force)
        struc_curr = struc.Structure(cell, species, pos_npy)
        gp_model.update_db(struc_curr, frc_npy, [atom_no])

    # train gp
    gp_model.train(True)

    # test gp
    pred_vec = np.array([])
    truth_vec = np.array([])
    std_vec = np.array([])
    for pos, force in zip(test_positions, test_forces):
        pos_npy = np.load(header+pos)
        frc_npy = np.load(header+force)
        struc_curr = struc.Structure(cell, species, pos_npy)
        predict_on_structure(struc_curr, gp_model)
        pred_vec = np.append(pred_vec, np.reshape(struc_curr.forces, -1))
        std_vec = np.append(std_vec, np.reshape(struc_curr.stds, -1))
        truth_vec = np.append(truth_vec, np.reshape(frc_npy, -1))

    noise_curr = gp_model.hyps[-1]
    err_curr = np.mean(np.abs(pred_vec - truth_vec))
    std_avg = np.mean(std_vec)
    mse_curr = np.mean((pred_vec - truth_vec)**2)
    var_curr = np.var(pred_vec - truth_vec)

    print(noise_curr)
    print(err_curr)
    print(std_avg)

    noise_pars = np.append(noise_pars, noise_curr)
    std_avgs = np.append(std_avgs, std_avg)
    test_errs = np.append(test_errs, err_curr)
    mse_errs = np.append(mse_errs, mse_curr)
    vars = np.append(vars, var_curr)

# record likelihood, prediction vector, and hyperparameters
test_err_file = 'test_errs'
std_file = 'std_avgs'
noise_file = 'noise'
mse_file = 'mse'

np.save(test_err_file, test_errs)
np.save(std_file, std_avgs)
np.save(noise_file, noise_pars)
np.save(mse_file, mse_errs)
np.save('vars', vars)
