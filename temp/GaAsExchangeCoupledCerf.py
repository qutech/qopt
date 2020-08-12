"""
This file is meant to reproduce the results of the paper 'A High-Fidelity Gate
Set for Exchange-Coupled Singlet-Triplet Qubits' by Cerfontaine et al.

All energies are given in inverse nano seconds. They denote angular frequencies.
All times in nano seconds.

"""
import numpy as np
import qutip
from qopt.matrix import DenseOperator

INFIDELITIES = {
    'leakage': 1.8e-5,
    'entanglement_bound': 1e-10,
    'fast_white': 1.6e-3,
    'quasi_static_elec': 4.9e-4,
    'quasi_static_magn': 1.7e-4
}

REQUIRED_ACCURACY = {
    'leakage': .1,
    'fast_white_mc': .15,
    'fast_white_me': .31,
    'quasi_static_elec': .1,
    'quasi_static_magn': .1
}

SIGMA_X = DenseOperator(qutip.sigmax())
SIGMA_Y = DenseOperator(qutip.sigmay())
SIGMA_Z = DenseOperator(qutip.sigmaz())
SIGMA_0 = DenseOperator(np.eye(2))

# global constants (Table 1 in the paper)
SIGMA_EPS = 8e-3  # mV
SIGMA_b = .3  # mT
EPSILON_0 = .272  # mV
J_0 = 1  # ns^-1
EPSILON_MIN = -5.4 * EPSILON_0  # mV
EPSILON_MAX = 2.4 * EPSILON_0  # mV
F_S = 1  # GS/s
B_G = 500  # mT

# changing dimensions: 1mT ~ 5.6 MHz ~ 5.6 / mu s ~ 0.0056 / ns
# milli_tesla_to_inv_ns = 5.6e-3 this was master thesis
milli_tesla_to_inv_ns = 6.2e-3
B_G *= milli_tesla_to_inv_ns
SIGMA_b *= milli_tesla_to_inv_ns
SIGMA_b *= 2 * np.pi  # to make it angular freq
# more constants
PADDING_END = 4 * 5  # due to oversampling
d_subspace = 6

AWG_RISE_TIME = .5  # ns
OFFSET = 0
DELTA_T = 1
N_TIME_SLICES = 46

B12 = 1  # in 1 / ns
B23 = 7
B34 = -1
# This has been used by Tobias (understand why? Doesnt change anything.)
B_G = 19.34702098

OVERSAMPLING = 5
N_QS_NOISE_TRACES = 20

# S_0 = 4e-5  # mV^2 / ns unit cancels when divided by epsilon 0
S_0 = 4e-5
S_12 = S_0
S_23 = S_0
S_34 = S_0
S_fast_noise = [S_12, S_23, S_34]

CONSTANTS = {
    'leakage_infid': 1.8e-5,
    'systematic_infid': 1e-10,
    'sigma_eps': SIGMA_EPS,
    'eps0': EPSILON_0,
    's0_white_noise': S_0,
    'sigma_b': SIGMA_b
}


# Operators

# subspace
sub_space_ind = np.ix_([3, 5, 6, 9, 10, 12], [3, 5, 6, 9, 10, 12])
comp_sub_sub_space_ind = [1, 2, 3, 4]

CNOT = np.zeros((d_subspace, d_subspace))
CNOT[0, 0] = 1
CNOT[1, 1] = 1
CNOT[2, 2] = 1
CNOT[3, 4] = 1
CNOT[4, 3] = 1
CNOT[5, 5] = 1
CNOT = DenseOperator(CNOT)

CNOT_4 = CNOT.truncate_to_subspace(comp_sub_sub_space_ind)


def exchange_interaction(amplitude):
    return np.exp(amplitude)


def deriv_exchange_interaction(amplitude):
    return np.exp(amplitude)


sig_1_z = SIGMA_Z.kron(SIGMA_0).kron(SIGMA_0).kron(SIGMA_0)
sig_2_z = SIGMA_0.kron(SIGMA_Z).kron(SIGMA_0).kron(SIGMA_0)
sig_3_z = SIGMA_0.kron(SIGMA_0).kron(SIGMA_Z).kron(SIGMA_0)
sig_4_z = SIGMA_0.kron(SIGMA_0).kron(SIGMA_0).kron(SIGMA_Z)

h_drift = B12 / 8 * ((-3 * sig_1_z) + sig_2_z + sig_3_z + sig_4_z) \
          + B23 / 4 * ((-1 * sig_1_z) - sig_2_z + sig_3_z + sig_4_z) \
          + B34 / 8 * ((-1 * sig_1_z) - sig_2_z - sig_3_z + (3 * sig_4_z))

# h_drift += B_G / 2 * (sig_1_z + sig_2_z + sig_3_z + sig_4_z)
# this contribution is 0 on the subspace.

h_drift = DenseOperator(h_drift.data[sub_space_ind])

h_ctrl = [DenseOperator(np.zeros((2 ** 4, 2 ** 4))) for _ in range(3)]

for pauli in [SIGMA_X, SIGMA_Y, SIGMA_Z]:
    h_ctrl[0] += .25 * pauli.kron(SIGMA_0).kron(SIGMA_0).kron(SIGMA_0) \
                 * SIGMA_0.kron(pauli).kron(SIGMA_0).kron(SIGMA_0)

    h_ctrl[1] += .25 * SIGMA_0.kron(pauli).kron(SIGMA_0).kron(SIGMA_0) \
                 * SIGMA_0.kron(SIGMA_0).kron(pauli).kron(SIGMA_0)

    h_ctrl[2] += .25 * SIGMA_0.kron(SIGMA_0).kron(pauli).kron(SIGMA_0) \
                 * SIGMA_0.kron(SIGMA_0).kron(SIGMA_0).kron(pauli)

for ctrl in range(3):
    h_ctrl[ctrl] = DenseOperator(h_ctrl[ctrl].data[sub_space_ind])
    h_ctrl[ctrl] = h_ctrl[ctrl] - h_ctrl[ctrl].tr() / \
        d_subspace * DenseOperator(np.eye(d_subspace))

initial_state = DenseOperator(np.eye(6))
initial_state_sup_op = DenseOperator(np.eye(36))


h_noise_electric = h_ctrl
h_noise_magnetic = [
    1 / 8 * ((-3 * sig_1_z) + sig_2_z + sig_3_z + sig_4_z),
    1 / 4 * ((-1 * sig_1_z) - sig_2_z + sig_3_z + sig_4_z),
    1 / 8 * ((-1 * sig_1_z) - sig_2_z - sig_3_z + (3 * sig_4_z))]

for i in range(len(h_noise_magnetic)):
    h_noise_magnetic[i] = type(h_noise_magnetic[i])(
        h_noise_magnetic[i].data[sub_space_ind])

lindbladians = [DenseOperator(np.zeros((2 ** 4, 2 ** 4))) for _ in range(3)]

for pauli in [SIGMA_X, SIGMA_Y, SIGMA_Z]:
    lindbladians[0] += pauli.kron(SIGMA_0).kron(SIGMA_0).kron(SIGMA_0) \
                       * SIGMA_0.kron(pauli).kron(SIGMA_0).kron(SIGMA_0)

    lindbladians[1] += SIGMA_0.kron(pauli).kron(SIGMA_0).kron(SIGMA_0) \
                       * SIGMA_0.kron(SIGMA_0).kron(pauli).kron(SIGMA_0)

    lindbladians[2] += SIGMA_0.kron(SIGMA_0).kron(pauli).kron(SIGMA_0) \
                       * SIGMA_0.kron(SIGMA_0).kron(SIGMA_0).kron(pauli)

diss_op = [[], [], []]

for i in range(len(lindbladians)):
    lindbladians[i] = DenseOperator(
        lindbladians[i].data[sub_space_ind])

    diss_op[i] = lindbladians[i].conj(do_copy=True).kron(lindbladians[i])
    diss_op[i] -= .5 * lindbladians[i].identity_like().kron(
        lindbladians[i].dag(do_copy=True) * lindbladians[i])
    diss_op[i] -= .5 * (lindbladians[i].conj(do_copy=True).dag(do_copy=True)
                        * lindbladians[i].conj()).kron(
        lindbladians[i].identity_like())

    normation_factor = 1 / ((2 * np.pi) ** 2) / (EPSILON_0 ** 2)


def create_diss_super_op_fkt(noise=S_fast_noise[0]):
    def diss_super_op_fkt_(_, transferred_parameters: np.ndarray):
        derivatives = deriv_exchange_interaction(transferred_parameters)
        derivatives = derivatives ** 2
        diss_super_op = []
        for t in range(transferred_parameters.shape[0]):
            # the factor of 2pi accounts for the fact that S is given per
            # frequency but we need angular frequencies.
            diss_super_op.append(
                noise * normation_factor * derivatives[
                    t, 0] * diss_op[0])
            for k in range(1, len(diss_op)):
                diss_super_op[-1] += noise * normation_factor \
                                     * derivatives[t, k] * diss_op[k]
        return diss_super_op
    return diss_super_op_fkt_


def diss_super_op_fkt(_, transferred_parameters: np.ndarray):
    derivatives = deriv_exchange_interaction(transferred_parameters)
    derivatives = derivatives ** 2
    diss_super_op = []
    for t in range(transferred_parameters.shape[0]):
        # the factor of 2pi accounts for the fact that S is given per
        # frequency but we need angular frequencies.
        diss_super_op.append(
            S_fast_noise[0] * normation_factor * derivatives[
                t, 0] * diss_op[0])
        for k in range(1, len(diss_op)):
            diss_super_op[-1] += S_fast_noise[k] * normation_factor \
                                 * derivatives[t, k] * diss_op[k]
    return diss_super_op


def diss_super_op_deriv_fkt(_, transferred_parameters: np.ndarray):
    derivatives = deriv_exchange_interaction(transferred_parameters)
    derivatives = (np.abs(derivatives) ** 2) * 2
    diss_super_op_deriv = []
    for t in range(transferred_parameters.shape[0]):
        diss_super_op_deriv.append(
            [S_fast_noise[0] * normation_factor * derivatives[
                t, 0] * diss_op[
                 0], ])
        for k in range(1, len(diss_op)):
            diss_super_op_deriv[-1].append(
                S_fast_noise[k] * normation_factor * derivatives[t, k] *
                diss_op[k])
    return diss_super_op_deriv


OPERATORS = {
    'CNOT_4': CNOT_4,
    'h_ctrl': h_ctrl,
    'h_drift': h_drift,
    'h_noise_magnetic': h_noise_magnetic,
    'initial_state': initial_state,
    'initial_state_sup_op': initial_state_sup_op
}
