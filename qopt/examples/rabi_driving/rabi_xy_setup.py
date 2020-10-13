"""
This file contains the physical constants, operators and convenience functions
for the simulation of rabi drive.

"""

import numpy as np
from qopt import *


# ##################### 1. Implementation Choices #############################

# dense control matrices are faster in low dimensions than sparse matrices
control_matrix = DenseOperator
# We use the scipy linalg expm_frechet implementation to calculate matrix
# exponentials
exponential_method = 'Frechet'

# ##################### 2. Constants ##########################################


# The time steps could be chosen more densely giving more flexibility at the
# cost of computational time
n_time_samples = 10
total_time = 1
time_step = total_time / n_time_samples
rabi_frequency_max = 2 * np.pi

oversampling = 5
awg_rise_time = time_step * .2
lin_freq_rel = 1.
sigma_eps = 1 * lin_freq_rel

# ##################### 3. Operators ##########################################

sigma_0 = control_matrix.pauli_0()
sigma_x = control_matrix.pauli_x()
sigma_y = control_matrix.pauli_y()
sigma_z = control_matrix.pauli_z()

# we implement the control on the x and y axis
h_ctrl = [.5 * sigma_x, .5 * sigma_y]
# We could ad a drift hamiltonian by detuning from the resonance frequency.
h_drift = 0 * .5 * sigma_z

h_noise = .5 * sigma_z

x_half = sigma_x.exp(tau=np.pi * .25j)
y_half = sigma_y.exp(tau=np.pi * .25j)

# ##################### 4. Transfer Function ##################################

# 4.1: Exponential Transfer Function
# This transfer function assumes an exponential saturation of voltages.
transfer_function = ExponentialTF(
    awg_rise_time=awg_rise_time,
    oversampling=oversampling,
    num_ctrls=2
)

# 4.2: Identity
# No transfer function. Here we assume ideal control electronics.
# transfer_function = OversamplingTF(oversampling=oversampling, num_ctrls=2)

# ##################### 5. Amplitude Function #################################

# 5.1: x,y control
lin_amp_func = UnaryAnalyticAmpFunc(
    value_function=lambda x: lin_freq_rel * x,
    derivative_function=lambda x: lin_freq_rel)

# ##################### 6. Noise Trace Generator ##############################

# 6.1 1/f noise spectrum
# until Tom provides the exact numbers I will go with
S_01 = 3e8
S_02 = 3e4
# S(f) = S_01 / f + S_02 / f^2

f_qs_min = 1 / 10 / 60  # 1 over 10 minutes
f_qs_max = 1 / (n_time_samples * time_step)

variance_f = S_01 * (np.log(f_qs_max) - np.log(f_qs_min)) \
    - S_02 * (1 / f_qs_max - 1 / f_qs_min)
sigma_f = np.sqrt(variance_f)


def toms_spectral_noise_density(f, white_floor=1e6):
    """ For fast noise only. I. e. without the 1/f^2 terms. """
    if not isinstance(f, np.ndarray):
        f = np.asarray(f)
    is_white = (f > white_floor).astype(int)
    noise_density = S_01 / f * (1 - is_white)
    noise_density += (S_01 / white_floor) * is_white
    return noise_density


# The noise trace generator explicitly simulates noise realizations
ntg_one_over_f_noise = NTGColoredNoise(
    noise_spectral_density=toms_spectral_noise_density,
    dt=(time_step / oversampling),
    n_samples_per_trace=n_time_samples * oversampling,
    n_traces=1000,
    n_noise_operators=1,
    always_redraw_samples=True
)


# 6.2 quasi static contribution
# for the remaining quasi static noise contribution, we integrate the spectral
# density from 10^-3 Hz to 1 / (time_step / oversampling)
ntg_quasi_static = NTGQuasiStatic(
    standard_deviation=[sigma_f, ],
    n_samples_per_trace=n_time_samples * oversampling,
    n_traces=8,
    always_redraw_samples=False,
    sampling_mode='uncorrelated_deterministic')

# ##################### 7. Time Slot Computer #################################
# The time slot computer calculates the evolution of the qubit taking into
# account the amplitude and transfer function and also the noise traces if
# required.

# 7.1 xy-control
solver_unperturbed_xy = SchroedingerSolver(
    h_drift=[0 * h_drift, ],
    h_ctrl=h_ctrl,
    initial_state=DenseOperator(np.eye(2)),
    tau=time_step * np.ones(n_time_samples),
    is_skew_hermitian=True,
    exponential_method=exponential_method,
    transfer_function=transfer_function,
    amplitude_function=lin_amp_func
)

solver_qs_noise_xy = SchroedingerSMonteCarlo(
    h_drift=[0 * h_drift, ],
    h_ctrl=h_ctrl,
    h_noise=[h_noise, ],
    noise_trace_generator=ntg_quasi_static,
    initial_state=DenseOperator(np.eye(2)),
    tau=time_step * np.ones(n_time_samples),
    is_skew_hermitian=True,
    exponential_method=exponential_method,
    transfer_function=transfer_function,
    amplitude_function=lin_amp_func
)


solver_colored_noise_xy = SchroedingerSMonteCarlo(
    h_drift=[0 * h_drift, ],
    h_ctrl=h_ctrl,
    h_noise=[h_noise, ],
    noise_trace_generator=ntg_one_over_f_noise,
    initial_state=DenseOperator(np.eye(2)),
    tau=time_step * np.ones(n_time_samples),
    is_skew_hermitian=True,
    exponential_method=exponential_method,
    transfer_function=transfer_function,
    amplitude_function=lin_amp_func
)


# ##################### 8. Cost Function ######################################
# The cost functions calculate the infidelities and are minimized by the
# optimiser.

# 8.1 xy-control
entanglement_infid_xy = OperationInfidelity(
    solver=solver_qs_noise_xy,
    target=x_half,
    fidelity_measure='entanglement',
    index=['Entanglement Fidelity XY-Control']
)

entanglement_infid_qs_noise_xy = OperationNoiseInfidelity(
    solver=solver_qs_noise_xy,
    target=y_half,
    fidelity_measure='entanglement',
    index=['Entanglement Fidelity QS-Noise XY-Control'],
    neglect_systematic_errors=True
)

entanglement_infid_colored_noise_xy = OperationNoiseInfidelity(
    solver=solver_colored_noise_xy,
    target=x_half,
    fidelity_measure='entanglement',
    index=['Entanglement Fidelity 1-over-f-Noise XY-Control'],
    neglect_systematic_errors=True
)


# ##################### 8. Convenience Functions ##############################
amp_bound = rabi_frequency_max * 2 * np.pi / lin_freq_rel
bounds_xy = [[0, amp_bound]] * (n_time_samples * len(h_ctrl))
bounds_xy_least_sq = [np.zeros((n_time_samples * len(h_ctrl))),
                      amp_bound * np.ones((n_time_samples * len(h_ctrl)))]


def random_xy_init_pulse(seed=None):
    if seed:
        np.random.seed(seed)
    return np.random.rand(n_time_samples, len(h_ctrl)) * amp_bound


