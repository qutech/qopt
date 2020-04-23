"""
This file contains the physical constants, operators and convenience functions
for the simulation of rabi drive.

"""

import numpy as np
from matrix import OperatorDense
from transfer_function import ExponentialTF, IdentityTF, \
    LinearTF, ParallelTF, ConcatenateTF
from amplitude_functions import UnaryAnalyticAmpFunc, \
    CustomAmpFunc
from solver_algorithms import SchroedingerSolver, SchroedingerSMonteCarlo
from noise import NTGQuasiStatic, NTGColoredNoise
from cost_functions import OperationInfidelity, \
    OperationNoiseInfidelity


# ##################### 1. Implementation Choices ##############################

# dense control matrices are faster in low dimensions than sparse matrices
control_matrix = OperatorDense
# We use the scipy linalg expm_frechet implementation to calculate matrix
# exponentials
exponential_method = 'Frechet'

# ##################### 2. Constants ###########################################

# 2.1: specificaly required for this simulation:
phase_max = 100  # degree
phase_min = 0  # degree

# The time steps could be chosen more densely giving more flexibility at the
# cost of computational time.
time_step = 6 * 25e-9  # seconds
rabi_frequency_max = 1e6  # Hertz

# 2.2: from our group
lin_freq_rel = 5.614e-4 * 1e6 * 1e3 * 2 * np.pi  # 2 pi * Hertz / Volt

# 2.3: Assumed
n_time_samples = 10
oversampling = 5
awg_rise_time = time_step * .2
sigma_eps = 1
sigma_rabi = sigma_eps * lin_freq_rel

# ##################### 3. Operators ###########################################

sigma_0 = control_matrix(np.asarray([[1, 0], [0, 1]]))
sigma_x = control_matrix(np.asarray([[0, 1], [1, 0]]))
sigma_y = control_matrix(np.asarray([[0, -1j], [1j, 0]]))
sigma_z = control_matrix(np.asarray([[1, 0], [0, -1]]))

# we implement the control on the x and y axis
h_ctrl = [.5 * sigma_x, .5 * sigma_y]
# We could ad a drift hamiltonian by detuning from the resonance frequency.
h_drift = .5 * sigma_z

x_half = sigma_x.exp(tau=np.pi * .25j)
y_half = sigma_y.exp(tau=np.pi * .25j)

# ##################### 4. Transfer Function ###################################

# 4.1: Exponential Transfer Function
# This transfer function assumes an exponential saturation of voltages.
exponential_transfer_function = ExponentialTF(
    awg_rise_time=awg_rise_time,
    oversampling=oversampling,
    num_ctrls=2
)
exponential_transfer_function.set_times(time_step * np.ones(n_time_samples))

# 4.2: Identity
# No transfer function. Here we assume ideal control electronics.
identity_transfer_function = IdentityTF(oversampling=oversampling, num_ctrls=2)
identity_transfer_function.set_times(time_step * np.ones(n_time_samples))

# ##################### 5. Amplitude Function ##################################

# 5.1: x,y control
lin_amp_func = UnaryAnalyticAmpFunc(
    value_function=lambda x: lin_freq_rel * x,
    derivative_function=lambda x: lin_freq_rel)

# 5.2: phase control


def amp_func_phase_control(x):
    amplitudes = np.zeros_like(x)
    assert x.shape == (n_time_samples * oversampling, 2)
    amplitudes[:, 0] = lin_freq_rel * x[:, 0] * np.cos(x[:, 1])
    amplitudes[:, 1] = lin_freq_rel * x[:, 0] * np.sin(x[:, 1])
    return amplitudes


def amp_func_deriv_phase_control(x):
    assert x.shape == (n_time_samples * oversampling, 2)
    derivs = np.zeros((n_time_samples * oversampling, 2, 2))
    derivs[:, 0, 0] = lin_freq_rel * np.cos(x[:, 1])
    derivs[:, 0, 1] = lin_freq_rel * np.sin(x[:, 1])
    derivs[:, 1, 0] = lin_freq_rel * -1 * x[:, 0] * np.sin(x[:, 1])
    derivs[:, 1, 1] = lin_freq_rel * x[:, 0] * np.cos(x[:, 1])
    return derivs


phase_ctrl_amp_func = CustomAmpFunc(
    value_function=amp_func_phase_control,
    derivative_function=amp_func_deriv_phase_control
)

# ##################### 6. Noise Trace Generator ###############################

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

# ##################### 7. Time Slot Computer ##################################
# The time slot computer calculates the evolution of the qubit taking into
# account the amplitude and transfer function and also the noise traces if
# required.

# 7.1 xy-control
time_slot_comp_unperturbed_xy = SchroedingerSolver(
    h_drift=[0 * h_drift, ] * n_time_samples * oversampling,
    h_ctrl=h_ctrl,
    initial_state=OperatorDense(np.eye(2)),
    tau=[time_step / oversampling, ] * n_time_samples * oversampling,
    is_skew_hermitian=True,
    exponential_method=exponential_method,
    transfer_function=exponential_transfer_function,
    amplitude_function=lin_amp_func
)

time_slot_comp_qs_noise_xy = SchroedingerSMonteCarlo(
    h_drift=[0 * h_drift, ] * n_time_samples * oversampling,
    h_ctrl=h_ctrl,
    h_noise=[h_drift, ],
    noise_trace_generator=ntg_quasi_static,
    initial_state=OperatorDense(np.eye(2)),
    tau=[time_step / oversampling, ] * n_time_samples * oversampling,
    is_skew_hermitian=True,
    exponential_method=exponential_method,
    transfer_function=exponential_transfer_function,
    amplitude_function=lin_amp_func
)

time_slot_comp_qs_noise_xy_spectral = SchroedingerSMonteCarlo(
    h_drift=[0 * h_drift, ] * n_time_samples * oversampling,
    h_ctrl=h_ctrl,
    h_noise=[h_drift, ],
    noise_trace_generator=ntg_quasi_static,
    initial_state=OperatorDense(np.eye(2)),
    tau=[time_step / oversampling, ] * n_time_samples * oversampling,
    is_skew_hermitian=True,
    exponential_method='spectral',
    transfer_function=exponential_transfer_function,
    amplitude_function=lin_amp_func
)


time_slot_comp_colored_noise_xy = SchroedingerSMonteCarlo(
    h_drift=[0 * h_drift, ] * n_time_samples * oversampling,
    h_ctrl=h_ctrl,
    h_noise=[h_drift, ],
    noise_trace_generator=ntg_one_over_f_noise,
    initial_state=OperatorDense(np.eye(2)),
    tau=[time_step / oversampling, ] * n_time_samples * oversampling,
    is_skew_hermitian=True,
    exponential_method=exponential_method,
    transfer_function=exponential_transfer_function,
    amplitude_function=lin_amp_func
)

# 7.2 phase-control
time_slot_comp_unperturbed_phase_control = SchroedingerSolver(
    h_drift=[0 * h_drift, ] * n_time_samples * oversampling,
    h_ctrl=h_ctrl,
    initial_state=OperatorDense(np.eye(2)),
    tau=[time_step / oversampling, ] * n_time_samples * oversampling,
    is_skew_hermitian=True,
    exponential_method=exponential_method,
    transfer_function=exponential_transfer_function,
    amplitude_function=phase_ctrl_amp_func
)

time_slot_comp_qs_noise_phase_control = SchroedingerSMonteCarlo(
    h_drift=[0 * h_drift, ] * n_time_samples * oversampling,
    h_ctrl=h_ctrl,
    h_noise=[h_drift, ],
    noise_trace_generator=ntg_quasi_static,
    initial_state=OperatorDense(np.eye(2)),
    tau=[time_step / oversampling, ] * n_time_samples * oversampling,
    is_skew_hermitian=True,
    exponential_method=exponential_method,
    transfer_function=exponential_transfer_function,
    amplitude_function=phase_ctrl_amp_func
)

time_slot_comp_colored_noise_phase_control = SchroedingerSMonteCarlo(
    h_drift=[0 * h_drift, ] * n_time_samples * oversampling,
    h_ctrl=h_ctrl,
    h_noise=[h_drift, ],
    noise_trace_generator=ntg_one_over_f_noise,
    initial_state=OperatorDense(np.eye(2)),
    tau=[time_step / oversampling, ] * n_time_samples * oversampling,
    is_skew_hermitian=True,
    exponential_method=exponential_method,
    transfer_function=exponential_transfer_function,
    amplitude_function=phase_ctrl_amp_func
)

# ##################### 8. Cost Function #######################################
# The cost functions calculate the infidelities and are minimized by the
# optimiser.

# 8.1 xy-control
entanglement_infid_xy = OperationInfidelity(
    t_slot_comp=time_slot_comp_qs_noise_xy,
    target=x_half,
    fidelity_measure='entanglement',
    index=['Entanglement Fidelity XY-Control']
)

entanglement_infid_qs_noise_xy = OperationNoiseInfidelity(
    t_slot_comp=time_slot_comp_qs_noise_xy,
    target=y_half,
    fidelity_measure='entanglement',
    index=['Entanglement Fidelity QS-Noise XY-Control'],
    neglect_systematic_errors=True
)

entanglement_infid_xy_spectral = OperationInfidelity(
    t_slot_comp=time_slot_comp_qs_noise_xy_spectral,
    target=x_half,
    fidelity_measure='entanglement',
    index=['Entanglement Fidelity XY-Control']
)

entanglement_infid_qs_noise_xy_spectral = OperationNoiseInfidelity(
    t_slot_comp=time_slot_comp_qs_noise_xy_spectral,
    target=y_half,
    fidelity_measure='entanglement',
    index=['Entanglement Fidelity QS-Noise XY-Control'],
    neglect_systematic_errors=True
)

entanglement_infid_colored_noise_xy = OperationNoiseInfidelity(
    t_slot_comp=time_slot_comp_colored_noise_xy,
    target=x_half,
    fidelity_measure='entanglement',
    index=['Entanglement Fidelity 1-over-f-Noise XY-Control'],
    neglect_systematic_errors=True
)

# 8.2 phase-control
entanglement_infid_phase_control = OperationInfidelity(
    t_slot_comp=time_slot_comp_unperturbed_phase_control,
    target=x_half,
    fidelity_measure='entanglement',
    index=['Entanglement Fidelity Phase Control']
)


# This time slot computer calculates the evolution under quasi static noise.
entanglement_infid_qs_noise_phase_control = OperationNoiseInfidelity(
    t_slot_comp=time_slot_comp_qs_noise_phase_control,
    target=x_half,
    fidelity_measure='entanglement',
    index=['Entanglement Fidelity QS-Noise Phase Control'],
    neglect_systematic_errors=True
)


# This time slot computer calculates the evolution under fast noise.
entanglement_infid_colored_noise_phase_control = OperationNoiseInfidelity(
    t_slot_comp=time_slot_comp_colored_noise_phase_control,
    target=x_half,
    fidelity_measure='entanglement',
    index=['Entanglement Fidelity 1-over-f-Noise Phase Control'],
    neglect_systematic_errors=True
)

# ##################### 8. Convenience Functions ###############################


def create_discrete_classes(n_bit_ph: int, n_bit_amp: int):
    n_max_phase = 2 ** n_bit_ph - 1
    delta_phase = phase_max / n_max_phase * np.pi / 180

    # 2.2: from our group
    amp_bound = rabi_frequency_max * 2 * np.pi / lin_freq_rel

    n_max_amp = 2 ** n_bit_amp - 1
    delta_amp = amp_bound / n_max_amp

    discrete_tf_phase = LinearTF(
        oversampling=1,
        bound_type=None,
        num_ctrls=1,
        linear_factor=delta_phase
    )

    discrete_tf_amp = LinearTF(
        oversampling=1,
        bound_type=None,
        num_ctrls=1,
        linear_factor=delta_amp
    )

    discrete_tf = ParallelTF(
        tf1=discrete_tf_amp,
        tf2=discrete_tf_phase
    )

    total_tf = ConcatenateTF(
        tf1=discrete_tf,
        tf2=exponential_transfer_function
    )
    total_tf.set_times(time_step * np.ones(n_time_samples))

    ts_comp_unperturbed_pc_discrete = SchroedingerSolver(
        h_drift=[0 * h_drift, ] * n_time_samples * oversampling,
        h_ctrl=h_ctrl,
        initial_state=OperatorDense(np.eye(2)),
        tau=[time_step / oversampling, ] * n_time_samples * oversampling,
        is_skew_hermitian=True,
        exponential_method=exponential_method,
        transfer_function=total_tf,
        amplitude_function=phase_ctrl_amp_func
    )

    time_slot_comp_qs_noise_pc_discrete = SchroedingerSMonteCarlo(
        h_drift=[0 * h_drift, ] * n_time_samples * oversampling,
        h_ctrl=h_ctrl,
        h_noise=[h_drift, ],
        noise_trace_generator=ntg_quasi_static,
        initial_state=OperatorDense(np.eye(2)),
        tau=[time_step / oversampling, ] * n_time_samples * oversampling,
        is_skew_hermitian=True,
        exponential_method=exponential_method,
        transfer_function=total_tf,
        amplitude_function=phase_ctrl_amp_func
    )

    qs_noise_pc_discrete = OperationNoiseInfidelity(
        t_slot_comp=time_slot_comp_qs_noise_pc_discrete,
        target=x_half,
        fidelity_measure='entanglement',
        index=['Entanglement Fidelity QS-Noise Phase Control'],
        neglect_systematic_errors=True
    )

    entanglement_infid_pc_discrete = OperationInfidelity(
        t_slot_comp=ts_comp_unperturbed_pc_discrete,
        target=x_half,
        fidelity_measure='entanglement',
        index=['Entanglement Fidelity Phase Control']
    )

    return [ts_comp_unperturbed_pc_discrete,
            time_slot_comp_qs_noise_pc_discrete], \
           [entanglement_infid_pc_discrete, qs_noise_pc_discrete]
