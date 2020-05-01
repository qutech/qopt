import math
import copy
import qutip.control_2.matrix
import qutip.control_2.cost_functions
import numpy as np
import pandas as pd
import filter_functions as ff
import matplotlib.pyplot as plt
import qutip.control_2.tslotcomp
import qutip.control_2.noise


n_t = 2 ** 7

h_drift = [
    qutip.control_2.matrix.ControlDense(np.zeros((2, 2), dtype=complex))
    for _ in range(n_t)]
h_control = np.asarray(
    [[.5 * 2 * math.pi * qutip.control_2.matrix.ControlDense(qutip.sigmax()),
      .5 * 2 * math.pi * qutip.control_2.matrix.ControlDense(qutip.sigmaz())]
        for _ in range(n_t)])
num_ctrl = 2
ctrl_amps = np.asarray(
    [[.5] * (n_t // 4) + [0] * (n_t // 4) + [.25] * (n_t // 4) + [.25] * (
                n_t // 4),
     [0] * (n_t // 4) + [.5] * (n_t // 4) + [0] * (n_t // 4) + [0] * (
                 n_t // 4)]).transfer_matrix
ctrl_amps /= (n_t / 4)
tau = [1] * n_t
initial_state = qutip.control_2.matrix.ControlDense(np.eye(2))


def dial_vectorized(f):
    s = 1e-7 * f ** -.7
    return s


def linear_noise_density(f):
    s = 1e-5 * f
    return s


def white_noise_density(f):
    return np.ones_like(f) * 1e-6


def yoneda_one_over_f_spectral_noise_density(f):
    s = 3e-8 / f
    return s


h_noise = np.asarray(
    [.5 * 2 * math.pi * qutip.control_2.matrix.ControlDense(
        qutip.sigmax())])
filter_function_h_n = [[h_noise[0].data, np.ones(n_t)]]

noise_densities = [dial_vectorized, white_noise_density,
                   yoneda_one_over_f_spectral_noise_density,
                   linear_noise_density]
noise_density_names = ["dial_vectorized", "white_noise_density",
                       "yoneda_one_over_f_spectral_noise_density",
                       "linear_noise_density"]

for name, noise_density_in_use in zip(noise_density_names, noise_densities):

    fast_noise_parameters = qutip.control_2.noise.NTGColoredNoise(
        noise_spectral_density=noise_density_in_use,
        dt=1, n_samples_per_trace=n_t, n_traces=200
    )

    tslot_obj = qutip.control_2.tslotcomp.TSCompSaveAllNoise(
        h_ctrl=h_control, h_drift=h_drift, num_ctrl=num_ctrl, tau=tau,
        initial_state=initial_state, ctrl_amps=ctrl_amps, num_t=n_t,
        h_noise=np.asarray([h_noise, ] * n_t).transfer_matrix,
        noise_trace_generator=fast_noise_parameters,
        filter_function_h_n=filter_function_h_n)

    tslot_obj_spectral = qutip.control_2.tslotcomp.TSCompSaveAllNoise(
        h_ctrl=h_control, h_drift=h_drift, num_ctrl=num_ctrl, tau=tau,
        initial_state=initial_state, ctrl_amps=ctrl_amps, num_t=n_t,
        h_noise=np.asarray([h_noise, ] * n_t).transfer_matrix,
        noise_trace_generator=fast_noise_parameters,
        filter_function_h_n=filter_function_h_n, exponential_method='spectral')

    # Fidelity for the fast noise by average fidelity
    # here as well we can use the QSNoise fidelity

    fidelity_computer_fast_noise_entanglement = \
        qutip.control_2.cost_functions.OperationNoiseInfidelity(
            t_slot_comp=tslot_obj,
            target=initial_state,
            index=['fast_noise_entanglement_fid'],
            use_unitary_derivatives=True,
            neglect_systematic_errors=True,
            fidelity_measure='entanglement'
        )

    fidelity_computer_fast_noise_entanglement_spectral = \
        qutip.control_2.cost_functions.OperationNoiseInfidelity(
            t_slot_comp=tslot_obj_spectral,
            target=initial_state,
            index=['fast_noise_entanglement_fid'],
            use_unitary_derivatives=True,
            neglect_systematic_errors=True,
            fidelity_measure='entanglement'
        )

    # Fidelity by filter function
    omega_min = 1 / n_t
    omega_max = 1 / 1

    omega_lin = np.linspace(omega_min, omega_max / 2, num=n_t // 2)
    s_lin = noise_density_in_use(copy.deepcopy(omega_lin))

    # s_lin, omega_lin = ff.util.symmetrize_spectrum(s_lin, omega_lin)

    omega_log = np.logspace(np.log10(omega_min), np.log10(omega_max / 2),
                            num=n_t // 2, base=10)
    s_log = noise_density_in_use(copy.deepcopy(omega_log))

    s_log, omega_log = ff.util.symmetrize_spectrum(s_log, omega_log)

    pulse_sequence = tslot_obj.create_pulse_sequence()
    infidelity, smallness = ff.infidelity(pulse=pulse_sequence, S=s_log,
                                          omega=omega_log,
                                          return_smallness=True)
    # ff.plotting.plot_filter_function(pulse=pulse_sequence, omega=omega_lin)
    # print('smallness:')
    # print(smallness)
    # omega = ff.util.get_sample_frequencies(pulse=pulse_sequence,
    # n_samples=n_t, spacing='lin', symmetric=True)

    fidelity_computer_ff_log = \
        qutip.control_2.cost_functions.OperatorFilterFunctionInfidelity(
            t_slot_comp=tslot_obj,
            omega=omega_log,
            noise_power_spec_density=s_log,
            target=initial_state
        )

    fidelity_computer_ff_lin = \
        qutip.control_2.cost_functions.OperatorFilterFunctionInfidelity(
            t_slot_comp=tslot_obj,
            omega=omega_lin,
            noise_power_spec_density=s_lin,
            target=initial_state
        )

    a = fidelity_computer_fast_noise_entanglement.costs()
    d = fidelity_computer_fast_noise_entanglement_spectral.costs()
    b = fidelity_computer_ff_log.costs()
    c = fidelity_computer_ff_lin.costs()

    fid_names = ["mc_entanglement", "mc_entanglement_spectral",
                 "ff_symmetrized_log", "ff_lin"]
    index = [name + ind for ind in fid_names]

    s = pd.Series(data=[a, d, b, c], index=index)
    print(s)
