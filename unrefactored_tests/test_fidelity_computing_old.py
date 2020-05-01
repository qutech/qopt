import math
import qutip.control_2.matrix
import qutip.control_2.cost_functions
import numpy as np
import filter_functions as ff
import qutip.control_2.tslotcomp
import qutip.control_2.noise

sig_x = qutip.control_2.matrix.ControlDense(qutip.sigmax())
sig_y = qutip.control_2.matrix.ControlDense(qutip.sigmay())

a = qutip.control_2.cost_functions.averge_gate_fidelity(sig_x, sig_y)
b = qutip.control_2.cost_functions.averge_gate_fidelity(sig_x, sig_x)
c = qutip.control_2.cost_functions.averge_gate_fidelity(
    sig_x, 1 / np.sqrt(2) * (sig_y + sig_x))

a_e = qutip.control_2.cost_functions.entanglement_fidelity(sig_x, sig_y)[0]
b_e = qutip.control_2.cost_functions.entanglement_fidelity(sig_x, sig_x)[0]
c_e = qutip.control_2.cost_functions.entanglement_fidelity(
    sig_x, 1 / np.sqrt(2) * (sig_y + sig_x))[0]

e_sig_x = sig_x.exp(tau=.5j * math.pi)
e_sig_y = sig_y.exp(tau=.5j * math.pi)

c = qutip.control_2.cost_functions.averge_gate_fidelity(e_sig_x, e_sig_y)
d = qutip.control_2.cost_functions.averge_gate_fidelity(e_sig_x, e_sig_x)


# test fidelity function vs monte carlo
n_t = 200

h_drift = [
    qutip.control_2.matrix.ControlDense(np.zeros((2, 2), dtype=complex))
    for _ in range(200)]
h_control = np.asarray([[.5 * 2 * math.pi * qutip.control_2.matrix.ControlDense(
    qutip.sigmax()),
              .5 * 2 * math.pi * qutip.control_2.matrix.ControlDense(
                  qutip.sigmaz())]
             for _ in range(n_t)])
num_ctrl = 2
ctrl_amps = np.asarray([[.5] * 50 + [0] * 50 + [.25] * 50 + [.25] * 50,
                       [0] * 50 + [.5] * 50 + [0] * 50 + [0] * 50]).transfer_matrix
ctrl_amps /= 50
tau = [1] * n_t
initial_state = qutip.control_2.matrix.ControlDense(np.eye(2))


def dial_et_al_spectral_noise_density(f):
    if 50e3 <= f < 1e6:
        return 8e-16 * f ** -.7
    elif 1e6 <= f <= 3e9:
        return 8e-16 * 1e6 ** -.7
    else:
        return 0


def dial_vectorized(f):
    for i in range(f.size):
        f[i] = dial_et_al_spectral_noise_density(f[i])
    return f


def fictional_noise_density(f):
    return np.ones_like(f) * 1e-6


def Yoneda_one_over_f_spectral_noise_density(f):
    S = 3e-9 / f
    return S


noise_density_in_use = Yoneda_one_over_f_spectral_noise_density

fast_noise_parameters = qutip.control_2.noise.NTGColoredNoise(
    noise_spectral_density=noise_density_in_use,
    dt=1, n_samples_per_trace=n_t, n_traces=100
)

h_noise = np.asarray([.5 * 2 * math.pi * qutip.control_2.matrix.ControlDense(
    qutip.sigmax())])
filter_function_h_n = [[h_noise[0].data, np.ones(n_t)]]

tslot_obj = qutip.control_2.tslotcomp.TSCompSaveAllNoise(
    h_ctrl=h_control, h_drift=h_drift, num_ctrl=num_ctrl, tau=tau,
    initial_state=initial_state, ctrl_amps=ctrl_amps, num_t=n_t,
    h_noise=h_noise, noise_trace_generator=fast_noise_parameters,
    filter_function_h_n=filter_function_h_n)

# Fidelity for the fast noise by average fidelity
# here as well we can use the QSNoise fidelity

fidelity_computer_fast_noise_average = \
    qutip.control_2.cost_functions.OperationNoiseInfidelity(
        t_slot_comp=tslot_obj,
        target=initial_state,
        index=['fast_noise_average_fid'],
        use_unitary_derivatives=True,
        neglect_systematic_errors=True,
        fidelity_measure='average'
    )

fidelity_computer_fast_noise_entanglement = \
    qutip.control_2.cost_functions.OperationNoiseInfidelity(
        t_slot_comp=tslot_obj,
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
s_lin = noise_density_in_use(omega_lin)

# s_lin, omega_lin = ff.util.symmetrize_spectrum(s_lin, omega_lin)

omega_log = np.logspace(np.log10(omega_min), np.log10(omega_max / 2),
                        num=n_t // 2, base=10)
s_log = noise_density_in_use(omega_log)

s_log, omega_log = ff.util.symmetrize_spectrum(s_log, omega_log)

pulse_sequence = tslot_obj.create_pulse_sequence()
infidelity, smallness = ff.infidelity(pulse=pulse_sequence, S=s_log,
                                      omega=omega_log,
                                      return_smallness=True)

# omega = ff.util.get_sample_frequencies(pulse=pulse_sequence, n_samples=n_t,
#                                        spacing='lin', symmetric=True)
# s = constant_noise_density(omega)

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
b = fidelity_computer_ff_log.costs()
c = fidelity_computer_ff_lin.costs()
