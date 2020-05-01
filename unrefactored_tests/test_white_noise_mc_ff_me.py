"""
We will compare infidelities due to Markovian noise calculated my Monte Carlo
simulations, filter functions and a Lindblad master equation.
"""

import math
import copy
import qutip.control_2.matrix
import qutip.control_2.cost_functions
import numpy as np
import pandas as pd
# import filter_functions as ff
# import matplotlib.pyplot as plt
import qutip.control_2.tslotcomp
import qutip.control_2.noise
import qutip.control_2.util

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
target = np.asarray([[0.-1.j, 0.+0.j], [0.+0.j, 0.+1.j]])
spectral_density = 3e-6


def white_noise_density(f):
    return np.ones_like(f) * spectral_density


h_noise = np.asarray(
    [.5 * 2 * math.pi * qutip.control_2.matrix.ControlDense(
        qutip.sigmax())])
filter_function_h_n = [[h_noise[0].data, np.ones(n_t)]]

fast_noise_parameters = qutip.control_2.noise.NTGColoredNoise(
    noise_spectral_density=white_noise_density,
    dt=1, n_samples_per_trace=n_t, n_traces=1
)

tslot_obj = qutip.control_2.tslotcomp.TSCompSaveAllNoise(
    h_ctrl=h_control, h_drift=h_drift, num_ctrl=num_ctrl, tau=tau,
    initial_state=initial_state, ctrl_amps=ctrl_amps, num_t=n_t,
    h_noise=np.asarray([h_noise, ] * n_t).transfer_matrix,
    noise_trace_generator=fast_noise_parameters,
    filter_function_h_n=filter_function_h_n)
h_noise_np = h_noise[0].data

h_noise_sup_op_np = np.kron(np.conj(h_noise_np), h_noise_np) \
                - .5 * np.kron(np.eye(2), h_noise_np @ np.conj(h_noise_np.transfer_matrix)) \
                - .5 * np.kron(np.conj(h_noise_np.transfer_matrix) @ h_noise_np, np.eye(2))

initial_diss_sup_op = np.expand_dims(h_noise_sup_op_np, axis=0) \
                      * spectral_density
initial_diss_sup_op = np.expand_dims(initial_diss_sup_op, axis=0)
initial_diss_sup_op = np.repeat(initial_diss_sup_op, n_t, axis=0)

lindblad = np.expand_dims(h_noise_np, axis=0)
lindblad = np.expand_dims(lindblad, axis=0)
lindblad = np.repeat(lindblad, n_t, axis=0)


def prefactor_function(ctrl_amps):
    num_t = ctrl_amps.shape[0]
    return spectral_density * np.ones(shape=(num_t, 1))


cpy = np.squeeze(copy.deepcopy(initial_diss_sup_op))


def diss_sup_op_function(ctrl_amps):
    return cpy


initial_state_me = np.eye(4)
tslot_obj_me = qutip.control_2.tslotcomp.TSCompLindblad(
    h_ctrl=h_control, h_drift=h_drift, num_ctrl=num_ctrl, tau=tau,
    initial_state=initial_state_me, ctrl_amps=ctrl_amps, num_t=n_t,
    filter_function_h_n=filter_function_h_n,
    initial_diss_super_op=initial_diss_sup_op)

tslot_obj_me_lindblad = qutip.control_2.tslotcomp.TSCompLindblad(
    h_ctrl=h_control, h_drift=h_drift, num_ctrl=num_ctrl, tau=tau,
    initial_state=initial_state_me, ctrl_amps=ctrl_amps, num_t=n_t,
    filter_function_h_n=filter_function_h_n,
    lindblad_operators=lindblad,
    prefactor_function=prefactor_function)

tslot_obj_me_fkt_handle = qutip.control_2.tslotcomp.TSCompLindblad(
    h_ctrl=h_control, h_drift=h_drift, num_ctrl=num_ctrl, tau=tau,
    initial_state=initial_state_me, ctrl_amps=ctrl_amps, num_t=n_t,
    filter_function_h_n=filter_function_h_n,
    super_operator_function=diss_sup_op_function)

fidelity_computer_fast_noise_entanglement = \
    qutip.control_2.cost_functions.OperationNoiseInfidelity(
        t_slot_comp=tslot_obj,
        target=target,
        index=['fast_noise_entanglement_fid'],
        use_unitary_derivatives=False,
        neglect_systematic_errors=True,
        fidelity_measure='entanglement'
    )

fidelity_computer_fast_noise_entanglement_me = \
    qutip.control_2.cost_functions.OperationInfidelity(
        t_slot_comp=tslot_obj_me,
        target=target,
        use_unitary_derivatives=False,
        fidelity_measure='entanglement',
        super_operator_formalism=True
    )

fidelity_computer_fast_noise_entanglement_me_lindblad = \
    qutip.control_2.cost_functions.OperationInfidelity(
        t_slot_comp=tslot_obj_me_lindblad,
        target=target,
        use_unitary_derivatives=False,
        fidelity_measure='entanglement',
        super_operator_formalism=True
    )

fidelity_computer_fast_noise_entanglement_me_sup_op_function = \
    qutip.control_2.cost_functions.OperationInfidelity(
        t_slot_comp=tslot_obj_me_fkt_handle,
        target=target,
        use_unitary_derivatives=False,
        fidelity_measure='entanglement',
        super_operator_formalism=True
    )

# Fidelity by filter function
omega_min = 1 / n_t
omega_max = 1 / 1

omega_lin = np.linspace(omega_min, omega_max / 2, num=n_t // 2)
s_lin = white_noise_density(copy.deepcopy(omega_lin))


fidelity_computer_ff_lin = \
    qutip.control_2.cost_functions.OperatorFilterFunctionInfidelity(
        t_slot_comp=tslot_obj,
        omega=omega_lin,
        noise_power_spec_density=s_lin,
        target=target
    )

a = fidelity_computer_fast_noise_entanglement.costs()
b = fidelity_computer_ff_lin.costs()
c = fidelity_computer_fast_noise_entanglement_me.costs()
d = fidelity_computer_fast_noise_entanglement_me_lindblad.costs()
e = fidelity_computer_fast_noise_entanglement_me_sup_op_function.costs()


fid_names = ["mc_entanglement", "ff_lin", "me_entanglement",
             "me_lindblad_entanglement", "me_super_op_func"]

s = pd.Series(data=[a, b, c, d, e], index=fid_names)
print(s)
