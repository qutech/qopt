import numpy as np

import math

import qutip.control_2.cost_functions
import qutip.control_2.dynamics
import qutip.control_2.transfer_function
import qutip.control_2.tslotcomp
import qutip.control_2.matrix
import qutip.control_2.stats
import qutip.control_2.noise

import qutip.control.optimresult

from importlib import reload

reload(qutip.control_2.dynamics)
reload(qutip.control_2.cost_functions)
reload(qutip.control_2.tslotcomp)
reload(qutip.control_2.transfer_function)

# constants

num_x = 12
num_ctrl = 2

over_sample_rate = 8
bound_type = ("n", 5)
num_u = over_sample_rate * num_x + 2 * bound_type[1]

tau = [100e-9 for _ in range(num_x)]
lin_freq_rel = 5.614e-4 * 1e6 * 1e3

h_ctrl = [
    [.5 * 2 * math.pi * qutip.control_2.matrix.ControlDense(qutip.sigmax()),
     .5 * 2 * math.pi * qutip.control_2.matrix.ControlDense(qutip.sigmay())]
    for _ in range(num_u)]
h_ctrl = np.asarray(h_ctrl)
h_drift = [qutip.control_2.matrix.ControlDense(np.zeros((2, 2), dtype=complex))
           for _ in range(num_u)]
h_noise_qs = [.5 * 2 * math.pi * qutip.control_2.matrix.ControlDense(
    qutip.sigmaz())]

# trivial transfer function

T = np.diag(num_x * [lin_freq_rel])
linear_transfer_function = \
    qutip.control_2.transfer_function.CustomTF(T)
gaussian_transfer_function = qutip.control_2.transfer_function.Gaussian(
    omega=2 * 8 / tau[0], over_sample_rate=over_sample_rate, start=0., end=0.,
    bound_type=bound_type,
    num_ctrls=2, times=tau[0] * np.arange(num_x + 1)
)
concatenated_tf = qutip.control_2.transfer_function.ConcatenateTF(
    tf1=linear_transfer_function, tf2=gaussian_transfer_function
)

# t_slot_comp

seed = 1

np.random.seed(seed)


initial_pulse = 4 * np.random.rand(num_x, num_ctrl) - 2
initial_ctrl_amps = concatenated_tf(initial_pulse)

initial_state = qutip.control_2.matrix.ControlDense(np.eye(2, dtype=complex))

qs_noise_parameters = qutip.control_2.noise.NTGQuasiStatic(
    standard_deviation=[100e3], n_samples_per_trace=10, n_traces=1
)

tau_u = [100e-9 / over_sample_rate for _ in range(num_u)]

t_slot_comp = qutip.control_2.tslotcomp.TSCompSaveAll(
    h_drift=h_drift,
    h_ctrl=h_ctrl,
    initial_state=initial_state,
    tau=tau_u, num_t=num_u,
    num_ctrl=num_ctrl,
    ctrl_amps=initial_ctrl_amps)


target = qutip.control_2.matrix.ControlDense(
    (np.eye(2, dtype=complex) + 1j * qutip.sigmax()) / np.sqrt(2))

fidelity_computer = qutip.control_2.cost_functions.OperatorMatrixNorm(
    t_slot_comp=t_slot_comp,
    target=target,
    mode='vector')
fidelity_computer.costs()

fidelity_computer_qs_noise = \
    qutip.control_2.cost_functions.OperationNoiseInfidelity(
        t_slot_comp=t_slot_comp,
        target=target,
        index=['qs_delta_eps']
    )

# stats

stats = qutip.control_2.stats.Stats()

# dynamics
dynamic = qutip.control_2.dynamics.Dynamics(
    pulse=initial_pulse, num_ctrl=2,
    transfer_function=concatenated_tf,
    tslot_comps=[t_slot_comp, ],
    cost_fktns=(fidelity_computer, fidelity_computer_qs_noise),
    stats=stats)

initial_cost = dynamic.wrapped_cost_functions(initial_pulse)
cost = np.zeros(shape=(3, num_ctrl * num_x))

for k in range(9):
    for i in range(num_x):
        for j in range(num_ctrl):
            cost[1, i + j * num_x] = np.real(initial_cost[k])
            delta = np.zeros(shape=initial_pulse.shape)
            delta[i, j] = -1e-2
            cost[0, i + j * num_x] = dynamic.wrapped_cost_functions(
                initial_pulse + delta)[k]
            delta[i, j] = 1e-2
            cost[2, i + j * num_x] = dynamic.wrapped_cost_functions(
                initial_pulse + delta)[k]

    numeric_gradient = np.gradient(cost, 1e-2, axis=0)

    analytic_gradient = dynamic.wrapped_jac_function()

    print(numeric_gradient[1, :])
    print("\n")
    print(analytic_gradient[k, :])
    print(t_slot_comp._fwd[-1][0, 0])
    print(np.angle(t_slot_comp._fwd[-1][0, 0]))
    print("\n")


# might be useful
def numeric_grad(pulse, cost_func, k=0, eps=1e-3):
    cost = np.zeros(shape=(3, num_ctrl * num_x))
    for i in range(num_x):
        for j in range(num_ctrl):
            cost[1, i + j * num_x] = cost_func(pulse)[k]
            delta = np.zeros(shape=pulse.shape)
            delta[i, j] = -eps
            cost[0, i + j * num_x] = cost_func(pulse + delta)[k]
            delta[i, j] = eps
            cost[2, i + j * num_x] = cost_func(pulse + delta)[k]

    return np.gradient(cost, eps, axis=0)[1]
