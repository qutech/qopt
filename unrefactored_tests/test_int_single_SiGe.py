import numpy as np
import filter_functions as ff

import math
import os
import cProfile

import qutip.control_2.cost_functions
import qutip.control_2.dynamics
import qutip.control_2.transfer_function
import qutip.control_2.tslotcomp
import qutip.control_2.matrix
import qutip.control_2.optimize
import qutip.control_2.stats
import qutip.control_2.data_container
import qutip.control_2.analyser
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
    qutip.sigmax())]

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

np.random.seed(0)
initial_pulse = 4 * np.random.rand(num_x, num_ctrl) - 2

vx = [1.99685657, 1.12829824, 1.74366693, -1.60690613, -1.99999995,
      -1.99999999, -1.99999999, -1.99999996, -1.16819191, 1.99999288,
      1.99996868, 1.11775814]

vy = [1.99999999, 1.99999989, -0.36258558, -1.96495835, -1.99545292,
      -1.99999794, -1.99999998, -1.99999994, -1.99999998, 0.15630521,
      2., 2.]

good_pulse = np.asarray([vx, vy]).transfer_matrix

# initial_pulse = good_pulse

initial_ctrl_amps = concatenated_tf(initial_pulse)

initial_state = qutip.control_2.matrix.ControlDense(np.eye(2, dtype=complex))

qs_noise_parameters = qutip.control_2.noise.NTGQuasiStatic(
    standard_deviation=[100e3], n_traces=20, n_samples_per_trace=106
)

def dial_et_al_spectral_noise_density(f):
    if 50e3 <= f < 1e6:
        return 8e-16 * f ** -.7
    elif 1e6 <= f <= 3e9:
        return 8e-16 * 1e6 ** -.7
    else:
        return 0


def Yoneda_one_over_f_spectral_noise_density(f):
    S = 3e6 / f
    return S

def fictional_noise_density(f):
    return np.ones_like(f) * 1e-1

noise_density_in_use = Yoneda_one_over_f_spectral_noise_density

# omega_min = 50e3
# omega_max = 1e9
dt = 100e-9 / 8
t_total = dt * 106
omega_min = 1 / t_total
omega_max = 1 / dt
# omega_min = 1e-2
# omega_max = 3e6
n_samples = 106
# omega = np.logspace(start=np.log10(omega_min), stop=np.log10(omega_max), num=n_samples)
omega = np.linspace(omega_min, omega_max / 2, n_samples // 2)


# s = np.array([dial_et_al_spectral_noise_density(w) for w in omega])
# s = np.array(yoneda_one_over_f_spectral_noise_density(omega))
s = noise_density_in_use(omega)

fast_noise_parameters = qutip.control_2.noise.NTGColoredNoise(
    noise_spectral_density=noise_density_in_use,
    dt=dt, n_samples_per_trace=106, n_traces=200
)

tau_u = [100e-9 / over_sample_rate for _ in range(num_u)]
filter_function_h_n = [[h_noise_qs[0].data, np.ones(num_u)]]
"""
t_slot_comp = qutip.control_2.t_slot_comp.TSCompSaveAll(
    h_drift=h_drift,
    h_ctrl=h_ctrl,
    initial_state=initial_state,
    tau=tau_u, num_t=num_u,
    num_ctrl=num_ctrl,
    ctrl_amps=initial_ctrl_amps,
    calculate_unitary_derivatives=False,
    filter_function_h_n=filter_function_h_n)


t_slot_comp = qutip.control_2.t_slot_comp.TSCompSaveAllNoise(
    h_drift=h_drift,
    h_ctrl=h_ctrl,
    initial_state=initial_state,
    tau=tau_u, num_t=num_u,
    num_ctrl=num_ctrl,
    ctrl_amps=initial_ctrl_amps,
    calculate_unitary_derivatives=True,
    h_noise=h_noise,
    noise_parameters=qs_noise_parameters)
"""

t_slot_comp = qutip.control_2.tslotcomp.TSCompSaveAllNoise(
    h_drift=h_drift,
    h_ctrl=h_ctrl,
    initial_state=initial_state,
    tau=tau_u, num_t=num_u,
    num_ctrl=num_ctrl,
    ctrl_amps=initial_ctrl_amps,
    calculate_unitary_derivatives=True,
    h_noise=h_noise_qs,
    noise_trace_generator=fast_noise_parameters,
    filter_function_h_n=filter_function_h_n)

t_slot_comp_spectral = qutip.control_2.tslotcomp.TSCompSaveAllNoise(
    h_drift=h_drift,
    h_ctrl=h_ctrl,
    initial_state=initial_state,
    tau=tau_u, num_t=num_u,
    num_ctrl=num_ctrl,
    ctrl_amps=initial_ctrl_amps,
    calculate_unitary_derivatives=True,
    h_noise=h_noise_qs,
    noise_trace_generator=fast_noise_parameters,
    filter_function_h_n=filter_function_h_n,
    exponential_method='spectral')

# test backward propagation
# t_slot_comp.compute_gen()
# b = t_slot_comp.reversed()
# a = t_slot_comp.reversed_cumulative()

# fid comp

# target = qutip.control_2.matrix.ControlDense(np.eye(2, dtype=complex))
target = qutip.control_2.matrix.ControlDense(
    (np.eye(2, dtype=complex) + 1j * qutip.sigmax()) / np.sqrt(2))

# Fidelity by matrix distance
fidelity_computer = qutip.control_2.cost_functions.OperatorMatrixNorm(
    t_slot_comp=t_slot_comp,
    target=target,
    mode='vector')

# Fidelity by average gate fidelity
fid_comp_average_gate = qutip.control_2.cost_functions.OperationInfidelity(
    t_slot_comp=t_slot_comp,
    target=target,
    use_unitary_derivatives=True
)


# Fidelity for the quasi static noise by average fidelity
fidelity_computer_qs_noise = \
    qutip.control_2.cost_functions.OperationNoiseInfidelity(
        t_slot_comp=t_slot_comp,
        target=target,
        index=['qs_average_fid'],
        use_unitary_derivatives=True
    )

# Fidelity for the fast noise by average fidelity
# here as well we can use the QSNoise fidelity computer
fidelity_computer_fast_noise_average = \
    qutip.control_2.cost_functions.OperationNoiseInfidelity(
        t_slot_comp=t_slot_comp,
        target=target,
        index=['fast_noise_average_fid'],
        use_unitary_derivatives=True,
        neglect_systematic_errors=True,
        fidelity_measure='average'
    )

fidelity_computer_fast_noise_entang = \
    qutip.control_2.cost_functions.OperationNoiseInfidelity(
        t_slot_comp=t_slot_comp,
        target=target,
        index=['fast_noise_entanglement_fid'],
        use_unitary_derivatives=True,
        neglect_systematic_errors=True,
        fidelity_measure='entanglement'
    )

fidelity_computer_fast_noise_entang_spectral = \
    qutip.control_2.cost_functions.OperationNoiseInfidelity(
        t_slot_comp=t_slot_comp_spectral,
        target=target,
        index=['fast_noise_entanglement_fid_spectral'],
        use_unitary_derivatives=True,
        neglect_systematic_errors=True,
        fidelity_measure='entanglement'
    )


# Fidelity for the quasi static noise by matrix distance
fidelity_computer_qs_noise_mat_dist = \
    qutip.control_2.cost_functions.FidCompOperatorQSNoise(
        tslotcomp=t_slot_comp,
        target=target,
        index=['qs_delta_eps']
    )

# Fidelity by filter function


# s, omega = ff.util.symmetrize_spectrum(s, omega)

fidelity_computer_ff = \
    qutip.control_2.cost_functions.OperatorFilterFunctionInfidelity(
        t_slot_comp=t_slot_comp,
        omega=omega,
        noise_power_spec_density=s,
        target=target
    )

# stats

stats = qutip.control_2.stats.Stats()

# dynamics
dynamic = qutip.control_2.dynamics.Dynamics(
    pulse=initial_pulse, num_ctrl=2,
    transfer_function=concatenated_tf,
    tslot_comps=t_slot_comp,
    cost_fktns=(
        fid_comp_average_gate, fidelity_computer_fast_noise_entang_spectral,
        fidelity_computer_fast_noise_entang, fidelity_computer_ff),
    stats=stats)

dynamic.wrapped_cost_functions()
# dynamic.wrapped_jac_function()






"""
# optimizer

optimizer = qutip.control_2.optimize.OptimizerLeastSquaresOld(
    dynamic.wrapped_cost_functions, x0=initial_pulse, stats=stats,
    grad=dynamic.wrapped_jac_function)

result = qutip.control.optimresult.OptimResult()
optimizer.approx_grad = False
optimizer.termination_conditions['max_iterations'] = 20
optimizer.termination_conditions['max_fid_func_calls'] = 20
optimizer.termination_conditions['min_gradient_norm'] = 1e-10
optimizer.termination_conditions['max_wall_time'] = 30
optimizer.termination_conditions["min_fid_gain"] = 1e-10
optimizer.add_bounds((-2, 2))
cProfile.run('optimizer.run_optimization(result)')
#optimizer.run_optimization(result)
# unoptimized: 39.127
# isinstance to type(): 35.639
# inplace calculations in the derivative of the average gate fidelity: 29.142
# use exp (scipy.linalg.expm) instead of dexp (scipy.linalg.dexp) 31 sec
# the time required for the 10 executions is not constant but around 30 sec
# using the frechet derivative from dexp 30 sec

concatenated_tf.set_times(
    np.arange(num_x + 1) * tau[0], 2)
concatenated_tf.plot_pulse(result.final_x)

# test plot
# t_slot_comp.plot_bloch_sphere()

# test data_container

directory = r"Z:\SimulationData\Qutip\Tests\integration_test"
file_name = r"single_SiGe"

reload(qutip.control_2.data_container)
data_container = qutip.control_2.data_container.DataContainer(
    storage_path=directory, file_name=file_name, append_time_to_path=False)
data_container.append_optim_result(optim_result=result, stats=stats)
data_container.to_pickle()

full_path = os.path.join(directory, file_name)
data_loaded = qutip.control_2.data_container.DataContainer.from_pickle(
    full_path)
# test analyser
analyser = qutip.control_2.analyser.Analyser(data_loaded)
analyser.plot_costs()


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

# a = numeric_grad(data_container.final_voltages[0], dynamic.wrapped_cost_functions, k=0)
# b = dynamic.wrapped_jac_function(data_container.final_voltages[0])

# compare with old results:


vx = [1.99685657, 1.12829824, 1.74366693, -1.60690613, -1.99999995,
      -1.99999999, -1.99999999, -1.99999996, -1.16819191, 1.99999288,
      1.99996868, 1.11775814]

vy = [1.99999999, 1.99999989, -0.36258558, -1.96495835, -1.99545292,
      -1.99999794, -1.99999998, -1.99999994, -1.99999998, 0.15630521,
      2., 2.]

good_pulse = np.asarray([vx, vy]).T

# dynamic.wrapped_cost_functions(good_pulse)
# t_slot_comp.set_ctrl_amps(concatenated_tf(good_pulse))
# t_slot_comp.plot_bloch_sphere()



This test indicated
# performance test indicated that the initialization of a dense control matrix 
does not have a considerable overhead over a numpy array.

def control_matrix_init(k=int(1e6)):
    b = np.eye(2)
    for i in range(k):
        a = qutip.control_2.matrix.ControlDense(b)

def numpy_matrix_init(k=int(1e6)):
    for i in range(k):
        a = np.eye(2)

cProfile.run('control_matrix_init()')
# 4.18
cProfile.run('numpy_matrix_init()')
# 3.86



# t_slot_comp._dU[0, 0].data / t_slot_comp.tau[0] / -1j

"""