from qopt import *
import numpy as np
import cProfile


n_time_steps = 200
time_step = 1  # ns
oversampling = 100
n_traces = 10

delta_omega_0 = 10e-3 * 2 * np.pi  # GHz = 10 * 2 pi Mhz


transfer_function = OversamplingTF(oversampling=oversampling, num_ctrls=2)
noise_trace_generator = NTGQuasiStatic(
    standard_deviation=[delta_omega_0, ] * 2,
    n_samples_per_trace=n_time_steps * oversampling,
    n_traces=n_traces,
)

solver = SchroedingerSMCControlNoise(
    transfer_function=transfer_function,
    h_ctrl=[DenseOperator.pauli_x(), DenseOperator.pauli_y()],
    h_drift=[DenseOperator.pauli_z() * 0],
    tau=time_step * np.ones(n_time_steps),
    noise_trace_generator=noise_trace_generator,
    calculate_propagator_derivatives=False
)

solver.set_optimization_parameters(np.random.rand(n_time_steps, 2))

cProfile.run('solver._compute_dyn_gen_noise()')  # 37 sec

cProfile.run('solver._compute_propagation('
             'calculate_propagator_derivatives=False)')  # 158

cProfile.run('solver._compute_propagation('
             'calculate_propagator_derivatives=False,'
             'processes=4)')  # 16

cProfile.run('solver._compute_propagation('
             'calculate_propagator_derivatives=False,'
             'processes=None)')  # 50

cProfile.run('solver._compute_propagation('
             'calculate_propagator_derivatives=True)')  # 19.085

cProfile.run('solver._compute_forward_propagation()')  # 1.3

cProfile.run('solver._compute_reversed_propagation()')  # .8

cProfile.run('solver._compute_propagation_derivatives()')  # 20

# Its only worth to parallelize the calculation of the propagators.


##################################

from qopt.examples.rabi_driving.rabi_xy_setup import *

qs_solver = solver_qs_noise_xy
fast_mc_solver = solver_colored_noise_xy
syst_infid = entanglement_infid_xy
qs_infid = entanglement_infid_qs_noise_xy
fast_infid = entanglement_infid_colored_noise_xy

def simulate_propagation(initial_pulse):
    simulator = Simulator(
        solvers=[qs_solver, fast_mc_solver],
        cost_fktns=[syst_infid, qs_infid, fast_infid]
    )
    total_propagator
