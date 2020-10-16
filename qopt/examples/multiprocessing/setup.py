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
from multiprocessing import Pool
import time


n_pulses = 2
random_pulses = np.random.rand(n_pulses, n_time_samples, len(qs_solver.h_ctrl))

start_parallel = time.time()
with Pool(processes=None) as pool:
    infids = pool.map(simulate_propagation, random_pulses)
end_parallel = time.time()
parallel_time = end_parallel - start_parallel  # 85.55985116958618

start_sequential = time.time()
infids_sequential = list(map(simulate_propagation, random_pulses))
end_sequential = time.time()
sequential_time = end_sequential - start_sequential  # 274.9153392314911

############################

from qopt.parallel import run_optimization_parallel


data = run_optimization_parallel(optimizer, initial_pulses=random_pulses)

optimizer.system_simulator.solvers[0].plot_bloch_sphere(
    data.final_parameters[0])

analyser = Analyser(data)
analyser.plot_costs(0)

#

from qopt.examples.rabi_driving.rabi_xy_setup import *

n_pulses = 2
random_pulses = np.random.rand(n_pulses, n_time_samples, len(qs_solver.h_ctrl))

from qopt.parallel import run_optimization_parallel


data = run_optimization_parallel(optimizer, initial_pulses=random_pulses)

analyser = Analyser(data)
analyser.plot_costs(0)
