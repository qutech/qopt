# Covers testing the scipy.minimize wrapper vs the scipy.least_squares_wrapper

from qopt import *
from qopt.examples.rabi_driving import setup

h_ctrl = [DenseOperator.pauli_x(), DenseOperator.pauli_y()]

simulator = Simulator(
    solvers=[setup.solver_qs_noise_xy, ],
    cost_fktns=[setup.entanglement_infid_xy,
                setup.entanglement_infid_qs_noise_xy]
)

optimizer = ScalarMinimizingOptimizer(
    system_simulator=simulator,
    cost_fktn_weights=[1, 1e2]
)

optimizer_no_jac = ScalarMinimizingOptimizer(
    system_simulator=simulator,
    cost_fktn_weights=[1, 1e2],
    use_jacobian_function=False
)

init_pulse = setup.random_xy_init_pulse(seed=1)
result = optimizer.run_optimization(init_pulse)

data_container = DataContainer()
data_container.append_optim_result(result)
analyzer = Analyser(data_container)
analyzer.plot_costs()

result_no_jac = optimizer_no_jac.run_optimization(init_pulse)
data_container2 = DataContainer()
data_container2.append_optim_result(result_no_jac)
analyzer = Analyser(data_container)
analyzer.plot_costs()
