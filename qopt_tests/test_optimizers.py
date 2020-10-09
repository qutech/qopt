# Covers testing the scipy.minimize wrapper vs the scipy.least_squares_wrapper

from qopt import *
from qopt.examples.rabi_driving import setup


simulator = Simulator(
    solvers=[setup.solver_qs_noise_xy, ],
    cost_fktns=[setup.entanglement_infid_xy,
                setup.entanglement_infid_qs_noise_xy]
)

optimizer = ScalarMinimizingOptimizer(
    system_simulator=simulator,
    cost_fktn_weights=[1, 1e2],
    bounds=setup.bounds_xy
)

optimizer_no_jac = ScalarMinimizingOptimizer(
    system_simulator=simulator,
    cost_fktn_weights=[1, 1e2],
    use_jacobian_function=False,
    bounds=setup.bounds_xy
)

optimizer_least_squares = LeastSquaresOptimizer(
    system_simulator=simulator,
    cost_fktn_weights=[1, 1e2],
    bounds=setup.bounds_xy_least_sq
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
analyzer2 = Analyser(data_container2)
analyzer2.plot_costs()

result_least_squres = optimizer_least_squares.run_optimization(init_pulse)
data_container3 = DataContainer()
data_container3.append_optim_result(result_least_squres)
analyzer3 = Analyser(data_container3)
analyzer3.plot_costs()
