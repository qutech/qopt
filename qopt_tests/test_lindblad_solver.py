import numpy as np

from qopt.matrix import DenseOperator
from qopt.solver_algorithms import LindbladSolver
from qopt.cost_functions import OperationInfidelity
from qopt.simulator import Simulator

sigma_minus = DenseOperator(np.asarray([[0, 0], [1, 0]]))
zero_matrix = DenseOperator(np.asarray([[0, 0], [0, 0]]))
gamma = 3


def prefactor_function(control_amplitudes):
    return gamma * np.expand_dims(
        np.sum(np.abs(control_amplitudes), axis=1), axis=1)


def prefactor_function_derivative(control_amplitudes):
    return np.expand_dims(gamma * np.sign(control_amplitudes), axis=2)


solver = LindbladSolver(
    h_drift=[zero_matrix, ],
    h_ctrl=[zero_matrix, zero_matrix],
    initial_state=DenseOperator(np.eye(4)),
    tau=[1, ],
    lindblad_operators=[sigma_minus, ],
    prefactor_function=prefactor_function,
    prefactor_derivative_function=prefactor_function_derivative
)

cost_func = OperationInfidelity(
    solver=solver,
    target=DenseOperator(np.eye(2)),
    super_operator_formalism=True
)

simulator = Simulator(
    solvers=[solver, ],
    cost_fktns=[cost_func, ]
)
pulse = 5 * np.ones((1, 2))

simulator.compare_numeric_to_analytic_gradient(pulse)

simulator.numeric_gradient(pulse)
simulator.wrapped_jac_function(pulse)
