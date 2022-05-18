"""
Test the derivative of the state fidelity.
"""
import numpy as np
import unittest

from qopt.matrix import DenseOperator
from qopt.solver_algorithms import SchroedingerSolver
from qopt.cost_functions import StateInfidelity
from qopt.simulator import Simulator

sigma_x = DenseOperator(np.asarray([[0, 1], [1, 0]]))
sigma_y = DenseOperator(np.asarray([[0, -1j], [1j, 0]]))
sigma_z = DenseOperator(np.asarray([[1, 0], [0, -1]]))

n_time_steps = 5
delta_t = .5 * np.pi

up = DenseOperator(np.asarray([[1], [0]]))
down = DenseOperator(np.asarray([[0], [1]]))

schroedinger_solver = SchroedingerSolver(
    h_drift=[0 * sigma_x] * n_time_steps,
    h_ctrl=[sigma_x, sigma_y],
    tau=delta_t * np.ones(n_time_steps),
    initial_state=up
)


class TestFidelitySchroedingerEq(unittest.TestCase):

    def test_state_fid(self):
        cost_fkt = StateInfidelity(
            schroedinger_solver,
            target=down
        )

        simulator = Simulator(
            solvers=[schroedinger_solver, ],
            cost_funcs=[cost_fkt, ]
        )

        np.random.seed(0)
        random_pulse = np.random.randn(5, 2)

        diff_norm, diff_rel = simulator.compare_numeric_to_analytic_gradient(
            pulse=random_pulse,
            delta_eps=1e-6
        )
        self.assertLess(diff_norm, 1e-5)
        self.assertLess(diff_rel, 1e-5)
