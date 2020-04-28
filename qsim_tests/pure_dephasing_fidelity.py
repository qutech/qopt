import numpy as np
import unittest

from qsim.noise import NTGQuasiStatic
from qsim.matrix import DenseOperator
from qsim.solver_algorithms import SchroedingerSMonteCarlo
from qsim.cost_functions import OperationNoiseInfidelity


class PureDephasing(unittest.TestCase):
    def test_quasi_static_noise_deterministic_sampling(self):
        """ The same problem has also been positively tested with two time
        steps. """
        h_ctrl = [DenseOperator(np.diag([.5, -.5])), ]
        h_drift = [DenseOperator(np.zeros((2, 2)))]

        noise_levels = 1e-4 * np.arange(1, 101)
        actual_noise_levels = np.zeros((100,))
        average_infids = np.zeros((100,))

        for i, std_dev in enumerate(noise_levels):
            ntg = NTGQuasiStatic(standard_deviation=[std_dev, ],
                                 n_samples_per_trace=1, n_traces=200,
                                 sampling_mode='uncorrelated_deterministic')

            ctrl_amps = 2 * np.pi * np.ones((1, 1))
            t_slot_comp = SchroedingerSMonteCarlo(
                h_drift=h_drift,
                h_ctrl=h_ctrl,
                initial_state=DenseOperator(np.eye(2)),
                tau=[1],
                h_noise=h_ctrl,
                noise_trace_generator=ntg
            )
            t_slot_comp.set_ctrl_amps(ctrl_amps)

            quasi_static_infid = OperationNoiseInfidelity(
                solver=t_slot_comp,
                target=DenseOperator(np.eye(2)),
                neglect_systematic_errors=True,
                fidelity_measure='entanglement'
            )
            average_infids[i] = quasi_static_infid.costs() * (2 / 3)
            actual_noise_levels[i] = np.std(ntg.noise_samples)

        self.assertLess(
            np.sum(np.abs((np.ones_like(average_infids)
                           - average_infids / (noise_levels ** 2 / 6)))) / 100,
            0.05)
        self.assertLess(
            np.sum(np.abs((np.ones_like(average_infids)
                           - average_infids / (
                                       actual_noise_levels ** 2 / 6)))) / 100,
            1e-5)

    def test_quasi_static_noise_monte_carlo(self):
        np.random.seed(0)
        h_ctrl = [DenseOperator(np.diag([.5, -.5])), ]
        h_drift = [DenseOperator(np.zeros((2, 2)))]

        n_noise_values = 20
        noise_levels = 1e-4 * np.arange(1, n_noise_values + 1)
        actual_noise_levels = np.zeros((n_noise_values,))
        average_infids = np.zeros((n_noise_values,))

        for i, std_dev in enumerate(noise_levels):
            ntg = NTGQuasiStatic(standard_deviation=[std_dev, ],
                                 n_samples_per_trace=1, n_traces=2000,
                                 sampling_mode='monte_carlo')

            ctrl_amps = 2 * np.pi * np.ones((1, 1)) * 0
            t_slot_comp = SchroedingerSMonteCarlo(
                h_drift=h_drift,
                h_ctrl=h_ctrl,
                initial_state=DenseOperator(np.eye(2)),
                tau=[1],
                h_noise=h_ctrl,
                noise_trace_generator=ntg
            )
            t_slot_comp.set_ctrl_amps(ctrl_amps)

            quasi_static_infid = OperationNoiseInfidelity(
                solver=t_slot_comp,
                target=DenseOperator(np.eye(2)),
                neglect_systematic_errors=False,
                fidelity_measure='entanglement'
            )
            average_infids[i] = quasi_static_infid.costs() * (2 / 3)
            actual_noise_levels[i] = np.std(ntg.noise_samples)

        self.assertLess(
            np.sum(np.abs((np.ones_like(average_infids)
                           - average_infids / (noise_levels ** 2 / 6)))) / 100,
            0.05)
        self.assertLess(
            np.sum(np.abs((np.ones_like(average_infids)
                           - average_infids / (
                                       actual_noise_levels ** 2 / 6)))) / 100,
            0.05)


"""
import matplotlib.pyplot as plt
plt.figure()
plt.plot(noise_levels, average_infids)
plt.plot(noise_levels, noise_levels ** 2 / 6)
plt.plot(noise_levels, actual_noise_levels ** 2 / 6)
plt.legend(['simulation', 'analytic_calculation', 'analytic_actual_noise'])
p = np.polyfit(noise_levels, average_infids, deg=2)
"""
