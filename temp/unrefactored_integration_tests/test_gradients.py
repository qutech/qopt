"""
This test uses the rabi driving as test case. So far it covers:

- Gradient of the entanglement fidelity
- Gradient of the entanglement fidelity in the presence of quasi static noise
"""

import qopt.examples.rabi_driving.setup as rabi
from qopt.solver_algorithms import SchroedingerSMonteCarlo
from qopt.matrix import DenseOperator
from qopt.cost_functions import OperationNoiseInfidelity, \
    OperationInfidelity
from qopt.simulator import Simulator
from qopt.noise import NTGQuasiStatic
import numpy as np
import unittest


class RabiTestCase(unittest.TestCase):
    def test_relative_gradients_xy(self):
        amp_bound = rabi.rabi_frequency_max / rabi.lin_freq_rel
        np.random.seed(0)
        initial_pulse = amp_bound * (
                2 * np.random.rand(rabi.n_time_samples, 2) - 1)

        ntg_quasi_static = NTGQuasiStatic(
            standard_deviation=[rabi.sigma_rabi, ],
            n_samples_per_trace=rabi.n_time_samples * rabi.oversampling,
            n_traces=10,
            always_redraw_samples=False,
            sampling_mode='uncorrelated_deterministic')

        tslot = SchroedingerSMonteCarlo(
            h_drift=[0 * rabi.h_drift, ],
            h_ctrl=rabi.h_ctrl,
            h_noise=[rabi.h_drift, ],
            noise_trace_generator=ntg_quasi_static,
            initial_state=DenseOperator(np.eye(2)),
            tau=[rabi.time_step, ] * rabi.n_time_samples,
            is_skew_hermitian=True,
            exponential_method='Frechet',
            transfer_function=rabi.exponential_transfer_function,
            amplitude_function=rabi.lin_amp_func
        )

        entanglement_infid = OperationInfidelity(
            solver=tslot,
            target=rabi.x_half,
            fidelity_measure='entanglement',
            label=['Entanglement Fidelity QS-Noise XY-Control']
        )

        tslot_noise = SchroedingerSMonteCarlo(
            h_drift=[0 * rabi.h_drift, ],
            h_ctrl=rabi.h_ctrl,
            h_noise=[rabi.h_drift, ],
            noise_trace_generator=ntg_quasi_static,
            initial_state=DenseOperator(np.eye(2)),
            tau=[rabi.time_step, ] * rabi.n_time_samples,
            is_skew_hermitian=True,
            exponential_method='Frechet',
            transfer_function=rabi.exponential_transfer_function,
            amplitude_function=rabi.lin_amp_func
        )

        entanglement_infid_qs_noise_xy = OperationNoiseInfidelity(
            solver=tslot_noise,
            target=rabi.x_half,
            fidelity_measure='entanglement',
            label=['Entanglement Fidelity QS-Noise XY-Control'],
            neglect_systematic_errors=True
        )

        dynamics = Simulator(
            solvers=[tslot, ],
            cost_funcs=[entanglement_infid, ]
        )

        dynamics_noise = Simulator(
            solvers=[tslot_noise, ],
            cost_funcs=[entanglement_infid_qs_noise_xy]
        )

        _, rel_grad_deviation_unperturbed = \
            dynamics.compare_numeric_to_analytic_gradient(initial_pulse)
        self.assertLess(rel_grad_deviation_unperturbed, 1e-6)

        _, rel_grad_deviation_qs_noise = \
            dynamics_noise.compare_numeric_to_analytic_gradient(initial_pulse)
        self.assertLess(rel_grad_deviation_qs_noise, 1e-4)
        # This gradient calculation is numerically unstable and anti correlates
        # with the number of traces.
        # 10 traces -> 20%
        # 100 traces -> 9%
        # if we use the scipy dexpm
        # 10 traces -> 5.4e-5

    def test_phase_control_gradient(self):
        amp_bound = rabi.rabi_frequency_max / rabi.lin_freq_rel
        phase_bound_upper = 50 / 180 * np.pi
        phase_bound_lower = -50 / 180 * np.pi

        def random_phase_control_pulse(n):
            amp = amp_bound * (2 * np.random.rand(n) - 1)
            phase = (phase_bound_upper - phase_bound_lower) \
                * np.random.rand(n) \
                - (phase_bound_upper - phase_bound_lower) / 2
            return np.concatenate(
                (np.expand_dims(amp, 1), np.expand_dims(phase, 1)), axis=1)

        dynamics_phase_control = Simulator(
            solvers=[rabi.solver_qs_noise_phase_control],
            cost_funcs=[rabi.entanglement_infid_phase_control]
        )

        ntg_quasi_static = NTGQuasiStatic(
            standard_deviation=[rabi.sigma_rabi, ],
            n_samples_per_trace=rabi.n_time_samples * rabi.oversampling,
            n_traces=10,
            always_redraw_samples=False,
            sampling_mode='uncorrelated_deterministic')

        time_slot_comp_qs_noise_phase_control = SchroedingerSMonteCarlo(
            h_drift=[0 * rabi.h_drift, ],
            h_ctrl=rabi.h_ctrl,
            h_noise=[rabi.h_drift, ],
            noise_trace_generator=ntg_quasi_static,
            initial_state=DenseOperator(np.eye(2)),
            tau=[rabi.time_step, ] * rabi.n_time_samples,
            is_skew_hermitian=True,
            exponential_method='Frechet',
            transfer_function=rabi.identity_transfer_function,
            amplitude_function=rabi.phase_ctrl_amp_func
        )

        entanglement_infid_qs_noise_phase_control = OperationNoiseInfidelity(
            solver=time_slot_comp_qs_noise_phase_control,
            target=rabi.x_half,
            fidelity_measure='entanglement',
            label=['Entanglement Fidelity QS-Noise Phase Control'],
            neglect_systematic_errors=True
        )

        dynamics_phase_control_qs_noise = Simulator(
            solvers=[time_slot_comp_qs_noise_phase_control, ],
            cost_funcs=[entanglement_infid_qs_noise_phase_control, ]
        )

        np.random.seed(0)
        inital_pulse = random_phase_control_pulse(rabi.n_time_samples)

        _, rel_grad_deviation_unperturbed = dynamics_phase_control.\
            compare_numeric_to_analytic_gradient(inital_pulse)
        self.assertLess(rel_grad_deviation_unperturbed, 2e-6)

        _, rel_grad_deviation_qs_noise = dynamics_phase_control_qs_noise.\
            compare_numeric_to_analytic_gradient(inital_pulse)
        self.assertLess(rel_grad_deviation_qs_noise, 5e-5)
