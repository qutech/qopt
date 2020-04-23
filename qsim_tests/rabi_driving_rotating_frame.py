from matrix import OperatorDense
from noise import NTGQuasiStatic
from solver_algorithms import SchroedingerSMonteCarlo, LindbladSolver

import tests.rabi_driving_setup as rabi
import numpy as np
import scipy.optimize

from unittest import TestCase


sigma_z = OperatorDense(np.asarray([[1, 0], [0, -1]]))
h_drift = OperatorDense(np.zeros((2, 2)))

n_time_steps = 120
n_noise_traces = 200  # int(10 * 60 * 1e6 / 35)
evolution_time = 35e-6

delta_t = evolution_time / n_time_steps
delta_rabi = 1.5 / 10 * 1e6

# Tom
# todo: he seems to assume angular frequencies in his spectrum
S_01 = 3e8
S_02 = 3e4
# S(f) = S_01 / f + S_02 / f^2

f_min = 1 / 10 / 60  # 1 over 10 minutes
f_max = 1 / 35e-6

variance_f = S_01 * (np.log(f_max) - np.log(f_min)) \
             - S_02 * (1 / f_max - 1 / f_min)
sigma_f = np.sqrt(variance_f)

variance_lindbladt = S_01 / delta_rabi
"""
# Yoneda
S_0 = 3.2 * 1e6 * 4 * np.pi * np.pi

f_min = 1e-2
f_max = 1 / 35e-6

variance_f = S_0 * (np.log(f_max) - np.log(f_min))
sigma_f = np.sqrt(variance_f)  # 29 kHz
"""

down = np.asarray([[0, 1]])
up = np.asarray([[1], [0]])
x_half = rabi.x_half.data

projector_left = down @ x_half
projector_right = x_half @ up


class RabiDrivingRotatingFrame(TestCase):

    def test_quasi_static_noise(self):

        expected_t2star = np.sqrt(2 / variance_f)

        ntg = NTGQuasiStatic(
            standard_deviation=[sigma_f, ],
            n_samples_per_trace=n_time_steps,
            n_traces=n_noise_traces,
            always_redraw_samples=False,
            sampling_mode='uncorrelated_deterministic'
        )

        tslot_comp = SchroedingerSMonteCarlo(
            h_drift=[h_drift, ] * n_time_steps,
            h_ctrl=[.5 * sigma_z, ],
            h_noise=[.5 * sigma_z],
            initial_state=OperatorDense(np.eye(2)),
            tau=[delta_t, ] * n_time_steps,
            noise_trace_generator=ntg,
            exponential_method='Frechet'
        )

        def up_amplitude(unitary):
            probability = projector_left @ unitary.data @ projector_right
            return np.abs(probability) ** 2

        tslot_comp.set_ctrl_amps(
            (2 * np.pi * delta_rabi) * np.ones((n_time_steps, 1)))
        forward_propagators = tslot_comp.forward_propagators_noise

        propabilities = np.zeros((n_noise_traces, n_time_steps))
        for i in range(n_noise_traces):
            for j in range(n_time_steps):
                propabilities[i, j] = up_amplitude(forward_propagators[i][j])

        propabilities = np.mean(propabilities, axis=0)
        # plt.figure()
        # plt.plot(delta_t * np.arange(n_time_steps), propabilities, marker='.')

        def t2star_decay(t, delta_f, t2_star):
            return .5 * np.exp(-(t / t2_star) ** 2) * np.cos(
                2 * np.pi * delta_f * t) + .5

        popt, pcov = scipy.optimize.curve_fit(
            t2star_decay,
            xdata=delta_t * np.arange(n_time_steps),
            ydata=propabilities,
            p0=np.asarray([delta_rabi, expected_t2star])
        )

        self.assertLess(
            np.linalg.norm(
                propabilities - t2star_decay(
                    delta_t * np.arange(n_time_steps),
                    popt[0],
                    popt[1]
                )
            ) / len(propabilities),
            1e-3
        )

        self.assertLess(
            np.abs((expected_t2star - popt[1]) / (expected_t2star + popt[1])),
            1e-2
        )

        """
        plt.plot(
            delta_t * np.arange(n_time_steps),
            t2star_decay(delta_t * np.arange(n_time_steps),
                         delta_f=popt[0],
                         t2_star=popt[1])
        )
        """

    def test_fast_noise(self):

        def prefactor_function(transferred_parameters):
            return variance_lindbladt * np.ones_like(
                transferred_parameters)

        expected_t2_lindbladt = 2 / variance_lindbladt

        lindbladt_operators = [
            .5 * OperatorDense(np.asarray([[0, 1], [1, 0]])), ]

        tslot_comp_lindblad = LindbladSolver(
            h_drift=[h_drift, ] * n_time_steps,
            h_ctrl=[.5 * sigma_z, ],
            initial_state=OperatorDense(np.eye(4)),
            tau=[delta_t, ] * n_time_steps,
            exponential_method='Frechet',
            lindblad_operators=lindbladt_operators,
            prefactor_function=prefactor_function
        )

        tslot_comp_lindblad.set_ctrl_amps(
            (2 * np.pi * delta_rabi) * np.ones((n_time_steps, 1)))

        forward_propagators_lindbladt = tslot_comp_lindblad.forward_propagators

        def vec_to_density_matrix(vec: np.ndarray):
            return vec @ vec.conj().T

        def linearize_matrix(matrix: np.ndarray):
            return matrix.T.flatten()

        def vector_to_matrix(vec: np.ndarray):
            return vec.reshape((2, 2)).T

        initial = linearize_matrix(vec_to_density_matrix(x_half @ up))
        probabilities_lindbladt = np.zeros(len(forward_propagators_lindbladt))
        probabilities_lindbladt_c = np.zeros(len(forward_propagators_lindbladt),
                                             dtype=complex)

        sigma_x = np.asarray([[0, 1], [1, 0]])

        for i, prop in enumerate(forward_propagators_lindbladt):
            density = prop.data @ initial
            density = vector_to_matrix(density)
            probabilities_lindbladt[i] = np.trace(density @ sigma_x)
            probabilities_lindbladt_c[i] = np.trace(density @ sigma_x)

        def t2_time(t, t2):
            return -1 * np.sin(2 * np.pi * t * delta_rabi) \
                   * np.exp(-.5 * t / t2)

        popt, pcov = scipy.optimize.curve_fit(
            t2_time,
            xdata=delta_t * np.arange(n_time_steps + 1),
            ydata=probabilities_lindbladt,
            p0=np.asarray([expected_t2_lindbladt])
        )

        self.assertLess(
            np.linalg.norm(probabilities_lindbladt - t2_time(
                delta_t * np.arange(n_time_steps + 1),
                t2=popt[0])) / len(probabilities_lindbladt),
            1e-6
        )

        self.assertLess(
            np.abs(
                (expected_t2_lindbladt - popt[0])
                / (expected_t2_lindbladt + popt[0])),
            1e-6
        )

        """
        plt.figure()
        plt.plot(delta_t * np.arange(n_time_steps + 1), probabilities_lindbladt)
        plt.plot(
            delta_t * np.arange(n_time_steps + 1),
            t2_time(delta_t * np.arange(n_time_steps + 1),
                    t2=popt[0])
        )
        
        plt.plot(
            delta_t * np.arange(n_time_steps + 1),
            t2_time(delta_t * np.arange(n_time_steps + 1),
                    t2=2e-2 * expected_t2)
        )
        """
