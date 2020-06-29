from qsim.matrix import DenseOperator
from qsim.noise import NTGQuasiStatic
from qsim.solver_algorithms import SchroedingerSMonteCarlo, LindbladSolver, \
    SchroedingerSolver
from qsim.amplitude_functions import CustomAmpFunc
from qsim.transfer_function import OversamplingTF

import examples.rabi_driving.setup as rabi
import numpy as np
import scipy.optimize

from unittest import TestCase


class Lab_frame_rabi_driving(TestCase):
    def quasi_static_noise(self):

        sigma_z = DenseOperator(np.asarray([[1, 0], [0, -1]]))
        sigma_x = DenseOperator(np.asarray([[0, 1], [1, 0]]))
        h_drift = DenseOperator(np.zeros((2, 2)))

        # reference_frequency = 20e9 * 2 * np.pi
        reference_frequency = 100e6 * 2 * np.pi
        driving_frequency = 1e6 * 2 * np.pi

        # 100 per reference period
        # int(reference_frequency / driving_frequency) to make one driving period
        # 20 driving periods
        n_time_steps = int(35 * reference_frequency / driving_frequency * 20)
        n_noise_traces = 100  # int(10 * 60 * 1e6 / 35)
        evolution_time = 35e-6

        delta_t = evolution_time / n_time_steps

        down = np.asarray([[0], [1]])
        up = np.asarray([[1], [0]])
        x_half = rabi.x_half.data

        projector_left = up.T
        projector_right = up

        def up_amplitude(unitary):
            probability = projector_left @ unitary.data @ projector_right
            return np.abs(probability) ** 2

        ctrl_amps = delta_t * np.arange(1, 1 + n_time_steps)
        ctrl_amps = driving_frequency * np.sin(reference_frequency * ctrl_amps)


        def rabi_driving(transferred_parameters, **_):
            ctrl_amps = delta_t * np.arange(1, 1 + n_time_steps)
            ctrl_amps = 2 * np.sin(reference_frequency * ctrl_amps)
            # times 2 because the rabi frequency is .5 * Amplitude
            ctrl_amps = np.einsum("tc, t->tc", transferred_parameters, ctrl_amps)
            return ctrl_amps


        def rabi_driving_noise(noise_samples, **_):
            ctrl_amps = delta_t * np.arange(1, 1 + n_time_steps)
            ctrl_amps = 2 * np.sin(reference_frequency * ctrl_amps)
            ctrl_amps = np.einsum("sno, t->tno", noise_samples, ctrl_amps)
            return ctrl_amps


        rabi_driving_amp_func = CustomAmpFunc(value_function=rabi_driving,
                                              derivative_function=None)
        id_transfer_func = OversamplingTF(oversampling=n_time_steps)
        id_transfer_func.set_times(np.asarray([evolution_time]))


        ts_comp_unperturbed = SchroedingerSolver(
            h_drift=[reference_frequency * .5 * sigma_z, ] * n_time_steps,
            h_ctrl=[.5 * sigma_x, ],
            initial_state=DenseOperator(np.eye(2)),
            tau=[delta_t, ] * n_time_steps,
            exponential_method='Frechet',

        )

        ts_comp_lindblad = LindbladSolver(
            h_drift=[reference_frequency * .5 * sigma_z, ] * n_time_steps,
            h_ctrl=[.5 * sigma_x, ],
            initial_state=DenseOperator(np.eye(2)),
            tau=[delta_t, ] * n_time_steps,
            exponential_method='Frechet'
        )

        ts_comp_unperturbed.set_optimization_parameters(np.expand_dims(ctrl_amps, 1))


        """
        # unperturbed:
        forward_propagators = ts_comp_unperturbed.forward_propagators
        
        propabilities = np.zeros((n_time_steps, ))
        for j in range(n_time_steps):
            propabilities[j] = up_amplitude(forward_propagators[j])
        
        plt.figure()
        plt.plot(delta_t * np.arange(n_time_steps), propabilities)
        """

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


        """
        # Yoneda
        S_0 = 3.2 * 1e6 * 4 * np.pi * np.pi
        
        f_min = 1e-2
        f_max = 1 / 35e-6
        
        variance_f = S_0 * (np.log(f_max) - np.log(f_min))
        sigma_f = np.sqrt(variance_f)  # 29 kHz
        """
        expected_t2star = np.sqrt(2 / variance_f)

        ntg = NTGQuasiStatic(
            standard_deviation=[sigma_f, ],
            n_samples_per_trace=1,
            n_traces=n_noise_traces,
            always_redraw_samples=False,
            sampling_mode='monte_carlo'
        )


        tslot_comp = SchroedingerSMonteCarlo(
            h_drift=[reference_frequency * .5 * sigma_z, ] * n_time_steps,
            h_ctrl=[.5 * sigma_x, ],
            h_noise=[.5 * sigma_x],
            initial_state=DenseOperator(np.eye(2)),
            tau=[delta_t, ] * n_time_steps,
            noise_trace_generator=ntg,
            exponential_method='Frechet',
            transfer_function=id_transfer_func,
            amplitude_function=rabi_driving_amp_func,
            noise_amplitude_function=rabi_driving_noise
        )

        """
        # for the rotating frame
        delta_rabi = 1.5 / 10 * 1e6
        tslot_comp.set_optimization_parameters(
            (2 * np.pi * delta_rabi) * np.ones((n_time_steps, 1)))
        """
        tslot_comp.set_optimization_parameters(np.asarray([[driving_frequency]]))

        forward_propagators = tslot_comp.forward_propagators_noise

        propabilities = np.zeros((n_noise_traces, n_time_steps))
        for i in range(n_noise_traces):
            for j in range(n_time_steps):
                propabilities[i, j] = up_amplitude(forward_propagators[i][j])

        propabilities = np.mean(propabilities, axis=0)

        """
        def t2star_decay(t, delta_f, t2_star):
            return .5 * np.exp(-(t / t2_star) ** 2) * np.cos(
                2 * np.pi * delta_f * t) + .5
        """


        def t2star_decay(t, sigma_driving):
            return .5 * np.exp(-.5 * (sigma_driving * t) ** 2) * np.cos(driving_frequency * t) + .5


        def t2star_decay_2(t, sigma_driving):
            return .5 * (1 + (sigma_driving ** 2 / driving_frequency * t)) ** -.25 * np.cos(driving_frequency * t) + .5


        def t2star_decay_3(t, sigma_driving, sigma_ref):
            up_prop = np.exp(-.5 * (sigma_driving * t) ** 2)
            up_prop *= (1 + (sigma_ref ** 2 / driving_frequency * t) ** 2) ** -.25
            up_prop *= .5 * np.cos(driving_frequency * t)
            up_prop += .5
            return up_prop


        def t2star_decay_4(t, sigma_driving, sigma_ref, periodicity):
            up_prop = np.exp(-.5 * (sigma_driving * t) ** 2)
            up_prop *= (1 + ((sigma_ref ** 2) / periodicity * t) ** 2) ** -.25
            up_prop *= .5 * np.cos(periodicity * t)
            up_prop += .5
            return up_prop


        def t2star_decay_5(t, sigma_driving, sigma_ref, periodicity, lin_decay):
            up_prop = np.exp(-.5 * (sigma_driving * t) ** 2)
            up_prop *= np.exp(-1 * lin_decay * t)
            up_prop *= (1 + ((sigma_ref ** 2) / periodicity * t) ** 2) ** -.25
            up_prop *= .5 * np.cos(periodicity * t)
            up_prop += .5
            return up_prop


        popt, pcov = scipy.optimize.curve_fit(
            t2star_decay_3,
            xdata=delta_t * np.arange(n_time_steps),
            ydata=propabilities,
            p0=np.asarray([sigma_f, sigma_f])
        )


        popt, pcov = scipy.optimize.curve_fit(
            t2star_decay_5,
            xdata=delta_t * np.arange(n_time_steps),
            ydata=propabilities,
            p0=np.asarray([sigma_f, sigma_f, driving_frequency, sigma_f])
        )

        self.assertLess(np.linalg.norm(
            propabilities - t2star_decay(delta_t * np.arange(n_time_steps),
                           sigma_driving=sigma_f)) / len(propabilities), 1e-3)

        """
        plt.figure()
        plt.plot(delta_t * np.arange(n_time_steps), propabilities)
        
        plt.plot(
            delta_t * np.arange(n_time_steps),
            t2star_decay_3(delta_t * np.arange(n_time_steps),
                           sigma_driving=popt[0],
                           sigma_ref=popt[1])
        )
        
        plt.plot(
            delta_t * np.arange(n_time_steps),
            t2star_decay(delta_t * np.arange(n_time_steps),
                           sigma_driving=sigma_f)
        )
        
        
        plt.plot(
            delta_t * np.arange(n_time_steps),
            t2star_decay_3(delta_t * np.arange(n_time_steps),
                           sigma_driving=sigma_f,
                           sigma_ref=sigma_f)
        )
        
        plt.plot(
            delta_t * np.arange(n_time_steps),
            t2star_decay_5(delta_t * np.arange(n_time_steps),
                           sigma_driving=popt[0],
                           sigma_ref=popt[1],
                           periodicity=popt[2],
                           lin_decay=popt[3])
        )
        """
