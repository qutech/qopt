"""
This file provides the setup for the simulation of a rabi driving experiment.

author: Julian Teske julian.teske@fz-juelich.de

"""
from qopt import *
import numpy as np

__all__ = ['build_lab_solver', 'build_rotational_frame_solver',
           'transform_ket_to_rot_frame', 'transform_prop_to_rot_frame']


def build_lab_solver(
        resonance_frequency,
        n_time_steps,
        delta_t,
        exponential_method='spectral'
):
    """
    Creates an instance of SchroedingerSolver for a qubit subjected to Rabi
    driving and described in the lab frame.

    Parameters
    ----------
    resonance_frequency: float
        The qubit's resonance frequency.

    n_time_steps: int
        Number of time steps.

    delta_t: float
        Length of time steps.

    exponential_method: str in ['spectral', 'Frechet'], optional
        Numeric method for the calculation of matrix exponentials, required to
        solve Schrodinger's equation. Defaults to 'spectral' because
        'Frechet' might have issues when used in after QuTiP's plot routines.

    Returns
    -------
    solver: SchroedingerSolver
        The created solver.

    """
    h_ctrl = [.5 * DenseOperator.pauli_x()]
    h_drift = [resonance_frequency * .5 * DenseOperator.pauli_z()]
    tau = delta_t * np.ones(n_time_steps)

    def periodic_signal(opt_pars):
        """
        Periodic driving signal. Mimics e.g. a microwave source.

        Parameters
        ----------
        opt_pars: array of float
            Optimization parameters.

        Returns
        -------
        signal: array, shape (n_time_steps, 1)

        Todo: replace arange by linspace

        """
        amplitude = opt_pars[0]
        frequency = opt_pars[1]
        phase_shift = opt_pars[2]
        pulse = amplitude * np.sin(
            frequency * np.arange(n_time_steps) * delta_t + phase_shift)
        return np.expand_dims(pulse, axis=1)

    # The derivative is not required, as we will not perform any pulse
    # optimization.
    def periodic_signal_deriv(_):
        raise NotImplementedError

    amplitude_function = CustomAmpFunc(
        value_function=periodic_signal,
        derivative_function=periodic_signal_deriv
    )

    solver = SchroedingerSolver(
        h_drift=h_drift,
        h_ctrl=h_ctrl,
        tau=tau,
        amplitude_function=amplitude_function,
        exponential_method=exponential_method
    )
    return solver


def build_rotational_frame_solver(
        frequency_detuning,
        n_time_steps,
        delta_t,
        exponential_method='spectral'
):
    """
    Creates an instance of SchroedingerSolver for a qubit subjected to Rabi
    driving and described in the rotational frame.

    Parameters
    ----------
    frequency_detuning: float
        Detuning between the driving and the resonance frequency.

    n_time_steps: int
        Number of time steps.

    delta_t: float
        Length of time steps.

    exponential_method: str in ['spectral', 'Frechet'], optional
        Numeric method for the calculation of matrix exponentials, required to
        solve Schrodinger's equation. Defaults to 'spectral' because
        'Frechet' might have issues when used in after QuTiP's plot routines.

    Returns
    -------
    solver: SchroedingerSolver
        The created solver.

    """
    h_ctrl = [.5 * DenseOperator.pauli_x(), .5 * DenseOperator.pauli_y()]
    h_drift = [frequency_detuning * .5 * DenseOperator.pauli_z()]
    tau = delta_t * np.ones(n_time_steps)

    solver = SchroedingerSolver(
        h_drift=h_drift,
        h_ctrl=h_ctrl,
        tau=tau,
        exponential_method=exponential_method
    )
    return solver


def transform_prop_to_rot_frame(
        propagator: DenseOperator,
        time_end: float,
        time_start: float,
        resonance_frequency: float
):
    """
    Transforms a propagator which propagates state vectors from time t_start
    to time t_end.

    Parameters
    ----------
    propagator: DenseOperator
        The propagator from time_start to time_end in the lab frame.

    time_end: float
        Final time of the propagator.

    time_start: float
        Initial time of the propagator.

    resonance_frequency: float
        The qubit's resonance frequency.

    Returns
    -------
    returned_prop: DenseOperator
        The transformed propagator in the rotational frame.

    """
    u_end = (-.5 * DenseOperator.pauli_z()).exp(
        1j * time_end * resonance_frequency)
    u_start = (-.5 * DenseOperator.pauli_z()).exp(
        1j * time_start * resonance_frequency)
    return u_end.dag() * propagator * u_start


def transform_ket_to_rot_frame(
        vector: DenseOperator,
        time,
        resonance_frequency
):
    """
    Transforms a state vector into the rotational frame at time.

    Parameters
    ----------
    vector: DenseOperator
        State vector in the lab frame.

    time: float
        Time of transformation.

    resonance_frequency: float
        The qubit's resonance frequency.

    Returns
    -------
    transformed_vector: DenseOperator
        Transformed vector in the rotational frame.

    """
    u = (.5 * DenseOperator.pauli_z()).exp(1j * time * resonance_frequency)
    return u.dag() * vector
