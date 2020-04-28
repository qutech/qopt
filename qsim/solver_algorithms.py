"""
The TimeSlotComputer calculates the propagators as solutions to Schroedingers
equation or a master equation in Lindblad form.

If requested, also derivatives of the propagators by the control amplitudes are
calculated or approximated. Please note that this Docstring only documents the
classes currently supported.

Classes
-------
Solver
    Abstract base class of the time slot computers.

SchroedingerSolver
    Basic time slot computer which solves the unperturbed Schroedinger equation.

SchroedingerSMonteCarlo
    Time slot computer which solves the Schroedinger equation for a number of
    noise traces.

LindbladSolver
    needs refactoring

LindbladSControlNoise
    needs refactoring

Functions
---------
lindblad_to_super_operator_old
    needs refactoring

"""

import numpy as np

from filter_functions import pulse_sequence
from filter_functions import plotting
from filter_functions import basis
from typing import Optional, List, Callable, Union
from abc import ABC, abstractmethod

import copy


import qutip.logging_utils as qutip_logging

from qsim import noise, matrix, matrix as q_mat

from qsim.transfer_function import TransferFunction
from qsim.amplitude_functions import AmplitudeFunction
from qsim.util import needs_refactoring

logger = qutip_logging.get_logger()


class Solver(ABC):
    """
    Abstract base class for Time Slot Computers

    Parameters
    ----------
    h_ctrl: List[ControlMatrix], len:  num_ctrl
        Control operators in the Hamiltonian as nested list of
        shape n_t, num_ctrl.

    h_drift: List[ControlMatrix], len: num_t
        Drift operators in the Hamiltonian.

    initial_state : ControlMatrix
        Initial state of the system as state vector. Can also be set to the
        identity matrix. Then the forward propagation gives the total
        propagator of the system.

    tau: array of float, shape: (num_t, )
        Durations of the time slices.

    ctrl_amps: np.ndarray, shape: (num_t, num_ctrl), optional
        The initial control amplitudes.

    filter_function_h_n: List[List[np.array]] or List[List[Qobj]]
        Nested list of noise Operators. Used in the filter function
        formalism. filter_function_h_n should look something like this:

            H = [[n_oper1, n_coeff1, n_oper_identifier1],
                 [n_oper2, n_coeff2, n_oper_identifier2], ...]

        The operators may be given either as NumPy arrays or QuTiP Qobjs
        and each coefficient array should have the same number of elements
        as *dt*, and should be given in units of :math:`\hbar`. If not every
        sublist (read: operator) was given a identifier, they are automatically
        filled up with 'A_i' where i is the position of the operator.

    exponential_method: string, optional
        Method used by the ControlMatrix class for the calculation of the matrix
        exponential. The default is 'Frechet'. See also the Docstring of the
        file 'control_2.matrix'.

    is_skew_hermitian: bool
        Only important for the exponential_method 'spectral'. If set to true,
        the dynamical generator is assumed to be skew hermitian during the
        spectral decomposition.

    transfer_function: TransferFunction
        The transfer function for reshaping the optimization parameters.

    amplitude_function: AmplitudeFunction
        The amplitude function connecting the transferred optimization
        parameters to the control amplitudes.

    paranoia_level: int
        The paranoia_level determines how many checks are conducted.
        0: No tests
        1: Some tests
        2: Exhaustive tests, dimension checks


    Attributes
    ----------
    h_ctrl: List[ControlMatrix], len: num_ctrl
        Control operators in the Hamiltonian as list of lenght num_ctrl.

    h_drift: List[ControlMatrix], len: num_t
        Drift operators in the Hamiltonian.

    initial_state : ControlMatrix
        Initial state of the system as state vector. Can also be set to the
        identity matrix. Then the forward propagation gives the total
        propagator of the system.

    tau: List[float]
        Durations of the time slices.

    filter_function_h_n: List[List[np.array]] or List[List[Qobj]]
        Nested list of noise Operators. Used in the filter function
        formalism.

    filter_function_basis: Basis
        The filter function pulse sequence will be expressed in this basis.
        See documentation of the filter function package.

    exponential_method: string, optional
        Method used by the ControlMatrix class for the calculation of the matrix
        exponential. The default is 'Frechet'. See also the Docstring of the
        file 'control_2.matrix'.

    transfer_function: TransferFunction
        The transfer function for reshaping the optimization parameters.

    amplitude_function: AmplitudeFunction
        The amplitude function connecting the transferred optimization
        parameters to the control amplitudes.

    _prop: List[ControlMatrix], len: num_t
        Propagators of the system.

    _fwd_prop: List[ControlMatrix], len: num_t + 1
        Cumulation of the propagators. They describe the forward propagation
        of the systems state.

    _reversed_prop: List[ControlMatrix], len: num_t + 1
        Cumulation of propagators in reversed order.

    _derivative_prop: List[List[ControlMatrix]], shape: [[] * num_t] * num_ctrl
        Frechet derivatives of the propagators by the control amplitudes.

    Methods
    -------
    set_ctrl_amps(u, **kwargs):
        Set the control amplitudes. The key word arguments may include the
        key transferred parameters.

    propagators: List[ControlMatrix], len: num_t
        Returns the propagators of the system.

    forward_propagators: List[ControlMatrix], len: num_t + 1
        Returns the forward propagation of the initial state. The element
        forward_propagators[i] propagates a state by the first i time steps, if
        the initial state is the identity matrix.

    frechet_deriv_propagators: List[List[ControlMatrix]],
                               shape: [[] * num_t] * num_ctrl
        Returns the frechet derivatives of the propagators by the control
        amplitudes.

    reversed_propagators: List[ControlMatrix], len: num_t + 1
        Returns the reversed propagation of the initial state. The element
        reversed_propagators[i] propagates a state by the last i time steps, if
        the initial state is the identity matrix.

    _compute_propagation: abstract method
        Computes the propagators.

    _compute_forward_propagation:
        Compute the forward propagation of the initial state / system.

    _compute_reversed_propagation:
        Compute the reversed propagation of the initial state / system.

    _compute_propagation_derivatives: abstract method
        Compute the derivatives of the propagators by the control amplitudes.

    create_pulse_sequence(new_amps):
        filter_functions.pulse_sequence.PulseSequence
        Creates a pulse sequence instance corresponding to the current control
        amplitudes.

    plot_bloch_sphere:
        Uses a pulse sequence to plot the systems evolution on the blochs
        sphere. For 2 dimensional systems only.


    TODO:
        * Write parser
            * setter for new hamiltonians
            * make hamiltonians private
            * also for the initial state
        * Implement the drift operator with an amplitude. Right now,
            * the operator is already multiplied with the amplitude, which is
            * not coherent with the pulse sequence interface. Alternatively
            * amplitude=1?
        * document the text attributes. implement their purpose?
        * tau should be taken from the transfer function

    """

    def __init__(
            self,
            h_drift: List[q_mat.OperatorMatrix],
            h_ctrl: List[q_mat.OperatorMatrix],
            initial_state: q_mat.OperatorMatrix,
            tau: np.ndarray,
            ctrl_amps: Optional[np.ndarray] = None,
            opt_pars: Optional[np.ndarray] = None,
            filter_function_h_n: Union[List[List], np.ndarray, None] = None,
            filter_function_basis: Optional[basis.Basis] = None,
            exponential_method: Optional[str] = None,
            is_skew_hermitian: bool = True,
            transfer_function: Optional[TransferFunction] = None,
            amplitude_function: Optional[AmplitudeFunction] = None,
            paranoia_level: int = 2
    ):
        self.id_text = 'TS_COMP_BASE'
        self.cache_text = 'Save'

        self.h_drift = h_drift
        self.h_ctrl = h_ctrl
        self._ctrl_amps = ctrl_amps
        self._opt_pars = opt_pars

        self.initial_state = initial_state
        self.tau = tau

        if exponential_method is None:
            self.exponential_method = 'Frechet'
        else:
            self.exponential_method = exponential_method

        self._prop = None
        self._fwd_prop = None
        self._reversed_prop = None
        self._derivative_prop = None

        self.pulse_sequence = None

        if filter_function_h_n is None:
            self.filter_function_h_n = []
        else:
            self.filter_function_h_n = filter_function_h_n
        self.filter_function_basis = filter_function_basis

        self._is_skew_hermitian = is_skew_hermitian

        self.transfer_function = transfer_function
        self.amplitude_function = amplitude_function
        self.transferred_parameters = None

        self.consistency_checks(paranoia_level=paranoia_level)

    def set_ctrl_amps(self, x: np.ndarray) -> None:
        """
        Set the control amplitudes.

        All computation flags are set to false.

        The new control amplitudes u are calculated:
        u: np.array, shape: (num_t, num_ctrl)

        Parameters
        ----------
        x: np.array, shape: (num_x, num_ctrl)
            Optimization parameters.

        """

        if np.array_equal(self._opt_pars, x):
            return
        else:
            self._opt_pars = x

        if self.transfer_function is not None:
            self.transferred_parameters = self.transfer_function(x)
        else:
            self.transferred_parameters = x

        if self.amplitude_function is not None:
            u = self.amplitude_function(
                self.transferred_parameters)
        else:
            u = self.transferred_parameters

        if len(u.shape) != 2:
            raise ValueError('The new control amplitudes set to the time slot'
                             'computer must have two dimensions! '
                             '(time, control operator)')

        if u.shape[0] != len(self.tau):
            raise ValueError('The new control amplitudes do not have the '
                             'correct number of entries on the time axis!')

        if u.shape[1] != len(self.h_ctrl):
            raise ValueError('The new control amplitudes do not have the '
                             'correnct number of entries on the control axis!')

        self._ctrl_amps = u
        self._prop = None
        self._fwd_prop = None
        self._derivative_prop = None
        self._reversed_prop = None
        self.pulse_sequence = None

    def consistency_checks(self, paranoia_level: int):
        if paranoia_level == 0:
            return

        elif paranoia_level >= 1:
            # check whether the hamiltonian is correct for the number of time
            # steps
            if isinstance(self.tau, List):
                self.tau = np.asarray(self.tau)
            if len(self.tau.shape) > 1:
                raise ValueError("Tau must be a one dimensional numpy array or"
                                 "a list.")
            n_time_steps = self.tau.shape[0]
            if not (n_time_steps == len(self.h_drift)
                    or len(self.h_drift) == 0):
                raise ValueError("The drift hamiltonian must have exactly one "
                                 "entry for each time step or no entry at all.")
            if paranoia_level >= 2:
                # check whether the Hamiltonian has the correct dimensions
                dim = self.h_ctrl[0].shape[0]

                for ctrl_matrix in self.h_ctrl:
                    assert(dim == ctrl_matrix.shape[0])
                    assert(dim == ctrl_matrix.shape[1])

                for drift_matrx in self.h_drift:
                    assert(dim == drift_matrx.shape[0])
                    assert(dim == drift_matrx.shape[1])

        else:
            raise ValueError("The paranoia level must be a positive integer.")

    @property
    def propagators(self) -> List[q_mat.OperatorMatrix]:
        """
        Returns the propagators of the system and calculates them if necessary.

        Returns
        -------
        propagators: List[ControlMatrix], len: num_t
            Propagators of the system.

        """
        if self._prop is None:
            self._compute_propagation()
        return self._prop

    @property
    def forward_propagators(self) -> List[q_mat.OperatorMatrix]:
        """
        Returns the forward propagation of the initial state for every time
        slice and calculate it if necessary. If the initial state is the
        identity matrix, then the cumulative propagators are given. The element
        forward_propagators[i] propagates a state by the first i time steps, if
        the initial state is the identity matrix.

        Returns
        -------
        forward_propagation: List[ControlMatrix], len: num_t + 1
            Propagation of the initial state of the system. fwd[0] gives the
            initial state itself.

        """
        if self._fwd_prop is None:
            self._compute_forward_propagation()
        return self._fwd_prop

    @property
    def frechet_deriv_propagators(self) -> List[List[q_mat.OperatorMatrix]]:
        """
        Returns the frechet derivatives of the propagators.

        Returns
        -------
        derivative_prop: List[List[ControlMatrix]],
                         shape: [[] * num_t] * num_ctrl
            Frechet derivatives of the propagators by the control amplitudes

        """
        if self._derivative_prop is None:
            self._compute_propagation_derivatives()
        return self._derivative_prop

    @property
    def reversed_propagators(self) -> List[q_mat.OperatorMatrix]:
        """
        Returns the reversed propagation of the initial state for every time
        slice and calculate it if necessary. If the initial state is the
        identity matrix, then the reversed cumulative propagators are given. The
        element forward_propagators[i] propagates a state by the first i time
        steps, if the initial state is the identity matrix.

        Returns
        -------
        reversed_propagation: List[ControlMatrix], len: num_t + 1
            Propagation of the initial state of the system. reversed[0] gives
            the initial state itself.

        """
        if self._reversed_prop is None:
            self._compute_reversed_propagation()
        return self._reversed_prop

    @abstractmethod
    def _compute_propagation(self) -> None:
        """
        Computes the propagators. Must set self._prop!

        Raises
        ------
        ValueError
            If the control amplitudes are not set.

        """
        if self._ctrl_amps is None:
            raise ValueError("The control amplitudes must be set to calculate "
                             "the propagation!")

    def _compute_forward_propagation(self) -> None:
        """Computes the forward propagators. """
        if self._prop is None:
            self._compute_propagation()
        self._fwd_prop = [self.initial_state.copy(), ]
        for prop in self._prop:
            self._fwd_prop.append(prop * self._fwd_prop[-1])

    def _compute_reversed_propagation(self) -> None:
        """Compute the reversed propagation. """
        if self._prop is None:
            self._compute_propagation()

        if type(self.initial_state) == matrix.DenseOperator:
            self._reversed_prop = [matrix.DenseOperator(
                np.eye(self._prop[0].shape[0])) * (1 + 0j), ]
        elif type(self.initial_state) == matrix.SparseOperator:
            self._reversed_prop = [matrix.SparseOperator(
                np.eye(self._prop[0].shape[0])) * (1 + 0j), ]
        else:
            raise TypeError("The initial state should be either a dense or "
                            "sparse control matrix.")

        for prop in self._prop[::-1]:
            self._reversed_prop.append(self._reversed_prop[-1] * prop)

    @abstractmethod
    def _compute_propagation_derivatives(self) -> None:
        """Compute the derivatives of the propagators by the control amplitudes.
        """
        pass

    def create_pulse_sequence(
            self, new_amps: Optional[np.ndarray] = None,
            ff_basis: Optional[basis.Basis] = None
    ) -> pulse_sequence:
        """
        Create a pulse sequence of the filter function package written by
        Tobias Hangleiter.

        See the documentation of the filter function package.

        Paramters
        ---------
        new_amps: np.ndarray, shape: (num_t, num_ctrl)
            New control amplitudes can be set before the pulse sequence is
            initialized.

        ff_basis: Basis
            The pulse sequence will be expanded in this basis. See documentation
            of the filter function package.

        Returns
        -------
        pulse_sequence: filter_functions.pulse_sequence.PulseSequence
            The pulse sequence corresponding to the control model and the
            control amplitudes set.

        """
        if new_amps is not None:
            self.set_ctrl_amps(new_amps)
        h_n = self.filter_function_h_n
        if not h_n:
            h_n = [[np.zeros(self.h_ctrl[0].shape),
                    np.zeros((len(self.tau), ))]]
        h_c = []
        for drift_operator in [self.h_drift[0], ]:
            if type(drift_operator) == matrix.DenseOperator:
                drift_operator = drift_operator.data
            h_c += [[drift_operator, len(self.tau) * [1]], ]
        for i, control_operator in enumerate(self.h_ctrl):
            h_c += [[control_operator.data, self._ctrl_amps[:, i]], ]

        dt = self.tau

        if ff_basis is not None:
            self.pulse_sequence = pulse_sequence.PulseSequence(
                h_c, h_n, dt, basis=ff_basis)
        elif self.filter_function_basis is not None:

            self.pulse_sequence = pulse_sequence.PulseSequence(
                h_c, h_n, dt, basis=self.filter_function_basis)
        else:
            self.pulse_sequence = pulse_sequence.PulseSequence(h_c, h_n, dt)

        return self.pulse_sequence

    def plot_bloch_sphere(self) -> None:
        """
        Uses the pulse sequence to plot the systems evolution on the bloch
        sphere.

        Only available for two dimensional systems.

        """
        if self.pulse_sequence is None:
            self.create_pulse_sequence()
        plotting.plot_bloch_vector_evolution(self.pulse_sequence, n_samples=500)


class SchroedingerSolver(Solver):
    """
    This time slot computer solves the unperturbed Schroedinger equation. All
    intermediary propagators are calculated and cached.

    Parameters
    ----------
    calculate_propagator_derivatives: bool
        If true, the derivatives of the propagators by the control amplitudes
        are always calculated. Otherwise only on demand.

    frechet_deriv_approx_method: Optional[str]
        Method for the approximation of the derivatives of the propagators, if
        they are not calculated analytically. Note that this method is never
        used if calculate_propagator_derivatives is set to True!
        Methods:
        None: The derivatives are not approximated by calculated by the control
        matrix class.
        'grape': use the approximation given in the original grape paper.

    Attributes
    ----------
    _dyn_gen: List[ControlMatrix], len: num_t
        The generators of the systems dynamics

    calculate_propagator_derivatives: bool
        If true, the derivatives of the propagators by the control amplitudes
        are always calculated. Otherwise only on demand.

    frechet_deriv_approx_method: Optional[str]
        Method for the approximation of the derivatives of the propagators, if
        they are not calculated analytically. Note that this method is never
        used if calculate_propagator_derivatives is set to True!
        Methods:
        'grape': use the approximation given in the original grape paper.

    Methods
    -------
    _compute_derivative_directions: List[List[q_mat.ControlMatrix]],
                                    shape: [[] * num_ctrl] * num_t
        Computes the directions of change with respect to the control
        parameters.

    _compute_dyn_gen: List[ControlMatrix], len: num_t
        Computes the dynamics generators.

    Todo:
        * raise a warning if the approximation method although the gradient
        is always calculated.
        * raise a warning if the grape approximation is chosen but its
        requirement of small time steps is not met.

    """

    def __init__(self,
                 h_drift: List[q_mat.OperatorMatrix],
                 h_ctrl: List[q_mat.OperatorMatrix],
                 initial_state: q_mat.OperatorMatrix,
                 tau: List[float],
                 ctrl_amps: Optional[np.ndarray] = None,
                 calculate_propagator_derivatives: bool = True,
                 filter_function_h_n: Optional[List] = None,
                 filter_function_basis: Optional[basis.Basis] = None,
                 exponential_method: Optional[str] = None,
                 frechet_deriv_approx_method: Optional[str] = None,
                 is_skew_hermitian: bool = True,
                 transfer_function: Optional[TransferFunction] = None,
                 amplitude_function: Optional[AmplitudeFunction] = None):
        super().__init__(
            h_drift=h_drift, h_ctrl=h_ctrl, initial_state=initial_state,
            tau=tau, ctrl_amps=ctrl_amps,
            filter_function_h_n=filter_function_h_n,
            filter_function_basis=filter_function_basis,
            exponential_method=exponential_method,
            is_skew_hermitian=is_skew_hermitian,
            transfer_function=transfer_function,
            amplitude_function=amplitude_function
        )
        self.id_text = 'ALL'
        self.cache_text = 'Save'
        self.calculate_propagator_derivatives = calculate_propagator_derivatives
        self.frechet_deriv_approx_method = frechet_deriv_approx_method

        self._dyn_gen = None

    def set_ctrl_amps(self, x: np.ndarray) -> None:
        """See base class. """
        if not np.array_equal(self._opt_pars, x):
            self._dyn_gen = None
        super().set_ctrl_amps(x)

    def _compute_dyn_gen(self) -> List[q_mat.OperatorMatrix]:
        """
        Computes the dynamics generators.

        Returns
        -------
        dyn_gen: List[ControlMatrix], len: num_t
            This is basically the total Hamiltonian.

        """
        self._dyn_gen = [-1j * h for h in self.h_drift]
        for ctrl, ctrl_op in enumerate(self.h_ctrl):
            for dyn_gen, ctrl_amp in \
                    zip(self._dyn_gen, self._ctrl_amps[:, ctrl]):
                dyn_gen += -1j * ctrl_amp * ctrl_op
        return self._dyn_gen

    def _compute_derivative_directions(self) -> List[List[q_mat.OperatorMatrix]]:
        """
        The directions of the frechet derivatives are the control operators.

        No deep copy is required because the result is not used for in-place
        operations.

        """
        # The list is multiplied (copied by reference) because the elements
        # will not be manipulated in place. (only as copy)
        return [[operator * -1j for operator in self.h_ctrl], ] * len(self.tau)

    def _compute_propagation(
            self, calculate_propagator_derivatives: Optional[bool] = None) \
            -> None:
        """See base class. """
        super()._compute_propagation()

        if self._dyn_gen is None:
            self._dyn_gen = self._compute_dyn_gen()

        if calculate_propagator_derivatives is None:
            calculate_propagator_derivatives = \
                self.calculate_propagator_derivatives

        # initialize the attributes
        self._prop = [None for _ in range(len(self.tau))]

        if calculate_propagator_derivatives:
            derivative_directions = self._compute_derivative_directions()
            self._derivative_prop = [[None for _ in range(len(self.tau))]
                                     for _2 in range(len(self.h_ctrl))]
            for t in range(len(self.tau)):
                for ctrl in range(len(self.h_ctrl)):
                    self._prop[t], self._derivative_prop[ctrl][t] \
                        = self._dyn_gen[t].dexp(
                        derivative_directions[t][ctrl], self.tau[t],
                        compute_expm=True, method=self.exponential_method,
                        is_skew_hermitian=self._is_skew_hermitian)
        else:
            for t in range(len(self.tau)):
                self._prop[t] = self._dyn_gen[t].exp(
                    tau=self.tau[t], method=self.exponential_method,
                    is_skew_hermitian=self._is_skew_hermitian)

    def _compute_propagation_derivatives(self) -> None:
        """
        Computes the frechet derivatives of the propagators.

        The derivatives are not returned but cached. Since the function is only
        called when no derivatives are cached, the approximation is prioritised.
        """
        if not self.frechet_deriv_approx_method:
            self._compute_propagation(calculate_propagator_derivatives=True)
        elif self.frechet_deriv_approx_method == 'grape':
            if self._prop is None:
                self._compute_propagation(
                    calculate_propagator_derivatives=False)
            self._derivative_prop = [[None for _ in range(len(self.h_ctrl))]
                                     for _2 in range(len(self.tau))]
            derivative_directions = self._compute_derivative_directions()
            for t in range(len(self.tau)):
                for ctrl in range(len(self.h_ctrl)):
                    self._derivative_prop[t][ctrl] = \
                        self.tau[t] * derivative_directions[t][ctrl] \
                        * self._prop[t]
        else:
            raise ValueError('Unknown gradient derivative approximation method:'
                             + str(self.frechet_deriv_approx_method))


class SchroedingerSMonteCarlo(SchroedingerSolver):
    """
    This time slot computer solves the Schroedinger equation explicitly for
    concrete noise realizations. The noise traces are generated by an instance
    of the Noise Trace Generator Class. Then they can be processed by the
    noise amplitude function, before they are multiplied by the noise
    hamiltionians.

    Parameters
    ----------
    h_noise: List[ControlMatrix], len: num_noise_operators
        List of noise operators occurring in the Hamiltonian.

    noise_trace_generator: noise.NoiseTraceGenerator
        Noise trace generator object.

    noise_amplitude_function: Callable[[noise_samples: np.ndarray,
        optimization_parameters: np.ndarray,
        transferred_parameters: np.ndarray,
        control_amplitudes: np.ndarray], np.ndarray]
        The noise amplitude function calculated the noisy control amplitudes
        corresponding to the noise samples. They recieve 4 keyword arguments
        being the noise samples, the optimization parameters, the transferred
        optimization parameters and the control amplitudes in this order.
        The noise samples are given with the shape (n_samples_per_trace,
        n_traces, n_noise_operators), the optimization parameters
        (num_x, num_ctrl), the transferred parameters (num_t, num_ctrl) and
        the control amplitudes (num_t, num_ctrl). The returned noise amplitudes
        should be of the shape (num_t, n_traces, n_noise_operators).

    Attributes
    ----------
    h_noise: List[ControlMatrix], len: num_noise_operators
        List of noise operators occurring in the Hamiltonian.

    noise_trace_generator: noise.NoiseTraceGenerator
        Noise trace generator object.

    _dyn_gen_noise: List[List[ControlMatrix]],
                    shape: [[] * num_t] * num_noise_traces
        Dynamics generators for the individual noise traces.

    _prop_noise: List[List[ControlMatrix]],
                 shape: [[] * num_t] * num_noise_traces
        Propagators for the individual noise traces.

    _fwd_prop_noise: List[List[ControlMatrix]],
                     shape: [[] * (num_t + 1)] * num_noise_traces
        Cumulation of the propagators for the individual noise traces. They
        describe the forward propagation of the systems state.

    _reversed_prop_noise: List[List[ControlMatrix]],
                          shape: [[] * (num_t + 1)] * num_noise_traces
        Cumulation of propagators in reversed order for the individual noise
        traces.

    _derivative_prop_noise: List[List[List[ControlMatrix]]],
                            shape: [[[] * num_t] * num_ctrl] * num_noise_traces
        Frechet derivatives of the propagators by the control amplitudes for the
        individual noise traces.

    Methods
    -------
    propagators_noise: List[List[ControlMatrix]],
                 shape: [[] * num_t] * num_noise_traces
        Propagators for the individual noise traces.

    forward_propagators_noise: List[List[ControlMatrix]],
                     shape: [[] * (num_t + 1)] * num_noise_traces
        Cumulation of the propagators for the individual noise traces. They
        describe the forward propagation of the systems state.

    reversed_propagators_noise: List[List[ControlMatrix]],
                          shape: [[] * (num_t + 1)] * num_noise_traces
        Cumulation of propagators in reversed order for the individual noise
        traces.

    frechet_deriv_propagators_noise: List[List[List[ControlMatrix]]],
                            shape: [[[] * num_t] * num_ctrl] * num_noise_traces
        Frechet derivatives of the propagators by the control amplitudes for the
        individual noise traces.

    """
    def __init__(
            self, h_drift: List[q_mat.OperatorMatrix],
            h_ctrl: List[q_mat.OperatorMatrix],
            initial_state: q_mat.OperatorMatrix,
            tau: List[float],
            h_noise: List[q_mat.OperatorMatrix],
            noise_trace_generator:
            Optional[noise.NoiseTraceGenerator],
            ctrl_amps: Optional[np.ndarray] = None,
            calculate_propagator_derivatives: bool = False,
            filter_function_h_n: Union[List[List], np.ndarray, None] = None,
            filter_function_basis: Optional[basis.Basis] = None,
            exponential_method: Optional[str] = None,
            frechet_deriv_approx_method: Optional[str] = None,
            is_skew_hermitian: bool = True,
            transfer_function: Optional[TransferFunction] = None,
            amplitude_function: Optional[AmplitudeFunction] = None,
            noise_amplitude_function: Optional[Callable[
                [np.ndarray, np.ndarray, np.ndarray,
                 np.ndarray], np.ndarray]] = None
    ):

        super().__init__(
            h_drift=h_drift, h_ctrl=h_ctrl, initial_state=initial_state,
            tau=tau, ctrl_amps=ctrl_amps,
            filter_function_h_n=filter_function_h_n,
            filter_function_basis=filter_function_basis,
            exponential_method=exponential_method,
            calculate_propagator_derivatives=calculate_propagator_derivatives,
            frechet_deriv_approx_method=frechet_deriv_approx_method,
            is_skew_hermitian=is_skew_hermitian,
            transfer_function=transfer_function,
            amplitude_function=amplitude_function)

        self.h_noise = h_noise
        self.noise_trace_generator = noise_trace_generator
        self.noise_amplitude_function = noise_amplitude_function

        self._dyn_gen_noise = None
        self._prop_noise = None
        self._derivative_prop_noise = None
        self._fwd_prop_noise = None
        self._reversed_prop_noise = None

    def set_ctrl_amps(self, x: np.ndarray) -> None:
        """See base class. """
        if not np.array_equal(self._opt_pars, x):
            self._dyn_gen_noise = None
            self._prop_noise = None
            self._derivative_prop_noise = None
            self._fwd_prop_noise = None
            self._reversed_prop_noise = None
        super().set_ctrl_amps(x)

    @property
    def propagators_noise(self) -> List[List[q_mat.OperatorMatrix]]:
        """
        Returns the propagators of the system for each noise trace and
        calculates them if necessary.

        Returns
        -------
        propagators_noise: List[List[ControlMatrix]],
                           shape: [[] * num_t] * num_noise_traces
            Propagators of the system for each noise trace.

        """
        if self._prop_noise is None:
            self._compute_propagation()
        return self._prop_noise

    @property
    def forward_propagators_noise(self) -> List[List[q_mat.OperatorMatrix]]:
        """
        Returns the forward propagation of the initial state for every time
        slice and every noise trace and calculate it if necessary. If the
        initial state is the identity matrix, then the cumulative propagators
        are given. The element forward_propagators[k][i] propagates a state by
        the first i time steps under the kth noise trace, if the initial state
        is the identity matrix.

        Returns
        -------
        forward_propagation:List[List[ControlMatrix]],
                     shape: [[] * (num_t + 1)] * num_noise_traces
            Propagation of the initial state of the system. fwd[0] gives the
            initial state itself.

        """
        if self._fwd_prop_noise is None:
            self._compute_forward_propagation()
        return self._fwd_prop_noise

    @property
    def frechet_deriv_propagators_noise(self) \
            -> List[List[List[q_mat.OperatorMatrix]]]:
        """
        Returns the frechet derivatives of the propagators with respect to the
        control amplitudes for each noise trace.

        Returns
        -------
        derivative_prop_noise: List[List[List[ControlMatrix]]],
                            shape: [[[] * num_t] * num_ctrl] * num_noise_traces
            Frechet derivatives of the propagators by the control amplitudes.

        """
        if self._derivative_prop_noise is None:
            self._compute_propagation_derivatives()
        return self._derivative_prop_noise

    @property
    def reversed_propagators_noise(self) -> List[List[q_mat.OperatorMatrix]]:
        """
        Returns the reversed propagation of the initial state for every noise
        trace and calculate it if necessary. If the initial state is the
        identity matrix, then the reversed cumulative propagators are given. The
        element forward_propagators[k][i] propagates a state by the first i time
        steps under the kth noise trace, if the initial state is the identity
        matrix.

        Returns
        -------
        reversed_propagation_noise: List[List[ControlMatrix]],
                                    shape: [[] * (num_t + 1)] * num_noise_traces
            Propagation of the initial state of the system. reversed[k][0] gives
            the initial state itself.

        """
        if self._reversed_prop_noise is None:
            self._compute_reversed_propagation()
        return self._reversed_prop_noise

    def _compute_dyn_gen_noise(self) -> List[List[q_mat.OperatorMatrix]]:
        """
        Computes the dynamics generators for the perturbed and unperturbed
        Schroedinger equation.

        Returns
        -------
        dyn_gen_noise: List[List[q_mat.ControlMatrix]],
                       shape: [[] * num_t] * num_noise_traces
            Dynamics generators for each noise trace.

        """
        # compute the generators of the unperturbed dynamics
        self._dyn_gen = super()._compute_dyn_gen()

        # compute the generators for the noise traces.
        n_noise_traces = self.noise_trace_generator.n_traces

        noise_samples = self.noise_trace_generator.noise_samples
        # we transpose, so we iterate over the time last
        noise_samples = np.transpose(noise_samples, (2, 1, 0))

        if self.noise_amplitude_function:
            noise_samples = self.noise_amplitude_function(
                noise_samples=noise_samples,
                optimization_parameters=self._opt_pars,
                transferred_parameters=self.transferred_parameters,
                control_amplitudes=self._ctrl_amps
            )

        self._dyn_gen_noise = [[dyn_gen.copy() for dyn_gen in self._dyn_gen]
                               for _ in range(n_noise_traces)]

        for t, sample_stack in enumerate(noise_samples):
            for n_trace, trace in enumerate(sample_stack):
                for operator_sample, operator in zip(trace, self.h_noise):
                    self._dyn_gen_noise[n_trace][t] += \
                        (-1j * operator_sample) * operator
        return self._dyn_gen_noise

    def _compute_propagation(
            self, calculate_propagator_derivatives: Optional[bool] = None) \
            -> None:
        """
        Computes the propagators for the perturbed Schroedinger equation and the
        derivatives on demand.

        Parameters
        ----------
        calculate_propagator_derivatives: bool
            Calculate the derivatives of the propagators with respect to the
            control amplitudes if true.

        """
        # call the parent method for the noiseless propagators
        super()._compute_propagation(
            calculate_propagator_derivatives=calculate_propagator_derivatives)

        if self._dyn_gen_noise is None:
            self._dyn_gen_noise = self._compute_dyn_gen_noise()

        n_noise_traces = self.noise_trace_generator.n_traces
        num_t = len(self.tau)
        num_ctrl = len(self.h_ctrl)

        self._prop_noise = [[None for _ in range(num_t)]
                            for _2 in range(n_noise_traces)]

        if calculate_propagator_derivatives is None:
            calculate_propagator_derivatives = \
                self.calculate_propagator_derivatives

        if calculate_propagator_derivatives:
            self._derivative_prop_noise = \
                [[[None for _ in range(num_t)]
                  for _2 in range(num_ctrl)]
                 for _3 in range(n_noise_traces)]
            derivative_directions = self._compute_derivative_directions()

            for k in range(n_noise_traces):
                for t in range(num_t):
                    for ctrl in range(len(self.h_ctrl)):
                        self._prop_noise[k][t], \
                            self._derivative_prop_noise[k][ctrl][t] \
                            = self._dyn_gen_noise[k][t].dexp(
                            derivative_directions[t][ctrl],
                            self.tau[t],
                            compute_expm=True,
                            method=self.exponential_method,
                            is_skew_hermitian=self._is_skew_hermitian)
        else:
            for k in range(n_noise_traces):
                for t in range(num_t):
                    self._prop_noise[k][t] = self._dyn_gen_noise[k][t].exp(
                        tau=self.tau[t], method=self.exponential_method,
                        is_skew_hermitian=self._is_skew_hermitian)

    def _compute_forward_propagation(self) -> None:
        """Computes the forward propagators. """
        super()._compute_forward_propagation()
        if self._prop_noise is None:
            self._compute_propagation()

        self._fwd_prop_noise = [
            [self.initial_state.copy(), ]
            for _ in range(self.noise_trace_generator.n_traces)]

        for fwd_per_trace, prop_per_trace in zip(self._fwd_prop_noise,
                                                 self._prop_noise):
            for prop in prop_per_trace:
                fwd_per_trace.append(prop * fwd_per_trace[-1])

    def _compute_reversed_propagation(self) -> None:
        """Compute the reversed propagation. For the perturbed and unperturbed
        Schroedinger equation. """
        super()._compute_reversed_propagation()
        if self._prop_noise is None:
            self._compute_propagation()

        self._reversed_prop_noise = [
            [self._prop[0].identity_like(), ]
            for _ in range(self.noise_trace_generator.n_traces)]

        for rev_per_trace, prop_per_trace in zip(self._reversed_prop_noise,
                                                 self._prop_noise):
            for prop in prop_per_trace[::-1]:
                rev_per_trace.append(rev_per_trace[-1] * prop)

    def _compute_propagation_derivatives(self) -> None:
        """
        Computes the frechet derivatives of the propagators.

        The derivatives are not returned but cached. Since the function is only
        called when no derivatives are cached, the approximation is prioritised.
        """
        if not self.frechet_deriv_approx_method:
            self._compute_propagation(calculate_propagator_derivatives=True)
        elif self.frechet_deriv_approx_method == 'grape':
            super()._compute_propagation_derivatives()

            if self._prop_noise is None:
                self._compute_propagation(
                    calculate_propagator_derivatives=False)

            n_noise_traces = self.noise_trace_generator.n_traces
            num_t = len(self.tau)
            num_ctrl = len(self.h_ctrl)

            self._derivative_prop_noise = [
                [[None for _ in range(num_t)]
                 for _2 in range(num_ctrl)]
                for _3 in range(n_noise_traces)]

            derivative_directions = self._compute_derivative_directions()

            for k in range(n_noise_traces):
                for t in range(len(self.tau)):
                    for ctrl in range(num_ctrl):
                        self._derivative_prop_noise[k][ctrl][t] = \
                            self.tau[t] * derivative_directions[t][ctrl] \
                            * self._prop_noise[k][t]
        else:
            raise ValueError('Unknown gradient derivative approximation method:'
                             + str(self.frechet_deriv_approx_method))


class SchroedingerSMCControlNoise(SchroedingerSMonteCarlo):
    """
    This time slot computer solves the Schroedinger equation explicitly for
    concrete control noise realizations. This time slot computer assumes,
    that the noise is sampled on the time scale of the already transferred
    optimization parameters. The control Hamiltionians are also used as noise
    Hamiltionians and the noise amplitude function adds the noise samples to
    the unperturbed transferred optimization parameters and applies the
    amplitude function of the control amplitudes.

    """
    def __init__(
            self,
            h_drift: List[q_mat.OperatorMatrix],
            h_ctrl: List[q_mat.OperatorMatrix],
            initial_state: q_mat.OperatorMatrix,
            tau: List[float],
            noise_trace_generator:
            Optional[noise.NoiseTraceGenerator],
            ctrl_amps: Optional[np.ndarray] = None,
            calculate_propagator_derivatives: bool = False,
            filter_function_h_n: Union[List[List], np.ndarray, None] = None,
            filter_function_basis: Optional[basis.Basis] = None,
            exponential_method: Optional[str] = None,
            frechet_deriv_approx_method: Optional[str] = None,
            is_skew_hermitian: bool = True,
            transfer_function: Optional[TransferFunction] = None,
            amplitude_function: Optional[AmplitudeFunction] = None):

        def noise_amplitude_function(noise_samples: np.ndarray,
                                     transferred_parameters: np.ndarray,
                                     control_amplitudes: np.ndarray,
                                     **_):
            noise_amplitudes = np.zeros_like(noise_samples)
            for trace_num in range(noise_samples.shape[1]):
                noise_amplitudes[:, trace_num, :] = self.amplitude_function(
                    transferred_parameters + noise_samples[:, trace_num, :]) \
                    - control_amplitudes
            return noise_amplitudes

        super().__init__(
            h_drift=h_drift,
            h_ctrl=h_ctrl,
            initial_state=initial_state,
            tau=tau,
            h_noise=h_ctrl,
            noise_trace_generator=noise_trace_generator,
            ctrl_amps=ctrl_amps,
            calculate_propagator_derivatives=calculate_propagator_derivatives,
            filter_function_h_n=filter_function_h_n,
            filter_function_basis=filter_function_basis,
            exponential_method=exponential_method,
            frechet_deriv_approx_method=frechet_deriv_approx_method,
            is_skew_hermitian=is_skew_hermitian,
            transfer_function=transfer_function,
            amplitude_function=amplitude_function,
            noise_amplitude_function=noise_amplitude_function
        )


class LindbladSolver(SchroedingerSolver):
    """
    Solves a master equation for an open quantum system in the Markov
    approximation using the Lindblad super operator formalism.

    The master equation to be solved is

    d \rho / dt = i [\rho, H] + \sum_k (L_k \rho L_k^\dag
                  - .5 L_k^\dag L_k \rho - .5 \rho L_k^\dag L_k)

    with the Lindblad operators L_k. The solution is calculated as

    \rho(t) = exp[(-i \mathcal{H} + \mathcal{G})t] \rho(0)

    with the dissipative super operator

    \mathcal{G} = \sum_k D(L_k)

    D(L) = L^\ast \otimes L - .5 I \otimes (L^\dag L)
           - .5 (L^T L^\ast) \otimes I

    The dissipation super operator can be given in three different ways.

    1. A nested list of dissipation super operators D(L_k) as control
    matrices.
    2. A nested list of Lindblad operators L as control matrices.
    3. A function handle receiving the control amplitudes as sole argument and
    returning a dissipation super operator as list of control matrices.

    Optionally a prefactor function can be given for 1. and 2. This function
    receives the control parameters and returns an array of the shape
    num_t x num_l where num_t is the number of time steps in the control and
    num_l is the number of Lindblad operators or dissipation super operators.

    If multiple construction arguments are given, the implementation prioritises
    the function (3.) over the Lindblad operators (2.) over the dissipation
    super operator (1.).

    Parameters
    ----------
    initial_diss_super_op: List[ControlMatrix], len: num_l
        Initial dissipation super operator; num_l is the number of
        Lindbladians. Set if you want to use (1.) (See documentation above!).
        The control matrices are expected to be of shape (dim, dim) where dim is
        the dimension of the system.

    lindblad_operators: List[ControlMatrix], len: num_l
        Lindblad operators; num_l is the number of Lindbladians. Set if you want
        to use (2.) (See documentation above!). The Lindblad operators are
        assumend to be of shape (dim, dim) where dim is the dimension of the
        system.

    prefactor_function: Callable[[np.ndarray], np.ndarray]
        Receives the control amplitudes u (as numpy array of shape
        (num_t, num_ctrl)) and returns prefactors as numpy array
        of shape (num_t, num_l). The prefactors a_k are used as weights in the
        sum of the total dissipation operator.
            \mathcal{G} = \sum_k a_k * D(L_k)
        If the Lindblad operator is for example given by a complex number b_k
        times a constant (in time) matrix C_k.
            L_k = b_k * C_k
        Then the prefactor is the squared absolute value of this number:
            a_k = |b_k|^2
        Set if you want to use method (1.) or (2.). (See class documentation.)

    prefactor_derivative_function: Callable[[np.ndarray], np.ndarray]
        Receives the control amplitudes u (as numpy array of shape
        (num_t, num_ctrl)) and returns the derivatives of the
        prefactors as numpy array of shape (num_t, num_ctrl, num_l). The
        derivatives d_k are used as weights in the sum of the derivative of the
        total dissipation operator.
            d \mathcal{G} / d u_k = \sum_k d_k * D(L_k)
        If the Lindblad operator is for example given by a complex number b_k
        times a constant (in time) matrix C_k. And this number depends on the
        control amplitudes u_k
            L_k = b_k (u_k) * C_k
        Then the derivative of the prefactor is the derivative of the squared
        absolute value of this number:
            d_k = d |b_k|^2 / d u_k
        Set if you want to use method (1.) or (2.). (See class documentation.)

    super_operator_function: Callable[[np.ndarray], List[ControlMatrix]]
        Receives the control amlitudes u (as numpy array of shape
        (num_t, num_ctrl)) and returns the total dissipation
        operators as list of length num_t. Set if you want to use method (3.).
        (See class documentation.)

    super_operator_derivative_function: Callable[[np.ndarray],
                                                 List[List[ControlMatrix]]]
        Receives the control amlitudes u (as numpy array of shape
        (num_t, num_ctrl)) and returns the derivatives of the total dissipation
        operators as nested list of shape [[] * num_ctrl] * num_t. Set if you
        want to use method (3.). (See class documentation.)

    is_skew_hermitian: bool
        If True, then the total dynamics generator is assumed to be skew
        hermitian.

    Attributes
    ----------
    _diss_sup_op: List[ControlMatrix], len: num_t
        Total dissipaton super operator.

    _diss_sup_op_deriv: List[List[ControlMatrix]],
                        shape: [[] * num_ctrl] * num_t
        Derivative of the total dissipation operator with respect to the
        control amplitudes.

    _initial_diss_super_op: List[ControlMatrix], len: num_l
        Initial dissipation super operator; num_l is the number of
        Lindbladians.

    _lindblad_operatorsList[ControlMatrix], len: num_l
        Lindblad operators; num_l is the number of Lindbladians.

    _prefactor_function: Callable[[np.ndarray], np.ndarray]
        Receives the control amplitudes u (as numpy array of shape
        (num_t, num_ctrl)) and returns prefactors as numpy array
        of shape (num_t, num_l). The prefactors a_k are used as weights in the
        sum of the total dissipation operator.
            \mathcal{G} = \sum_k a_k * D(L_k)
        If the Lindblad operator is for example given by a complex number b_k
        times a constant (in time) matrix C_k.
            L_k = b_k * C_k
        Then the prefactor is the squared absolute value of this number:
            a_k = |b_k|^2
        Set if you want to use method (1.) or (2.). (See class documentation.)

    _prefactor_deriv_function: Callable[[np.ndarray], np.ndarray]
        Receives the control amplitudes u (as numpy array of shape
        (num_t, num_ctrl)) and returns the derivatives of the
        prefactors as numpy array of shape (num_t, num_ctrl, num_l). The
        derivatives d_k are used as weights in the sum of the derivative of the
        total dissipation operator.
            d \mathcal{G} / d u_k = \sum_k d_k * D(L_k)
        If the Lindblad operator is for example given by a complex number b_k
        times a constant (in time) matrix C_k. And this number depends on the
        control amplitudes u_k
            L_k = b_k (u_k) * C_k
        Then the derivative of the prefactor is the derivative of the squared
        absolute value of this number:
            d_k = d |b_k|^2 / d u_k

    _sup_op_func: Callable[[np.ndarray], List[ControlMatrix]]
        Receives the control amlitudes u (as numpy array of shape
        (num_t, num_ctrl)) and returns the total dissipation
        operators as list of length num_t.

    _sup_op_deriv_func: Callable[[np.ndarray], List[List[ControlMatrix]]]
        Receives the control amlitudes u (as numpy array of shape
        (num_t, num_ctrl)) and returns the derivatives of the total dissipation
        operators as nested list of shape [[] * num_ctrl] * num_t.

    Methods
    -------
    _parse_dissipative_super_operator: None

    _calc_diss_sup_op: List[ControlMatrix]
        Calculates the total dissipation super operator.

    _calc_diss_sup_op_deriv: Optional[List[List[ControlMatrix]]]
        Calculates the derivatives of the total dissipation super operators
        with respect to the control amplitudes.

    Todo:
        * Write parser
    """

    def __init__(
            self,
            h_drift: List[q_mat.OperatorMatrix],
            h_ctrl: List[q_mat.OperatorMatrix],
            initial_state: q_mat.OperatorMatrix,
            tau: List[float],
            ctrl_amps: Optional[np.ndarray] = None,
            calculate_unitary_derivatives: bool = False,
            filter_function_h_n:
            Union[List[List], np.ndarray, None] = None,
            filter_function_basis: Optional[basis.Basis] = None,
            exponential_method: Optional[str] = None,
            frechet_deriv_approx_method: Optional[str] = None,
            initial_diss_super_op: List[q_mat.OperatorMatrix] = None,
            lindblad_operators: List[q_mat.OperatorMatrix] = None,
            prefactor_function: Callable[[np.ndarray], np.ndarray] = None,
            prefactor_derivative_function:
            Callable[[np.ndarray], np.ndarray] = None,
            super_operator_function:
            Callable[[np.ndarray], List[q_mat.OperatorMatrix]] = None,
            super_operator_derivative_function:
            Callable[[np.ndarray], List[List[q_mat.OperatorMatrix]]] = None,
            is_skew_hermitian: bool = False,
            transfer_function: Optional[TransferFunction] = None,
            amplitude_function: Optional[AmplitudeFunction] = None) \
            -> None:
        super().__init__(
            h_drift=h_drift, h_ctrl=h_ctrl, initial_state=initial_state,
            tau=tau, ctrl_amps=ctrl_amps,
            calculate_propagator_derivatives=calculate_unitary_derivatives,
            filter_function_h_n=filter_function_h_n,
            filter_function_basis=filter_function_basis,
            exponential_method=exponential_method,
            frechet_deriv_approx_method=frechet_deriv_approx_method,
            is_skew_hermitian=is_skew_hermitian,
            transfer_function=transfer_function,
            amplitude_function=amplitude_function)

        self._diss_sup_op = None
        self._diss_sup_op_deriv = None

        # we do not throw away any operators or functions, just in case
        self._initial_diss_super_op = initial_diss_super_op
        self._lindblad_operators = lindblad_operators
        self._prefactor_function = prefactor_function
        self._prefactor_deriv_function = prefactor_derivative_function
        self._sup_op_func = super_operator_function
        self._sup_op_deriv_func = super_operator_derivative_function
        self._is_hermitian = is_skew_hermitian

    def set_ctrl_amps(self, x: np.ndarray) -> None:
        """See base class. """
        if not np.array_equal(self._opt_pars, x):
            super().set_ctrl_amps(x)
            if self._prefactor_function is not None \
                    or self._sup_op_func is not None:
                self._diss_sup_op = None
                self._diss_sup_op_deriv = None

    def _calc_diss_sup_op(self) -> List[q_mat.OperatorMatrix]:
        """
        Calculates the dissipative super operator as described in the class
        doc string.

        Returns
        -------
        diss_sup_op: List[ControlMatrix], len: num_l
            Dissipation super operator; Where num_l is the number of Lindblad
            terms.

        """
        if self._sup_op_func is None:
            # use Lindblad operators
            if self._lindblad_operators is None:
                # use dissipation_sup_op
                const_diss_sup_op = self._initial_diss_super_op
            else:
                # Calculate the time constant dissipation super operators
                # without time dependence
                const_diss_sup_op = []
                identity = self._lindblad_operators[0].identity_like()

                for lindblad in self._lindblad_operators:
                    const_diss_sup_op.append(
                        (lindblad.conj(copy_=True)).kron(lindblad))
                    const_diss_sup_op[-1] -= .5 * identity.kron(
                        lindblad.dag(copy_=True) * lindblad)
                    const_diss_sup_op[-1] -= .5 * (
                        lindblad.transpose(copy_=True)
                        * lindblad.conj(copy_=True)).kron(identity)

            # Add the time dependence
            if self._prefactor_function is not None:
                self._diss_sup_op = []
                prefactors = self._prefactor_function(
                    copy.deepcopy(self._ctrl_amps))
                for factor_at_time_t in prefactors:
                    self._diss_sup_op.append(
                        const_diss_sup_op[0] * factor_at_time_t[0])
                    for sup_op, factor \
                            in zip(const_diss_sup_op[1:], factor_at_time_t[1:]):
                        self._diss_sup_op[-1] += sup_op * factor
            else:
                self._diss_sup_op = [const_diss_sup_op[0], ]
                for sup_op in const_diss_sup_op[1:]:
                    self._diss_sup_op[0] += sup_op
                self._diss_sup_op *= len(self.tau)
        else:
            self._diss_sup_op = self._sup_op_func(
                self._ctrl_amps,
                self.transferred_parameters)
        return self._diss_sup_op

    def _calc_diss_sup_op_deriv(self) \
            -> Optional[List[List[q_mat.OperatorMatrix]]]:
        """
        Calculates the derivatives of the dissipation super operator with
        respect to the control amplitudes.

        If the dissipation super operator is given as constant (1.) or as
        lindblad operators (2.) they are assumed not to depend on the control
        parameters and only the derivative of the prefactor is to be taken into
        account. In order to do so, a function handle containing the derivatives
        must be given. This function receives the control amplitudes as
        num_t x num_ctrl numpy array and returns the derivatives as
        num_t x num_l x num_ctrl array.

        If the dissipation super operator is given as function handle (3.),
        then the derivatives must also be given as function handle receiving
        the control amplitudes and returning a nested list of super operators as
        control matrices.

        If the requested derivative functions are not provided (None), then
        the dissipation super operator is considered constant in the control
        amplitudes and the function returns None.

        Returns
        -------
        diss_sup_op_deriv: Optional[List[List[q_mat.ControlMatrix]]],
                           shape [[] * num_ctrl] * num_t
            The derivatives of the dissipation super operator with respect to
            the control variables.

        """

        if self._sup_op_deriv_func is not None:
            self._diss_sup_op_deriv = \
                self._sup_op_deriv_func(
                    self._ctrl_amps,
                    self.transferred_parameters)
            return self._diss_sup_op_deriv

        elif self._prefactor_deriv_function is not None:
            if self._lindblad_operators is None:
                # use dissipation_sup_op
                const_diss_sup_op = self._initial_diss_super_op
            else:
                # Calculate the time constant dissipation super operators
                # without time dependence
                const_diss_sup_op = []
                identity = self._lindblad_operators[0].identity_like()

                for lindblad in self._lindblad_operators:
                    const_diss_sup_op.append(
                        (lindblad.conj(copy_=True)).kron(lindblad))
                    const_diss_sup_op[-1] -= .5 * identity.kron(
                        lindblad * lindblad.dag(copy_=True))
                    const_diss_sup_op[-1] -= .5 * (
                        lindblad.transpose(copy_=True)
                        * lindblad.conj(copy_=True)).kron(identity)

            prefactor_derivatives = \
                self._prefactor_deriv_function(self._ctrl_amps)
            # prefactor_derivatives: shape (num_t, num_ctrl, num_l)
            diss_sup_op_deriv = []
            for factor_per_ctrl_lind in prefactor_derivatives:
                diss_sup_op_deriv.append([])
                for factor_per_lind in factor_per_ctrl_lind:
                    diss_sup_op_deriv[-1].append(
                        const_diss_sup_op[0] * factor_per_lind[0])
                    for diss_sup_op, factor in zip(
                            const_diss_sup_op, factor_per_lind):
                        diss_sup_op_deriv[-1][-1] += diss_sup_op * factor
            self._diss_sup_op_deriv = diss_sup_op_deriv
            return diss_sup_op_deriv
        else:
            return None

    def _compute_derivative_directions(self) -> List[List[q_mat.OperatorMatrix]]:
        """
        Computes the derivative directions of the total dynamics generator.

        Returns
        -------
        deriv_directions: List[List[q_mat.ControlMatrix]],
                          shape: [[] * num_ctrl] * num_t
            Derivative directions given by
            -1j * (I \otimes H_k - H_k \otimes I) + d \mathcal{G} / d u_k

        """
        # derivative of the coherent part
        identity_times_i = self.h_ctrl[0].identity_like()
        identity_times_i *= -1j
        h_ctrl_sup_op = []
        for ctrl_op in self.h_ctrl:
            h_ctrl_sup_op.append(identity_times_i.kron(ctrl_op))
            h_ctrl_sup_op[-1] -= ctrl_op.kron(identity_times_i)

        # add derivative of the dissipation part
        if self._diss_sup_op_deriv is None:
            self._diss_sup_op_deriv = self._calc_diss_sup_op_deriv()
        if self._diss_sup_op_deriv is not None:
            dh_by_ctrl = []
            for diss_sup_op_deriv_at_t in self._diss_sup_op_deriv:
                dh_by_ctrl.append([])
                for diss_sup_op_deriv, ctrl_sup_op \
                        in zip(diss_sup_op_deriv_at_t, h_ctrl_sup_op):
                    dh_by_ctrl[-1].append(diss_sup_op_deriv + ctrl_sup_op)
        else:
            dh_by_ctrl = [h_ctrl_sup_op, ] * len(self.tau)

        return dh_by_ctrl

    def _parse_dissipative_super_operator(self) -> None:
        """
        check the dissipative super operator for dimensional consistency
        (maybe even physical properties)
        :return:
        """
        pass

    def _compute_dyn_gen(self) -> List[q_mat.OperatorMatrix]:
        """
        Computes the dynamics generator for the Lindblad master equation.

        The Hamiltonian is translated into the master equation formalism as

        \mathcal{H} = I \otimes H - H^\ast \otimes I

        Then the dissipation super operator is added.

        Returns
        -------
        dyn_gen: List[ControlMatrix], len: num_t
            Dynamics generators for the master equation.

        Raises
        ------
        ValueError:
            The computation is only defined for the use of dense control
            atrices.

        """
        self._dyn_gen = super()._compute_dyn_gen()

        if self._diss_sup_op is None:
            self._diss_sup_op = self._calc_diss_sup_op()

        identiy_operator = self._dyn_gen[0].identity_like()
        sup_op_dyn_gen = []

        assert(len(self._dyn_gen) == len(self._diss_sup_op))

        for dyn_gen, diss_sup_op in zip(self._dyn_gen, self._diss_sup_op):
            sup_op_dyn_gen.append(identiy_operator.kron(dyn_gen))
            # the cancelling minus sign accounts for the -i factor, which is
            # also conjugated (included in the dyn gen)
            sup_op_dyn_gen[-1] += dyn_gen.conj(copy_=True).kron(
                identiy_operator)
            sup_op_dyn_gen[-1] += diss_sup_op

        self._dyn_gen = sup_op_dyn_gen
        return sup_op_dyn_gen


class LindbladSControlNoise(LindbladSolver):
    """
    Special case of the Lindblad master equation. It considers white noise on
    the control parameters. The same functionality should be implementable
    with the parent class, but less convenient.
    """

    @needs_refactoring
    def __init__(self, h_drift, h_ctrl, initial_state, tau,
                 ctrl_amps, transfer_function=None,
                 calculate_unitary_derivatives=True, filter_function_h_n=None,
                 exponential_method=None, lindblad_operators=None,
                 constant_lindblad_operators=False, noise_psd=1):
        super().__init__(
            h_drift=h_drift, h_ctrl=h_ctrl, initial_state=initial_state,
            tau=tau, ctrl_amps=ctrl_amps,
            calculate_unitary_derivatives=calculate_unitary_derivatives,
            filter_function_h_n=filter_function_h_n,
            exponential_method=exponential_method)

        if lindblad_operators is None:
            self.lindblad_super_operator = None
        else:
            d = lindblad_operators[0].shape[0]
            self.lindblad_super_operator = np.zeros(
                (len(lindblad_operators), d**2, d**2))
            for i, l in enumerate(lindblad_operators):
                self.lindblad_super_operator[i, :, :] += np.kron(np.conj(l), l)
                self.lindblad_super_operator[i, :, :] += -.5 * np.kron(
                    np.eye(d), l.T.conj() @ l)
                self.lindblad_super_operator[i, :, :] += -.5 * np.kron(
                    l.T @ l.conj(), np.eye(d))

        self.transfer_function = transfer_function
        # if no transfer function is given it might be consider to be identity
        # its not necessarily required

        self.constant_lindblad_operators = constant_lindblad_operators
        self.noise_psd = noise_psd
        self.incoherent_dyn_gen = None

    def _compute_propagation(self):
        """

        """
        # Compute and cache all dyn_gen (basically the total hamiltonian)
        self._dyn_gen = copy.deepcopy(self.h_drift)
        self._dyn_gen += np.sum(self._ctrl_amps * self.h_ctrl, axis=1)

        # initialize the attributes
        self._prop = [None] * self.num_t
        self._dU = np.ndarray(shape=(self.num_t, self.num_ctrl),
                              dtype=matrix.DenseOperator)
        self._fwd = [self.initial_state]

        # super operator calculation
        # this is the special case for charge noise on the control parameters
        # the required filter function contains
        if not self.constant_lindblad_operators or \
                self.incoherent_dyn_gen is None:
            transfer_matrix = self.transfer_function.T
            self.incoherent_dyn_gen = np.einsum('ijk,klm,k->ilm',
                                                transfer_matrix,
                                                self.lindblad_super_operator,
                                                self.noise_psd)
        dim = self._dyn_gen[0].shape[0]
        for i, gen in enumerate(self._dyn_gen):
            gen = -1j * np.kron(
                np.eye(dim), gen.data) - np.kron(gen.data, np.eye(dim))
            gen += self.incoherent_dyn_gen[i, :, :]
            gen = matrix.DenseOperator(gen)

        # calculation of the propagators
        for t in range(len(self.num_t)):
            if self.calculate_propagator_derivatives:
                for ctrl in range(self.num_ctrl):
                    direction = np.kron(
                        np.eye(dim), self.h_ctrl[t][ctrl]) - np.kron(
                        self.h_ctrl[t][ctrl], np.eye(dim))
                    self._prop[t], self._dU[t, ctrl] = self._dyn_gen[t].dexp(
                        direction=direction, tau=self.tau[t],
                        compute_expm=True, method=self.exponential_method)

            else:
                self._prop[t] = self._dyn_gen[t].exp(
                    tau=self.tau[t], method=self.exponential_method)

            self._fwd.append(self._prop[t] * self._fwd[t])

        self.prop_calculated = True