# -*- coding: utf-8 -*-
# =============================================================================
#     qopt
#     Copyright (C) 2020 Julian Teske, Forschungszentrum Juelich
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program. If not, see <http://www.gnu.org/licenses/>.
#
#     Contact email: j.teske@fz-juelich.de
# =============================================================================
""" Implements the algorithms to solve differential equations like
Schroedinger's equation or a master equation.

The `Solver` class is the central piece of the actual simulation. It calculates
propagators from the differential equations describing the quantum dynamics.
The abstract base class inherits among other things an interface to the
`PulseSequence` class of the filter_functions package.

The `Solver` classes can have an amplitude and a transfer function as attribute
and automate their use. The Monte Carlo solvers also hold an instance of a
noise trace generator.

If requested, also derivatives of the propagators by the control amplitudes are
calculated or approximated.

Classes
-------
:class:`Solver`
    Abstract base class of the time slot computers.

:class:`SchroedingerSolver`
    Solver for the the unperturbed Schroedinger equation.

:class:`SchroedingerSMonteCarlo`
    Solver for the Schroedinger equation under the influence of noise.

:class:`SchroedingerSMCControlNoise`
    Solver for the Schroedinger equation under the influence of noise affecting
    the control terms.

:class:`LindbladSolver`
    Solves the master equation in Lindblad form.

Notes
-----
The implementation was inspired by the optimal control package of QuTiP [1]_
(Quantum Toolbox in Python)

References
----------
.. [1] J. R. Johansson, P. D. Nation, and F. Nori: "QuTiP 2: A Python framework
    for the dynamics of open quantum systems.", Comp. Phys. Comm. 184, 1234
    (2013) [DOI: 10.1016/j.cpc.2012.11.019].

"""

import numpy as np
import copy
from typing import Optional, List, Callable, Union
from abc import ABC, abstractmethod
from multiprocessing import Pool

from filter_functions import pulse_sequence, basis, numeric
from filter_functions import plotting as ff_plotting

from qopt import noise, matrix, matrix as q_mat
from qopt.transfer_function import TransferFunction, IdentityTF
from qopt.amplitude_functions import AmplitudeFunction, IdentityAmpFunc
from qopt.util import needs_refactoring


class Solver(ABC):
    r"""
    Abstract base class for Solvers.

    Parameters
    ----------
    h_ctrl: List[ControlMatrix], len  num_ctrl
        Control operators in the Hamiltonian as nested list of
        shape n_t, num_ctrl.

    h_drift: List[ControlMatrix], len num_t or 1
        Drift operators in the Hamiltonian. You can either give a single element
        or one for each transferred time step.

    initial_state : ControlMatrix
        Initial state of the system as state vector. Can also be set to the
        identity matrix. Then the forward propagation gives the total
        propagator of the system.

    tau: array of float, shape (num_t, )
        Durations of the time slices.

    opt_pars: np.array, shape (num_y, num_par), optional
        Raw optimization parameters.

    ctrl_amps: np.array, shape (num_t, num_ctrl), optional
        The initial control amplitudes.

    filter_function_h_n: List[List[np.array]] or List[List[Qobj]] or callable
        Nested list of noise Operators. Used in the filter function
        formalism. _filter_function_h_n should look something like this:

        >>> H = [[n_oper1, n_coeff1, n_oper_identifier1],
        >>>      [n_oper2, n_coeff2, n_oper_identifier2], ...]

        The operators may be given either as NumPy arrays or QuTiP Qobjs
        and each coefficient array should have the same number of elements
        as *dt*, and should be given in units of :math:`\hbar`. Alternatively,
        the argument can be a callable. This should have the signature of three
        input arguments, which are (Optimization parameters, transferred
        parameters, control amplitudes). The callable should return an nested
        list of the form given above.

    filter_function_basis: Basis, shape (d**2, d, d), optional
        The operator basis in which to calculate. If a Generalized Gell-Mann
        basis (see :meth:`~basis.Basis.ggm`) is chosen, some calculations will
        be faster for large dimensions due to a simpler basis expansion.
        However, when extending the pulse sequence to larger qubit registers,
        cached filter functions cannot be retained since the GGM basis does not
        factor into tensor products. In this case a Pauli basis is preferable.

    filter_function_n_coeffs_deriv: Callable numpy array to numpy array
        This function calculates the derivatives of the noise susceptibility in
        the filter function formalism. It receives the optimization parameters
        as array of shape (num_opt, num_t) and returns the derivatives as array
        of shape (num_noise_op, n_ctrl, num_t).
        The order of the noise operators must correspond to the order specified
        by filter_function_h_n.

    exponential_method: string, optional
        Method used by the ControlMatrix class for the calculation of the
        matrix exponential. The default is 'Frechet'. See also the Docstring of
        the file 'qopt.matrix'.

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
        0 No tests
        1 Some tests
        2 Exhaustive tests, dimension checks

    Attributes
    ----------
    h_ctrl : List[ControlMatrix], len num_ctrl
        Control operators in the Hamiltonian as list of length num_ctrl.

    h_drift : List[ControlMatrix], len num_t
        Drift operators in the Hamiltonian.

    initial_state : ControlMatrix
        Initial state of the system as state vector. Can also be set to the
        identity matrix. Then the forward propagation gives the total
        propagator of the system.

    transferred_time: List[float]
        Durations of the time slices.

    filter_function_h_n: List[List[np.array]] or List[List[Qobj]]
        Nested list of noise Operators. Used in the filter function
        formalism.

    filter_function_basis: Basis
        The filter function pulse sequence will be expressed in this basis.
        See documentation of the filter function package.

    exponential_method: string, optional
        Method used by the ControlMatrix class for the calculation of the
        matrix exponential. The default is 'Frechet'. See also the Docstring of
        the file 'qopt.matrix'.

    transfer_function: TransferFunction
        The transfer function for reshaping the optimization parameters.

    amplitude_function: AmplitudeFunction
        The amplitude function connecting the transferred optimization
        parameters to the control amplitudes.

    _prop: List[ControlMatrix], len num_t
        Propagators of the system.

    _fwd_prop: List[ControlMatrix], len num_t + 1
        Ordered product of the propagators. They describe the forward
        propagation of the systems state.

    _reversed_prop: List[ControlMatrix], len num_t + 1
        Ordered product of propagators in reversed order.

    _derivative_prop: List[List[ControlMatrix]], shape [[] * num_t] * num_ctrl
        Frechet derivatives of the propagators by the control amplitudes.

    Methods
    -------
    propagators: List[ControlMatrix], len num_t
        Returns the propagators of the system.

    forward_propagators: List[ControlMatrix], len num_t + 1
        Returns the forward propagation of the initial state. The element
        forward_propagators[i] propagates a state by the first i time steps, if
        the initial state is the identity matrix.

    frechet_deriv_propagators: List[List[ControlMatrix]],
        shape [[] * num_t] * num_ctrl
        Returns the frechet derivatives of the propagators by the control
        amplitudes.

    reversed_propagators: List[ControlMatrix], len num_t + 1
        Returns the reversed propagation of the initial state. The element
        reversed_propagators[i] propagates a state by the last i time steps, if
        the initial state is the identity matrix.

    _compute_propagation: abstract method
        Computes the propagators.

    _compute_forward_propagation
        Compute the forward propagation of the initial state / system.

    _compute_reversed_propagation
        Compute the reversed propagation of the initial state / system.

    _compute_propagation_derivatives: abstract method
        Compute the derivatives of the propagators by the control amplitudes.

    create_pulse_sequence(new_amps): PulseSequence
        Creates a pulse sequence instance corresponding to the current control
        amplitudes.

    `Todo`
        * Write parser
            * setter for new hamiltonians
            * make hamiltonians private
            * also for the initial state
            * extend constant drift hamiltonian
        * Implement the drift operator with an amplitude. Right now,
            * the operator is already multiplied with the amplitude, which is
            * not coherent with the pulse sequence interface. Alternatively
            * amplitude=1?
        * transferred_time should be taken from the transfer function
        * Use own plotting for the plotting
        * Consequent try catches for the computation of the matrix exponential

    """

    def __init__(
            self,
            h_ctrl: List[q_mat.OperatorMatrix],
            h_drift: List[q_mat.OperatorMatrix],
            tau: np.array,
            initial_state: q_mat.OperatorMatrix = None,
            opt_pars: Optional[np.array] = None,
            ctrl_amps: Optional[np.array] = None,
            filter_function_h_n: Union[
                Callable, List[List], None] = None,
            filter_function_basis: Optional[basis.Basis] = None,
            filter_function_n_coeffs_deriv: Optional[
                Callable[[np.ndarray], np.ndarray]] = None,
            exponential_method: Optional[str] = None,
            is_skew_hermitian: bool = True,
            transfer_function: Optional[TransferFunction] = None,
            amplitude_function: Optional[AmplitudeFunction] = None,
            paranoia_level: int = 2
    ):

        self.h_ctrl = h_ctrl
        self._ctrl_amps = ctrl_amps
        self._opt_pars = opt_pars

        if initial_state is None:
            dim = self.h_ctrl[0].shape[0]
            self.initial_state = type(self.h_ctrl[0])(np.eye(dim))
        else:
            self.initial_state = initial_state

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
            self._filter_function_h_n = []
        else:
            self._filter_function_h_n = filter_function_h_n

        # we store the order of the noise operators. They must coincide with
        # the order in filter_functions_n_coeffs_deriv
        self.filter_function_n_oper_identifiers = []

        self.filter_function_basis = filter_function_basis
        self.filter_function_n_coeffs_deriv = filter_function_n_coeffs_deriv

        self._is_skew_hermitian = is_skew_hermitian

        if transfer_function is None:
            self.transfer_function = IdentityTF(num_ctrls=len(h_ctrl))
        else:
            self.transfer_function = transfer_function

        self.transferred_time = None
        self.set_times(tau=tau)

        if type(h_drift) in [matrix.DenseOperator, matrix.SparseOperator]:
            self.h_drift = [h_drift, ] * self.transfer_function.num_x
        elif len(h_drift) == 1:
            self.h_drift = h_drift * self.transfer_function.num_x
        else:
            self.h_drift = h_drift

        if amplitude_function is None:
            self.amplitude_function = IdentityAmpFunc()
        else:
            self.amplitude_function = amplitude_function

        self.transferred_parameters = None

        self.consistency_checks(paranoia_level=paranoia_level)

    def set_times(self, tau):
        """ Set time values by passing them to the transfer function.

        Parameters
        ----------
        tau: array of float, shape (num_t, )
            Durations of the time slices.

        """
        self.transfer_function.set_times(tau)
        self.transferred_time = self.transfer_function.x_times
        self.reset_cached_propagators()

    def set_optimization_parameters(self, y: np.array) -> None:
        """
        Set the control amplitudes.

        All computation flags are set to false.

        The new control amplitudes u are calculated:
        u: np.array, shape (num_t, num_ctrl)

        Parameters
        ----------
        y: np.array, shape (num_x, num_ctrl)
            Raw optimization parameters.

        """

        if np.array_equal(self._opt_pars, y):
            return
        else:
            self._opt_pars = np.copy(y)

        if self.transfer_function is not None:
            self.transferred_parameters = self.transfer_function(y)
        else:
            self.transferred_parameters = np.copy(y)

        if self.amplitude_function is not None:
            u = self.amplitude_function(
                self.transferred_parameters)
        else:
            u = self.transferred_parameters

        if len(u.shape) != 2:
            raise ValueError('The new control amplitudes must have two '
                             'dimensions! '
                             '(time, control operator)')

        if u.shape[0] != len(self.transferred_time):
            raise ValueError('The new control amplitudes do not have the '
                             'correct number of entries on the time axis!')

        if u.shape[1] != len(self.h_ctrl):
            raise ValueError('The new control amplitudes do not have the '
                             'correnct number of entries on the control axis!')

        self._ctrl_amps = u
        self.reset_cached_propagators()

    def reset_cached_propagators(self):
        """ Resets all cached propagators. """
        self._prop = None
        self._fwd_prop = None
        self._derivative_prop = None
        self._reversed_prop = None
        self.pulse_sequence = None

    def consistency_checks(self, paranoia_level: int):
        """Checks attributes for inner consistency.

        Parameters
        ----------
        paranoia_level: int
            The paranoia_level determines how many checks are conducted.
            0: No tests
            1: Some tests
            2: Exhaustive tests, dimension checks

        """
        if paranoia_level == 0:
            return

        elif paranoia_level >= 1:
            # check whether the hamiltonian is correct for the number of time
            # steps
            if isinstance(self.transferred_time, List):
                self.transferred_time = np.asarray(self.transferred_time)
            if len(self.transferred_time.shape) > 1:
                raise ValueError("Tau must be a one dimensional numpy array or"
                                 "a list.")
            n_time_steps = self.transferred_time.shape[0]

            if len(self.h_drift) == 1:
                self.h_drift = self.h_drift * n_time_steps

            if not (n_time_steps == len(self.h_drift)
                    or len(self.h_drift) == 0):
                raise ValueError(
                    "The drift hamiltonian must have exactly one entry for "
                    "each transferred time step or no entry at all or a single"
                    " entry. Your transferred time has " + str(n_time_steps)
                    + " steps."
                )
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
        propagators: List[ControlMatrix], len num_t
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
        forward_propagation: List[ControlMatrix], len num_t + 1
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
                         shape [[] * num_t] * num_ctrl
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
        identity matrix, then the reversed cumulative propagators are given.
        The element forward_propagators[i] propagates a state by the first i
        time steps, if the initial state is the identity matrix.

        Returns
        -------
        reversed_propagation: List[ControlMatrix], len num_t + 1
            Propagation of the initial state of the system. reversed[0] gives
            the initial state itself.

        """
        if self._reversed_prop is None:
            self._compute_reversed_propagation()
        return self._reversed_prop

    @property
    def filter_function_n_coeffs_deriv_vals(self) -> Optional[np.ndarray]:
        """
        Calculates the derivatives of the noise susceptibilities from the filter
        function formalism.

        Returns
        -------
        n_coeffs_deriv: numpy array of shape (num_noise_op, n_ctrl, num_t)
            Derivatives of the noise susceptibilities by the control amplitudes.

        """
        if self.filter_function_n_coeffs_deriv is None:
            return None
        else:
            try:
                return self.filter_function_n_coeffs_deriv(
                    self._opt_pars, self.transferred_parameters,
                    self._ctrl_amps)
            except TypeError:
                print("Warning, you are used the old interface for the "
                      "filter_functio_h_n. If you choose it as callable,"
                      "it should receive the three arguments "
                      "(optimization parameters, transferred parameters,"
                      "control amplitudes). ")
                return self.filter_function_n_coeffs_deriv(self._ctrl_amps)

    @property
    def create_ff_h_n(self) -> list:
        """Creates the noise hamiltonian of the filter function formalism.

        Returns
        -------
        create_ff_h_n: nested list
            Noise Hamiltonian of the filter function formalism.

        """
        if type(self._filter_function_h_n) == list:
            h_n = self._filter_function_h_n
        else:
            try:
                h_n = self._filter_function_h_n(
                    self._opt_pars, self.transferred_parameters,
                    self._ctrl_amps)
            except TypeError:
                print("Warning, you are used the old interface for the "
                      "filter_functio_h_n. If you choose it as callable,"
                      "it should receive the three arguments "
                      "(optimization parameters, transferred parameters,"
                      "control amplitudes). ")
                h_n = self._filter_function_h_n(self._ctrl_amps)

        if not h_n:
            h_n = []

        # we store the order of the noise operators. They must coincide with
        # the order in filter_functions_n_coeffs_deriv
        self.filter_function_n_oper_identifiers = []
        for noise_term in h_n:
            if not len(noise_term) == 3:
                raise ValueError(
                    "The noise operators for the filter function must be given"
                    "as nested list of the form: \n"
                    "H = [[n_oper1, n_coeff1, n_oper_identifier1], \n"
                    "[n_oper2, n_coeff2, n_oper_identifier2], ...] \n"
                    "but not every element of the list you defined has three"
                    "elements."
                )
            if not type(noise_term[2]) == str:
                raise ValueError(
                    "The identifiers for the noise terms must be given as "
                    "string."
                )
            self.filter_function_n_oper_identifiers.append(noise_term[2])

        return h_n

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
            raise NotImplementedError
            # self._reversed_prop = [matrix.SparseOperator(
            #    np.eye(self._prop[0].shape[0])) * (1 + 0j), ]
        else:
            raise TypeError("The initial state should be either a dense or "
                            "sparse control matrix.")

        for prop in self._prop[::-1]:
            self._reversed_prop.append(self._reversed_prop[-1] * prop)

    @abstractmethod
    def _compute_propagation_derivatives(self) -> None:
        """Compute the derivatives of the propagators by the control
        amplitudes.
        """
        pass

    def _diagonalize_and_propagate_pulse_sequence(self) -> None:
        """Manually set eigendecomposition of the PulseSequence.

        Work around incompatibility of drift Hamiltonian
        representations."""
        ps = self.pulse_sequence
        drift_hamiltonian = np.array([h.data for h in self.h_drift])
        control_hamiltonian = np.einsum('ijk,il->ljk', ps.c_opers, ps.c_coeffs)
        ps.eigvals, ps.eigvecs, ps.propagators = numeric.diagonalize(
            drift_hamiltonian + control_hamiltonian, ps.dt
        )
        ps.total_propagator = ps.propagators[-1]

    def create_pulse_sequence(
            self, new_amps: Optional[np.array] = None,
            ff_basis: Optional[basis.Basis] = None
    ) -> pulse_sequence.PulseSequence:
        """
        Create a pulse sequence of the filter function package written by
        Tobias Hangleiter.

        See the documentation of the filter function package.

        Parameters
        ----------
        new_amps: np.array, shape (num_t, num_ctrl), optional
            New control amplitudes can be set before the pulse sequence is
            initialized.

        ff_basis: Basis
            The pulse sequence will be expanded in this basis. See
            documentation of the filter function package.

        Returns
        -------
        pulse_sequence: filter_functions.pulse_sequence.PulseSequence
            The pulse sequence corresponding to the control model and the
            control amplitudes set.

        """
        if new_amps is not None:
            self.set_optimization_parameters(new_amps)
        else:
            if self._ctrl_amps is None:
                raise ValueError('No optimization parameters set. '
                                 'Please supply new_amps argument')


        if ff_basis is not None:
            basis = ff_basis
        elif self.filter_function_basis is not None:
            basis = self.filter_function_basis
        else:
            basis = None

        # We have to work around different interfaces for the drift
        # operators. Since in qopt the drift can be arbitrary (incl.
        # nonlinear coupling), but in filter_functions the form H =
        # a(t) A is imposed, we don't tell the PulseSequence object
        # about H_drift and set the eigendecomposition after the fact.
        if self.pulse_sequence is None:
            h_c = list(zip(
                self.h_ctrl,
                self._ctrl_amps.T,
                [f'Control{i}' for i in range(len(self.h_ctrl))]
            ))
            self.pulse_sequence = pulse_sequence.PulseSequence(
                h_c, self.create_ff_h_n, self.transferred_time, basis
            )
        else:
            # Clean up the caches and update coefficients
            self.pulse_sequence.cleanup('all')
            self.pulse_sequence.c_coeffs = self._ctrl_amps.T
            # Not the most elegant, but necessary for the current
            # implementation.
            self.pulse_sequence.n_coeffs = pulse_sequence._parse_Hamiltonian(
                self.create_ff_h_n,
                len(self.transferred_time), 'H_n')[2]

            if basis is not None:
                self.pulse_sequence.basis = basis

        self._diagonalize_and_propagate_pulse_sequence()
        return self.pulse_sequence

    def plot_bloch_sphere(
            self, new_amps=None, return_Bloch: bool = False) -> None:
        """
        Uses the pulse sequence to plot the systems evolution on the bloch
        sphere.

        Only available for two dimensional systems.

        Parameters
        ----------
        new_amps: np.array, shape (num_t, num_ctrl), optional
            New control amplitudes can be set before the pulse sequence is
            initialized.

        return_Bloch: bool
            If True, then qutips Bloch object is returned.

        Returns
        -------
        b: Bloch
            Qutips Bloch object. Only returned if return_Bloch is set to True.

        """
        # Already takes care of updating and cleaning the PulseSequence object
        pulse_sequence = self.create_pulse_sequence(new_amps=new_amps)
        return ff_plotting.plot_bloch_vector_evolution(
            pulse_sequence,
            n_samples=500,
            return_Bloch=return_Bloch)


class SchroedingerSolver(Solver):
    """
    This time slot computer solves the unperturbed Schroedinger equation.

    All intermediary propagators are calculated and cached. Takes also input
    parameters of the base class.

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
    _dyn_gen: List[ControlMatrix], len num_t
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
    shape [[] * num_ctrl] * num_t
        Computes the directions of change with respect to the control
        parameters.

    _compute_dyn_gen: List[ControlMatrix], len num_t
        Computes the dynamics generators.

    `Todo`
        * raise a warning if the approximation method although the gradient
            is always calculated.
        * raise a warning if the grape approximation is chosen but its
            requirement of small time steps is not met.

    """

    def __init__(self,
                 h_drift: List[q_mat.OperatorMatrix],
                 h_ctrl: List[q_mat.OperatorMatrix],
                 tau: np.array,
                 initial_state: q_mat.OperatorMatrix = None,
                 ctrl_amps: Optional[np.array] = None,
                 calculate_propagator_derivatives: bool = True,
                 filter_function_h_n: Union[
                     Callable, List[List], None] = None,
                 filter_function_basis: Optional[basis.Basis] = None,
                 filter_function_n_coeffs_deriv: Optional[
                    Callable[[np.ndarray], np.ndarray]] = None,
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
            filter_function_n_coeffs_deriv=filter_function_n_coeffs_deriv,
            exponential_method=exponential_method,
            is_skew_hermitian=is_skew_hermitian,
            transfer_function=transfer_function,
            amplitude_function=amplitude_function
        )
        self.id_text = 'ALL'
        self.cache_text = 'Save'
        self.calculate_propagator_derivatives = \
            calculate_propagator_derivatives
        self.frechet_deriv_approx_method = frechet_deriv_approx_method

        self._dyn_gen = None

    def set_optimization_parameters(self, y: np.array) -> None:
        """See base class. """
        if not np.array_equal(self._opt_pars, y):
            self.reset_cached_propagators()
        super().set_optimization_parameters(y)

    def reset_cached_propagators(self):
        """See base class. """
        self._dyn_gen = None
        super().reset_cached_propagators()

    def _compute_dyn_gen(self) -> List[q_mat.OperatorMatrix]:
        """
        Computes the dynamics generators.

        Returns
        -------
        dyn_gen: List[ControlMatrix], len num_t
            This is basically the total Hamiltonian.

        """
        self._dyn_gen = [-1j * h for h in self.h_drift]
        for ctrl, ctrl_op in enumerate(self.h_ctrl):
            for dyn_gen, ctrl_amp in \
                    zip(self._dyn_gen, self._ctrl_amps[:, ctrl]):
                dyn_gen += -1j * ctrl_amp * ctrl_op
        return self._dyn_gen

    def _compute_derivative_directions(
            self) -> List[List[q_mat.OperatorMatrix]]:
        """
        The directions of the frechet derivatives are the control operators.

        No deep copy is required because the result is not used for in-place
        operations.

        """
        # The list is multiplied (copied by reference) because the elements
        # will not be manipulated in place. (only as copy)
        return [[operator * -1j for operator in self.h_ctrl], ] * len(self.transferred_time)

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
        self._prop = [None for _ in range(len(self.transferred_time))]

        if calculate_propagator_derivatives:
            derivative_directions = self._compute_derivative_directions()
            self._derivative_prop = [
                [None for _ in range(len(self.transferred_time))]
                for _2 in range(len(self.h_ctrl))]
            for t in range(len(self.transferred_time)):
                for ctrl in range(len(self.h_ctrl)):
                    try:
                        self._prop[t], self._derivative_prop[ctrl][t] \
                            = self._dyn_gen[t].dexp(
                            derivative_directions[t][ctrl],
                            self.transferred_time[t],
                            compute_expm=True, method=self.exponential_method,
                            is_skew_hermitian=self._is_skew_hermitian)
                    except ValueError:
                        raise ValueError('The computation has failed with '
                                         'a value error. Try another '
                                         'exponentiation method.')
        else:
            for t in range(len(self.transferred_time)):
                self._prop[t] = self._dyn_gen[t].exp(
                    tau=self.transferred_time[t], method=self.exponential_method,
                    is_skew_hermitian=self._is_skew_hermitian)

    def _compute_propagation_derivatives(self) -> None:
        """
        Computes the frechet derivatives of the propagators.

        The derivatives are not returned but cached. Since the function is only
        called when no derivatives are cached, the approximation is
        prioritised.
        """
        if not self.frechet_deriv_approx_method:
            self._compute_propagation(calculate_propagator_derivatives=True)
        elif self.frechet_deriv_approx_method == 'grape':
            if self._prop is None:
                self._compute_propagation(
                    calculate_propagator_derivatives=False)
            self._derivative_prop = [[None for _ in range(len(self.h_ctrl))]
                                     for _2 in range(len(self.transferred_time))]
            derivative_directions = self._compute_derivative_directions()
            for t in range(len(self.transferred_time)):
                for ctrl in range(len(self.h_ctrl)):
                    self._derivative_prop[t][ctrl] = \
                        self.transferred_time[t] * derivative_directions[t][ctrl] \
                        * self._prop[t]
        else:
            raise ValueError('Unknown gradient derivative approximation '
                             'method:'
                             + str(self.frechet_deriv_approx_method))


def _compute_matrix_exponentials(input_dict):
    """Computes the propagator of the Schroedinger equation by evaluation of
    a matrix exponential.

    Parameters
    ----------
    input_dict: dict
        Holds the parameters in a single dict, because the function
        multiprocessing.Pool.map requires a single input argument. The dict
        has the fields time, matrices, method and is_skew_hermitian. See also
        _compute_propagator.

    Returns
    -------
    exponentials: list of ControlMatrix
        A list of the propagators.

    """
    time = input_dict['time']
    matrices = input_dict['matrices']
    method = input_dict['method']
    is_skew_hermitian = input_dict['is_skew_hermitian']

    exponentials = [None, ] * len(time)
    for i, m, t in zip(range(len(matrices)), matrices, time):
        exponentials[i] = m.exp(
            tau=t,
            method=method,
            is_skew_hermitian=is_skew_hermitian)
    return exponentials


class SchroedingerSMonteCarlo(SchroedingerSolver):
    r"""
    Solves Schroedinger's equation for explicit noise realisations as Monte
    Carlo experiment.

    This time slot computer solves the Schroedinger equation explicitly for
    concrete noise realizations. The noise traces are generated by an instance
    of the Noise Trace Generator Class. Then they can be processed by the
    noise amplitude function, before they are multiplied by the noise
    hamiltionians.

    Parameters
    ----------
    h_noise: List[ControlMatrix], len num_noise_operators
        List of noise operators occurring in the Hamiltonian.

    noise_trace_generator: noise.NoiseTraceGenerator
        Noise trace generator object.

    processes: int, optional
        If an integer is given, then the propagation is calculated in
        this number of parallel processes. If 1 then no parallel
        computing is applied. If None then cpu_count() is called to use
        all cores available. Defaults to 1.

    noise_amplitude_function: Callable[[noise_samples: np.array,
        optimization_parameters: np.array,
        transferred_parameters: np.array,
        control_amplitudes: np.array], np.array]
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
    h_noise: List[ControlMatrix], len num_noise_operators
        List of noise operators occurring in the Hamiltonian.

    noise_trace_generator: noise.NoiseTraceGenerator
        Noise trace generator object.

    _dyn_gen_noise: List[List[ControlMatrix]],
        shape [[] * num_t] * num_noise_traces
        Dynamics generators for the individual noise traces.

    _prop_noise: List[List[ControlMatrix]],
        shape [[] * num_t] * num_noise_traces
        Propagators for the individual noise traces.

    _fwd_prop_noise: List[List[ControlMatrix]],
        shape [[] * (num_t + 1)] * num_noise_traces
        Cumulation of the propagators for the individual noise traces. They
        describe the forward propagation of the systems state.

    _reversed_prop_noise: List[List[ControlMatrix]],
        shape [[] * (num_t + 1)] * num_noise_traces
        Cumulation of propagators in reversed order for the individual noise
        traces.

    _derivative_prop_noise: List[List[List[ControlMatrix]]],
        shape [[[] * num_t] * num_ctrl] * num_noise_traces
        Frechet derivatives of the propagators by the control amplitudes for
        the individual noise traces.

    Methods
    -------
    propagators_noise: List[List[ControlMatrix]],
        shape [[] * num_t] * num_noise_traces
        Propagators for the individual noise traces.

    forward_propagators_noise: List[List[ControlMatrix]],
        shape [[] * (num_t + 1)] * num_noise_traces
        Cumulation of the propagators for the individual noise traces. They
        describe the forward propagation of the systems state.

    reversed_propagators_noise: List[List[ControlMatrix]],
        shape [[] * (num_t + 1)] * num_noise_traces
        Cumulation of propagators in reversed order for the individual noise
        traces.

    frechet_deriv_propagators_noise: List[List[List[ControlMatrix]]],
        shape [[[] * num_t] * num_ctrl] * num_noise_traces
        Frechet derivatives of the propagators by the control amplitudes for
        the individual noise traces.

    create_ff_h_n(self): List[List[np.ndarray, list, str]], 
        shape [[]]*num_noise_operators
        Creates the noise hamiltonian of the filter function formalism.

    """
    def __init__(
            self, h_drift: List[q_mat.OperatorMatrix],
            h_ctrl: List[q_mat.OperatorMatrix],
            tau: np.array,
            h_noise: List[q_mat.OperatorMatrix],
            noise_trace_generator:
            Optional[noise.NoiseTraceGenerator],
            initial_state: q_mat.OperatorMatrix = None,
            ctrl_amps: Optional[np.array] = None,
            calculate_propagator_derivatives: bool = False,
            processes: Optional[int] = 1,
            filter_function_h_n: Union[
                Callable, List[List], None] = None,
            filter_function_basis: Optional[basis.Basis] = None,
            filter_function_n_coeffs_deriv: Optional[
                Callable[[np.ndarray], np.ndarray]] = None,
            exponential_method: Optional[str] = None,
            frechet_deriv_approx_method: Optional[str] = None,
            is_skew_hermitian: bool = True,
            transfer_function: Optional[TransferFunction] = None,
            amplitude_function: Optional[AmplitudeFunction] = None,
            noise_amplitude_function: Optional[Callable[
                [np.array, np.array, np.array,
                 np.array], np.array]] = None
    ):

        super().__init__(
            h_drift=h_drift, h_ctrl=h_ctrl, initial_state=initial_state,
            tau=tau, ctrl_amps=ctrl_amps,
            filter_function_h_n=filter_function_h_n,
            filter_function_basis=filter_function_basis,
            filter_function_n_coeffs_deriv=filter_function_n_coeffs_deriv,
            exponential_method=exponential_method,
            calculate_propagator_derivatives=calculate_propagator_derivatives,
            frechet_deriv_approx_method=frechet_deriv_approx_method,
            is_skew_hermitian=is_skew_hermitian,
            transfer_function=transfer_function,
            amplitude_function=amplitude_function)

        self.h_noise = h_noise
        self.noise_trace_generator = noise_trace_generator
        self.noise_amplitude_function = noise_amplitude_function
        self.processes = processes

        self._dyn_gen_noise = None
        self._prop_noise = None
        self._derivative_prop_noise = None
        self._fwd_prop_noise = None
        self._reversed_prop_noise = None

    def set_optimization_parameters(self, y: np.array) -> None:
        """See base class. """
        if not np.array_equal(self._opt_pars, y):
            self.reset_cached_propagators()
        super().set_optimization_parameters(y)

    def reset_cached_propagators(self):
        """See base class. """
        super().reset_cached_propagators()
        self._dyn_gen_noise = None
        self._prop_noise = None
        self._derivative_prop_noise = None
        self._fwd_prop_noise = None
        self._reversed_prop_noise = None


    @property
    def propagators_noise(self) -> List[List[q_mat.OperatorMatrix]]:
        """
        Returns the propagators of the system for each noise trace and
        calculates them if necessary.

        Returns
        -------
        propagators_noise: List[List[ControlMatrix]],
        shape [[] * num_t] * num_noise_traces
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
        shape [[] * (num_t + 1)] * num_noise_traces
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
        shape [[[] * num_t] * num_ctrl] * num_noise_traces
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
        identity matrix, then the reversed cumulative propagators are given.
        The element forward_propagators[k][i] propagates a state by the first i
        time steps under the kth noise trace, if the initial state is the
        identity matrix.

        Returns
        -------
        reversed_propagation_noise: List[List[ControlMatrix]],
        shape [[] * (num_t + 1)] * num_noise_traces
            Propagation of the initial state of the system. reversed[k][0]
            gives the initial state itself.

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
        shape [[] * num_t] * num_noise_traces
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
            self, calculate_propagator_derivatives: Optional[bool] = None
    ) -> None:
        """
        Computes the propagators for the perturbed Schroedinger equation and
        the derivatives on demand.

        Parameters
        ----------
        calculate_propagator_derivatives: bool, optional
            Calculate the derivatives of the propagators with respect to the
            control amplitudes if true.

        """

        if self._dyn_gen_noise is None:
            self._dyn_gen_noise = self._compute_dyn_gen_noise()

        n_noise_traces = self.noise_trace_generator.n_traces
        num_t = len(self.transferred_time)
        num_ctrl = len(self.h_ctrl)

        self._prop_noise = [[None for _ in range(num_t)]
                            for _2 in range(n_noise_traces)]

        if calculate_propagator_derivatives is None:
            calculate_propagator_derivatives = \
                self.calculate_propagator_derivatives

        # parallelization of following code probably unnecessary
        if calculate_propagator_derivatives:
            self._derivative_prop_noise = \
                [[[None for _ in range(num_t)]
                  for _2 in range(num_ctrl)]
                 for _3 in range(n_noise_traces)]
            derivative_directions = self._compute_derivative_directions()

        # call the parent method for the noiseless propagators
        super()._compute_propagation(
            calculate_propagator_derivatives=calculate_propagator_derivatives)

        if self.processes == 1:
            if calculate_propagator_derivatives:
                for k in range(n_noise_traces):
                    for t in range(num_t):
                        for ctrl in range(len(self.h_ctrl)):
                            self._prop_noise[k][t], \
                                self._derivative_prop_noise[k][ctrl][t] \
                                = self._dyn_gen_noise[k][t].dexp(
                                derivative_directions[t][ctrl],
                                self.transferred_time[t],
                                compute_expm=True,
                                method=self.exponential_method,
                                is_skew_hermitian=self._is_skew_hermitian)
            else:
                for k in range(n_noise_traces):
                    for t in range(num_t):
                        self._prop_noise[k][t] = self._dyn_gen_noise[k][t].exp(
                            tau=self.transferred_time[t],
                            method=self.exponential_method,
                            is_skew_hermitian=self._is_skew_hermitian)

        elif (type(self.processes) == int and self.processes > 0) \
                or self.processes is None:

            if calculate_propagator_derivatives:
                raise NotImplementedError
            else:
                input_dicts = []
                for k in range(n_noise_traces):
                    input_dicts.append(dict())
                    input_dicts[-1]['time'] = self.transferred_time
                    input_dicts[-1]['matrices'] = self._dyn_gen_noise[k]
                    input_dicts[-1]['method'] = self.exponential_method
                    input_dicts[-1][
                        'is_skew_hermitian'] = self._is_skew_hermitian

                with Pool(processes=self.processes) as pool:
                    self._prop_noise = pool.map(
                        _compute_matrix_exponentials, input_dicts)

        else:
            raise ValueError('Invalid number of processes for parallel '
                             'computation!')

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
        called when no derivatives are cached, the approximation is
        prioritised.
        """
        if not self.frechet_deriv_approx_method:
            self._compute_propagation(calculate_propagator_derivatives=True)
        elif self.frechet_deriv_approx_method == 'grape':
            super()._compute_propagation_derivatives()

            if self._prop_noise is None:
                self._compute_propagation(
                    calculate_propagator_derivatives=False)

            n_noise_traces = self.noise_trace_generator.n_traces
            num_t = len(self.transferred_time)
            num_ctrl = len(self.h_ctrl)

            self._derivative_prop_noise = [
                [[None for _ in range(num_t)]
                 for _2 in range(num_ctrl)]
                for _3 in range(n_noise_traces)]

            derivative_directions = self._compute_derivative_directions()

            for k in range(n_noise_traces):
                for t in range(len(self.transferred_time)):
                    for ctrl in range(num_ctrl):
                        self._derivative_prop_noise[k][ctrl][t] = \
                            self.transferred_time[t] * derivative_directions[t][ctrl] \
                            * self._prop_noise[k][t]
        else:
            raise ValueError('Unknown gradient derivative approximation '
                             'method:'
                             + str(self.frechet_deriv_approx_method))

    @property
    def create_ff_h_n(self) -> list:
        """Creates the noise hamiltonian of the filter function formalism.

        If `filter_function_h_n` is None, then the filter function noise
        Hamiltonian is created from the Monte Carlo noise Hamiltonian by
        directly using the Operators and assuming all noise susceptibilities
        equal 1.

        Returns
        -------
        create_ff_h_n: nested list
            Noise Hamiltonian of the filter function formalism.

        """
        if type(self._filter_function_h_n) == list:
            h_n = self._filter_function_h_n
        else:
            h_n = self._filter_function_h_n(self._opt_pars)

        if not h_n:
            h_n = []
            for i, noise_operator in enumerate(self.h_noise):
                if type(noise_operator) == matrix.DenseOperator:
                    noise_operator = noise_operator.data
                h_n += [[noise_operator, len(self.transferred_time) * [1], 'Noise' + str(i)], ]

        return h_n


class SchroedingerSMCControlNoise(SchroedingerSMonteCarlo):
    """
    Convenience class like `SchroedingerSMonteCarlo` but with noise on the
    optimization parameters.

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
            tau: np.array,
            noise_trace_generator:
            Optional[noise.NoiseTraceGenerator],
            initial_state: q_mat.OperatorMatrix = None,
            ctrl_amps: Optional[np.array] = None,
            calculate_propagator_derivatives: bool = False,
            processes: Optional[int] = 1,
            filter_function_h_n: Union[
                Callable, List[List], None] = None,
            filter_function_basis: Optional[basis.Basis] = None,
            filter_function_n_coeffs_deriv: Optional[
                Callable[[np.ndarray], np.ndarray]] = None,
            exponential_method: Optional[str] = None,
            frechet_deriv_approx_method: Optional[str] = None,
            is_skew_hermitian: bool = True,
            transfer_function: Optional[TransferFunction] = None,
            amplitude_function: Optional[AmplitudeFunction] = None):

        def noise_amplitude_function(noise_samples: np.array,
                                     transferred_parameters: np.array,
                                     control_amplitudes: np.array,
                                     **_):
            """Calculates the noise amplitudes.

            Takes into account the actual optimization parameters and random
            variations.

            Parameters
            ----------
            noise_samples: np.array, shape()
                Noise samples calculated by the noise trace generator.

            transferred_parameters: np.array
                Transferred optimization parameters.

            control_amplitudes: np.array
                Control amplitudes.

            """
            # noise_amplitudes = np.zeros_like(noise_samples, dtype=complex)
            noise_amplitudes = np.zeros(
                (noise_samples.shape[0], noise_samples.shape[1],
                    control_amplitudes.shape[1]), dtype=complex)

            # complex values were requested.
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
            processes=processes,
            filter_function_h_n=filter_function_h_n,
            filter_function_basis=filter_function_basis,
            filter_function_n_coeffs_deriv=filter_function_n_coeffs_deriv,
            exponential_method=exponential_method,
            frechet_deriv_approx_method=frechet_deriv_approx_method,
            is_skew_hermitian=is_skew_hermitian,
            transfer_function=transfer_function,
            amplitude_function=amplitude_function,
            noise_amplitude_function=noise_amplitude_function
        )


class LindbladSolver(SchroedingerSolver):
    r"""
    Solves a master equation for an open quantum system in the Markov
    approximation using the Lindblad super operator formalism.

    The master equation to be solved is

    .. math::

        d \rho / dt = i [\rho, H] + \sum_k (L_k \rho L_k^\dagger
        - .5 L_k^\dagger L_k \rho - .5 \rho L_k^\dagger L_k)


    with the Lindblad operators L_k. The solution is calculated as

    .. math::

        \rho(t) = exp[(-i \mathcal{H} + \mathcal{G})t] \rho(0)

    with the dissipative super operator

    .. math::

        \mathcal{G} = \sum_k D(L_k)

    .. math::

        D(L) = L^\ast \otimes L - .5 I \otimes (L^\dagger L)
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

    If multiple construction arguments are given, the implementation
    prioritises the function (3.) over the Lindblad operators (2.) over the
    dissipation super operator (1.).

    Parameters
    ----------
    initial_diss_super_op: List[ControlMatrix], len num_t
        Initial dissipation super operator; num_l is the number of
        Lindbladians. Set if you want to use (1.) (See documentation above!).
        The control matrices are expected to be of shape (dim, dim) where dim
        is the dimension of the system.

    lindblad_operators: List[ControlMatrix], len num_l
        Lindblad operators; num_l is the number of Lindbladians. Set if you
        want to use (2.) (See documentation above!). The Lindblad operators are
        assumend to be of shape (dim, dim) where dim is the dimension of the
        system.

    prefactor_function: Callable[[np.array, np.array], np.array]
        Receives the control amplitudes u (as numpy array of shape
        (num_t, num_ctrl)) and the transferred optimization parameters (as
        numpy array of shape (num_t, num_opt)) and returns prefactors as numpy
        array of shape (num_t, num_l). The prefactors a_k are used as weights in
        the sum of the total dissipation operator.

        .. math::

            \mathcal{G} = \sum_k a_k * D(L_k)

        If the Lindblad operator is for example given by a complex number b_k
        times a constant (in time) matrix C_k.

        .. math::

            L_k = b_k * C_k

        Then the prefactor is the squared absolute value of this number:

        .. math::

            a_k = |b_k|^2

        Set if you want to use method (1.) or (2.). (See class documentation.)

    prefactor_derivative_function: Callable[[np.array, np.array], np.array]
        Receives the control amplitudes u (as numpy array of shape
        (num_t, num_ctrl)) and the transferred optimization parameters (as
        numpy array of shape (num_t, num_opt)) and returns the derivatives of
        the prefactors as numpy array of shape (num_t, num_ctrl, num_l). The
        derivatives d_k are used as weights in the sum of the derivative of the
        total dissipation operator.

        .. math::

            d \mathcal{G} / d u_k = \sum_k d_k * D(L_k)

        If the Lindblad operator is for example given by a complex number b_k
        times a constant (in time) matrix C_k. And this number depends on the
        control amplitudes u_k

        .. math::

            L_k = b_k (u_k) * C_k

        Then the derivative of the prefactor is the derivative of the squared
        absolute value of this number:

        .. math::

            d_k = d |b_k|^2 / d u_k

        Set if you want to use method (1.) or (2.). (See class documentation.)

    super_operator_function: Callable[[np.array, np.array], List[ControlMatrix]]
        Receives the control amlitudes u (as numpy array of shape
        (num_t, num_ctrl)) and the transferred optimization parameters (as
        numpy array of shape (num_t, num_opt)) and returns the total dissipation
        operators as list of length num_t. Set if you want to use method (3.).
        (See class documentation.)

    super_operator_derivative_function: Callable[[np.array, np.array],
        List[List[ControlMatrix]]]
        Receives the control amlitudes u (as numpy array of shape
        (num_t, num_ctrl)) and the transferred optimization parameters (as
        numpy array of shape (num_t, num_opt)) and returns the derivatives of
        the total dissipation operators as nested list of
        shape [[] * num_ctrl] * num_t. Set if you
        want to use method (3.). (See class documentation.)

    is_skew_hermitian: bool
        If True, then the total dynamics generator is assumed to be skew
        hermitian.

    Attributes
    ----------
    _diss_sup_op: List[ControlMatrix], len num_t
        Total dissipaton super operator.

    _diss_sup_op_deriv: List[List[ControlMatrix]],
        shape [[] * num_ctrl] * num_t
        Derivative of the total dissipation operator with respect to the
        control amplitudes.

    _initial_diss_super_op: List[ControlMatrix], len num_l
        Initial dissipation super operator; num_l is the number of
        Lindbladians.

    _lindblad_operatorsList[ControlMatrix], len num_l
        Lindblad operators; num_l is the number of Lindbladians.

    _prefactor_function: Callable[[np.array], np.array]
        Receives the control amplitudes u (as numpy array of shape
        (num_t, num_ctrl)) and returns prefactors as numpy array
        of shape (num_t, num_l). The prefactors a_k are used as weights in the
        sum of the total dissipation operator.

        .. math::

            \mathcal{G} = \sum_k a_k * D(L_k)

        If the Lindblad operator is for example given by a complex number b_k
        times a constant (in time) matrix C_k.

        .. math::

            L_k = b_k * C_k

        Then the prefactor is the squared absolute value of this number:

        .. math::

            a_k = |b_k|^2

        Set if you want to use method (1.) or (2.). (See class documentation.)

    _prefactor_deriv_function: Callable[[np.array], np.array]
        Receives the control amplitudes u (as numpy array of shape
        (num_t, num_ctrl)) and returns the derivatives of the
        prefactors as numpy array of shape (num_t, num_ctrl, num_l). The
        derivatives d_k are used as weights in the sum of the derivative of the
        total dissipation operator.

        .. math::

            d \mathcal{G} / d u_k = \sum_k d_k * D(L_k)

        If the Lindblad operator is for example given by a complex number b_k
        times a constant (in time) matrix C_k. And this number depends on the
        control amplitudes u_k

        .. math::

            L_k = b_k (u_k) * C_k

        Then the derivative of the prefactor is the derivative of the squared
        absolute value of this number:

        .. math::

            d_k = d |b_k|^2 / d u_k

    _sup_op_func: Callable[[np.array], List[ControlMatrix]]
        Receives the control amplitudes u (as numpy array of shape
        (num_t, num_ctrl)) and returns the total dissipation
        operators as list of length num_t.

    _sup_op_deriv_func: Callable[[np.array], List[List[ControlMatrix]]]
        Receives the control amplitudes u (as numpy array of shape
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

    `Todo`
        * Write parser

    """

    def __init__(
            self,
            h_drift: List[q_mat.OperatorMatrix],
            h_ctrl: List[q_mat.OperatorMatrix],
            tau: np.array,
            initial_state: q_mat.OperatorMatrix = None,
            ctrl_amps: Optional[np.array] = None,
            calculate_unitary_derivatives: bool = False,
            filter_function_h_n: Union[
                Callable, List[List], None] = None,
            filter_function_basis: Optional[basis.Basis] = None,
            filter_function_n_coeffs_deriv: Optional[
                Callable[[np.ndarray], np.ndarray]] = None,
            exponential_method: Optional[str] = None,
            frechet_deriv_approx_method: Optional[str] = None,
            initial_diss_super_op: List[q_mat.OperatorMatrix] = None,
            lindblad_operators: List[q_mat.OperatorMatrix] = None,
            prefactor_function: Callable[[np.array, np.array], np.array] = None,
            prefactor_derivative_function:
            Callable[[np.array, np.array], np.array] = None,
            super_operator_function:
            Callable[[np.array, np.array], List[q_mat.OperatorMatrix]] = None,
            super_operator_derivative_function:
            Callable[[np.array, np.array],
                     List[List[q_mat.OperatorMatrix]]] = None,
            is_skew_hermitian: bool = False,
            transfer_function: Optional[TransferFunction] = None,
            amplitude_function: Optional[AmplitudeFunction] = None) \
            -> None:

        if initial_state is None:
            dim = h_ctrl[0].shape[0]
            initial_state = type(h_ctrl[0])(np.eye(dim ** 2))

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

        super().__init__(
            h_drift=h_drift, h_ctrl=h_ctrl, initial_state=initial_state,
            tau=tau, ctrl_amps=ctrl_amps,
            calculate_propagator_derivatives=calculate_unitary_derivatives,
            filter_function_h_n=filter_function_h_n,
            filter_function_basis=filter_function_basis,
            filter_function_n_coeffs_deriv=filter_function_n_coeffs_deriv,
            exponential_method=exponential_method,
            frechet_deriv_approx_method=frechet_deriv_approx_method,
            is_skew_hermitian=is_skew_hermitian,
            transfer_function=transfer_function,
            amplitude_function=amplitude_function)

    def set_optimization_parameters(self, y: np.array) -> None:
        """See base class. """
        if not np.array_equal(self._opt_pars, y):
            super().set_optimization_parameters(y)
            self.reset_cached_propagators()

    def reset_cached_propagators(self):
        """ See base class. """
        super().reset_cached_propagators()
        if self._prefactor_function is not None \
                or self._sup_op_func is not None:
            self._diss_sup_op = None
            self._diss_sup_op_deriv = None

    def _calc_diss_sup_op(self) -> List[q_mat.OperatorMatrix]:
        r"""
        Calculates the dissipative super operator as described in the class
        doc string.

        Returns
        -------
        diss_sup_op: List[ControlMatrix], len num_t
            Dissipation super operator; Where num_t is the number of time
            steps.
        
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
                        (lindblad.conj(do_copy=True)).kron(lindblad))
                    const_diss_sup_op[-1] -= .5 * identity.kron(
                        lindblad.dag(do_copy=True) * lindblad)
                    const_diss_sup_op[-1] -= .5 * (
                        lindblad.transpose(do_copy=True)
                        * lindblad.conj(do_copy=True)).kron(identity)

            # Add the time dependence
            if self._prefactor_function is not None:
                self._diss_sup_op = []
                prefactors = self._prefactor_function(
                    copy.deepcopy(self._ctrl_amps),
                    copy.deepcopy(self.transferred_parameters))
                for factor_at_time_t in prefactors:
                    self._diss_sup_op.append(
                        const_diss_sup_op[0] * factor_at_time_t[0])
                    for sup_op, factor \
                            in zip(const_diss_sup_op[1:],
                                   factor_at_time_t[1:]):
                        self._diss_sup_op[-1] += sup_op * factor
            else:
                self._diss_sup_op = [const_diss_sup_op[0], ]
                for sup_op in const_diss_sup_op[1:]:
                    self._diss_sup_op[0] += sup_op
                self._diss_sup_op *= len(self.transferred_time)
        else:
            self._diss_sup_op = self._sup_op_func(
                copy.deepcopy(self._ctrl_amps),
                copy.deepcopy(self.transferred_parameters))
        return self._diss_sup_op

    def _calc_diss_sup_op_deriv(self) \
            -> Optional[List[List[q_mat.OperatorMatrix]]]:
        r"""
        Calculates the derivatives of the dissipation super operator with
        respect to the control amplitudes.

        If the dissipation super operator is given as constant (1.) or as
        lindblad operators (2.) they are assumed not to depend on the control
        parameters and only the derivative of the prefactor is to be taken into
        account. In order to do so, a function handle containing the
        derivatives must be given. This function receives the control
        amplitudes as num_t x num_ctrl numpy array and returns the derivatives
        as num_t x num_l x num_ctrl array.

        If the dissipation super operator is given as function handle (3.),
        then the derivatives must also be given as function handle receiving
        the control amplitudes and returning a nested list of super operators
        as control matrices.

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
                    copy.deepcopy(self._ctrl_amps),
                    copy.deepcopy(self.transferred_parameters))
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
                        (lindblad.conj(do_copy=True)).kron(lindblad))
                    const_diss_sup_op[-1] -= .5 * identity.kron(
                        lindblad.dag(do_copy=True) * lindblad)
                    const_diss_sup_op[-1] -= .5 * (
                        lindblad.transpose(do_copy=True)
                        * lindblad.conj(do_copy=True)).kron(identity)

            prefactor_derivatives = \
                self._prefactor_deriv_function(
                copy.deepcopy(self._ctrl_amps),
                copy.deepcopy(self.transferred_parameters))

            # Todo: Assert that the prefactor returns the right dimension

            # prefactor_derivatives: shape (num_t, num_ctrl, num_l)
            diss_sup_op_deriv = []
            for factor_per_ctrl_lind in prefactor_derivatives:
                # create new sub list for eacht time step
                diss_sup_op_deriv.append([])
                for factor_per_lind in factor_per_ctrl_lind:
                    # add the first term for each control direction
                    diss_sup_op_deriv[-1].append(
                        const_diss_sup_op[0] * factor_per_lind[0])
                    for diss_sup_op, factor in zip(
                            const_diss_sup_op[1:], factor_per_lind[1:]):
                        # add the remaining terms
                        diss_sup_op_deriv[-1][-1] += diss_sup_op * factor
            self._diss_sup_op_deriv = diss_sup_op_deriv
            return diss_sup_op_deriv
        else:
            return None

    def _compute_derivative_directions(
            self) -> List[List[q_mat.OperatorMatrix]]:
        r"""
        Computes the derivative directions of the total dynamics generator.

        Returns
        -------
        deriv_directions: List[List[q_mat.ControlMatrix]],
                          shape [[] * num_ctrl] * num_t
            Derivative directions given by

            .. math::

                -1j * (I \otimes H_k - H_k \otimes I) + d \mathcal{G} / d u_k

        """
        # derivative of the coherent part
        identity_times_i = self.h_ctrl[0].identity_like()
        identity_times_i *= -1j
        h_ctrl_sup_op = []
        for ctrl_op in self.h_ctrl:
            h_ctrl_sup_op.append(identity_times_i.kron(ctrl_op))
            h_ctrl_sup_op[-1] -= (ctrl_op.transpose(do_copy=True)).kron(
                identity_times_i)

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
            dh_by_ctrl = [h_ctrl_sup_op, ] * len(self.transferred_time)

        return dh_by_ctrl

    def _parse_dissipative_super_operator(self) -> None:
        r"""
        check the dissipative super operator for dimensional consistency
        (maybe even physical properties)
        - not implemented yet -
        """
        pass

    def _compute_dyn_gen(self) -> List[q_mat.OperatorMatrix]:
        r"""
        Computes the dynamics generator for the Lindblad master equation.

        The Hamiltonian is translated into the master equation formalism as

        .. math::

            \mathcal{H} = I \otimes H - H^\ast \otimes I

        Then the dissipation super operator is added.

        Returns
        -------
        dyn_gen: List[ControlMatrix], len num_t
            Dynamics generators for the master equation.

        Raises
        ------
        ValueError:
            The computation is only defined for the use of dense control
            matrices.

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
            sup_op_dyn_gen[-1] += dyn_gen.conj(do_copy=True).kron(
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
        self._dU = np.array(shape=(self.num_t, self.num_ctrl),
                              dtype=matrix.DenseOperator)
        self._fwd = [self.initial_state]

        # super operator calculation
        # this is the special case for charge noise on the control parameters
        # the required filter function contains
        if not self.constant_lindblad_operators or \
                self.incoherent_dyn_gen is None:
            transfer_matrix = self.transfer_function.transfer_matrix
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
                        direction=direction, tau=self.transferred_time[t],
                        compute_expm=True, method=self.exponential_method)

            else:
                self._prop[t] = self._dyn_gen[t].exp(
                    tau=self.transferred_time[t], method=self.exponential_method)

            self._fwd.append(self._prop[t] * self._fwd[t])

        self.prop_calculated = True
