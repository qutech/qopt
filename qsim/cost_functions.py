"""
Cost Function

These classes calculate the cost function to be minimised
and its gradient (Jacobian), which is used to direct the optimisation

The most frequent use will be the calculation of fidelities.

Classes
-------
CostFunction:
    Abstract base class of the fidelity computer.

OperatorMatrixNorm:
    Calculates the cost as matrix norm of the difference between the actual
    evolution and the target.

OperationInfidelity:
    Calculates the cost as operator infidelity of propagators.

OperationNoiseInfidelity:
    Like Operationfidelity but averaged over noise traces.

Functions
---------
angle_axis_representation:
    Calculates the representation of a unitary matrix as rotation axis and
    angle.

entanglement_fidelity:
    Calculates the entanglement fidelity between a unitary target evolution and
    a simulated unitary evolution.

deriv_entanglement_fid_sup_op_with_du:
    Calculates the derivatives of the entanglement fidelity with respect to
    the control amplitudes.

entanglement_fidelity_super_operator:
    Calculates the entanglement fidelity between two propagators in the super
    operator formalism.

derivative_entanglement_fidelity_with_du:
    Calculates the derivatives of the entanglement fidelity in the super
    operator formalism with respect to the control amplitudes.

"""

import numpy as np
import itertools
from scipy.linalg import sqrtm, inv
from typing import Sequence, Union, List, Optional, Callable, Dict
# QuTiP
from qutip import vec2mat
from qsim import matrix, solver_algorithms
from qsim.util import needs_refactoring, deprecated
import filter_functions.numeric

from abc import ABC, abstractmethod


class CostFunction(ABC):
    """
    Abstract base class of the fidelity computer.

    Attributes
    ----------
    t_slot_comp : TimeSlotComputer
        Object that compute the forward/backward evolution and propagator.

    target: np.array
        State or operator which is which the evolution is compared to.

    index: List[str]
        Indices of the returned infidelities for distinction in the analysis.

    Methods
    -------
    costs():

    grad():

    """
    def __init__(self):
        self.t_slot_comp = None
        self.target = None
        self.index = "Unspecified Cost Function"

    @abstractmethod
    def costs(self) -> Union[float, np.ndarray]:
        """The costs or infidelity of the quantum channel.

        Returns
        -------
        costs : np.array or float
            Result of the cost function's evaluation.
        """
        pass

    @abstractmethod
    def grad(self) -> np.ndarray:
        """The gradient of the costs or infidelity of the quantum channel.

        Returns
        -------
        gradient : np.array
            shape: (num_t, num_ctrl, num_f) where num_t is the number of time
            slices, num_ctrl the number of control parameters and num_f the
            length of the costs returned by the cost function. Derivatives of
            the cost function by the control amplitudes. """
        pass


def angle_axis_representation(u: Union[np.ndarray, matrix.DenseOperator]) \
        -> (float, np.ndarray):
    """
    Calculates the representation of a unitary matrix by a rotational axis and
    a rotation angle.

    Parameters
    ----------
    u: np.ndarray
        A unitary matrix.

    Returns
    -------
    beta, n: float, np.ndarray
        beta is the angle of the rotation and n the rotational axis.

    TODO:
        * implement for control matrices. Not only numpy arrays.

    """
    # check if u is unitary
    ident = u @ np.conjugate(np.transpose(u))
    is_unitary = np.isclose(ident[0, 0], 1) and np.isclose(ident[1, 0], 0) and \
        np.isclose(ident[0, 1], 0) and np.isclose(ident[0, 0], 1)
    if not is_unitary:
        raise ValueError("Your input matrix must be unitary to calculate a "
                         "angle axis representation!")

    cos = .5 * (u[0, 0] + u[1, 1])
    if np.isclose(cos, 1):
        return 0, np.array([1, 0, 0])
    sin = np.sqrt(1 - cos ** 2)
    n_1 = np.real((u[0, 1] + u[1, 0]) / 1j / sin / 2)
    n_2 = np.real((u[0, 1] - u[1, 0]) / sin / 2)
    n_3 = np.real((u[0, 0] - u[1, 1]) / 1j / sin / 2)
    # beta = np.real(np.arccos(cos) * 2)
    # It seems more coherent to neglect the factor of 2
    beta = np.real(np.arccos(cos) * 2)
    assert np.isclose(np.linalg.norm(np.array([n_1, n_2, n_3])), 1, atol=1e-5)
    return beta, np.array([n_1, n_2, n_3])


class OperatorMatrixNorm(CostFunction):
    """
    Computes the fidelity as differences in the unitary evolution without
    global phase.

    The result can be returned as absolute value or vector.

    Parameters
    ----------
    t_slot_comp: TimeSlotComputer
        Computes the evolution of the system.

    target: ControlMatrix
        The ideal evolution.

    mode: string
        The type of calculation.
        'scalar': The difference is returned as scalar.
        'vector': The difference of the individual elements is returned as
        vector.
        'rotation_axis': For unitary evolutions only. The evolution is described
        by its rotation axis and a rotation angle. The first element of the
        rotation axis is multiplied by the angle so save one return argument.


    Attributes
    ----------
    mode: string
        Type of calculation


    TODO:
        * implementation for target[0,0] != 0

    """

    @needs_refactoring
    def __init__(self, t_slot_comp: solver_algorithms.Solver,
                 target: matrix.OperatorMatrix, mode: str = 'scalar',
                 index: Optional[List[str]] = None):
        super().__init__()
        self.t_slot_comp = t_slot_comp
        self.target = target
        self.mode = mode
        if index is not None:
            self.index = index
        elif mode == 'scalar':
            self.index = ['Matrix Norm Distance']
        elif mode == 'vector':
            dim = target.shape[0]
            self.index = ['redu' + str(i) + str(j)
                          for i in range(1, dim + 1)
                          for j in range(1, dim + 1)] + [
                             'imdu' + str(i) + str(j)
                             for i in range(1, dim + 1)
                             for j in range(1, dim + 1)]
        elif mode == 'rotation_axis':
            self.index = ['n1 * phi', 'n2', 'n3']
        else:
            raise ValueError('Unknown fidelity computer mode: ' + str(mode)
                             + ' \n possible modes are: "scalar", "vector" '
                               'and in 2 "dimensions rotations_axis".')

    def costs(self) -> Union[np.ndarray, float]:
        """
        The costs or infidelity of the quantum channel.

        These costs are given as difference between a simulated unitary
        evolution and the unitary target evolution depending on the mode.
        (See class documentation. )

        Returns
        -------
        costs: Union[np.ndarray, float]
            The costs of infidelity.

        """
        final = self.t_slot_comp.forward_propagators[-1]

        # to eliminate a global phase we require final[0, 0] to be real.
        if self.mode == 'rotation_axis':
            ax1 = angle_axis_representation(self.target.data)
            ax2 = angle_axis_representation(final.data)
            diff = ax1[0] * ax1[1] - ax2[0] * ax2[1]
            return diff
        else:
            if not np.isclose(final.data[0, 0], 0):
                final_phase_eliminated = final * (1 / (
                        final[0, 0] / np.abs(final[0, 0])))
            else:
                raise NotImplementedError
            diff = final_phase_eliminated - self.target
            if self.mode == 'scalar':
                return np.sum(np.abs(diff.data))
            elif self.mode == 'vector':
                return np.concatenate(
                    (np.real(diff.data.flatten()),
                     np.imag(diff.data.flatten())))
            else:
                raise ValueError('Unknown mode in the fidelity computer.')

    def grad(self) -> np.ndarray:
        """
        Calculates the Jacobian of the matrix difference.

        Only implemented for the mode 'vector'.

        Returns
        -------
        jacobian: np.ndarray
            Jacobian of the matrix difference.

        Raises
        ------
        NotImplementedError:
            If self.mode is not 'vector'.

        """
        if self.mode != 'vector':
            raise NotImplementedError('The gradient calculation is currently'
                                      'only implemented for the mode "vector".')

        # grape
        propagators = self.t_slot_comp.propagators
        forward_prop_cumulative = self.t_slot_comp.forward_propagators
        # reversed_prop_cumulative = self.t_slot_comp.reversed_prop_cumulative
        unity = matrix.DenseOperator(
            np.eye(propagators[0].data.shape[0]))
        propagators_future = [unity]
        for prop in propagators[:0:-1]:
            propagators_future.append(propagators_future[-1] * prop)
        propagators_future = propagators_future[::-1]

        if isinstance(self.t_slot_comp.tau, list):
            tau = self.t_slot_comp.tau[0]
        elif isinstance(self.t_slot_comp.tau, float):
            tau = self.t_slot_comp.tau
        else:
            raise NotImplementedError

        num_t = len(self.t_slot_comp.tau)
        num_ctrl = len(self.t_slot_comp.h_ctrl)
        jacobian_complex_full = np.zeros(
            shape=[self.target.data.size, num_t,
                   num_ctrl]).astype(complex)
        final = self.t_slot_comp.forward_propagators[-1]
        exp_iphi = final[0, 0] / np.abs(final[0, 0])
        # * 2 for the seperation of imaginary and real part

        for j in range(num_ctrl):
            for i, (prop, fwd_prop, future_prop) in enumerate(
                    zip(propagators, forward_prop_cumulative,
                        propagators_future)):
                # here i applied the grape approximations
                complex_jac = (
                    -1j * tau * future_prop * self.t_slot_comp.h_ctrl[j]
                    * fwd_prop).flatten()
                jacobian_complex_full[:, i, j] = complex_jac.data

        dphi_by_du = (
            np.imag(jacobian_complex_full[0, :, :]) * np.real(final[0, 0]) -
            np.real(jacobian_complex_full[0, :, :]) * np.imag(final[0, 0])
            ) / ((np.abs(final[0, 0])) ** 2)
        final.flatten()

        dphi_by_du_times_u = np.concatenate\
            ([np.reshape(dphi_by_du, (1, dphi_by_du.shape[0],
                                      dphi_by_du.shape[1])) * fin
             for fin in final.data])

        jacobian_complex_full = (jacobian_complex_full -
                                 1j * dphi_by_du_times_u) * (1 / exp_iphi)

        # The result must be corrected by a sign depending on the angle phi

        jacobian = np.concatenate([np.real(jacobian_complex_full),
                                   np.imag(jacobian_complex_full)], axis=0)
        return jacobian


def entanglement_fidelity(
        target_unitary: Union[np.ndarray, matrix.OperatorMatrix],
        unitary: Union[np.ndarray, matrix.OperatorMatrix],
        computational_states: Optional[List[int]] = None,
        map_to_closest_unitary: bool = False
) -> np.float64:
    """
    The entanglement fidelity between a unitary simulated and target unitary.

    Parameters
    ----------
    unitary: Union[np.ndarray, ControlMatrix]
        The simulated unitary evolution.

    target_unitary: Union[np.ndarray, ControlMatrix]
        The target unitary evolution.

    computational_states: Optional[List[int]]
        If set, the entanglement fidelity is only calculated for the specified
        subspace.

    map_to_closest_unitary: bool
        If True, then the final propagator is mapped to the closest unitary
        before the infidelity is evaluated.

    Returns
    -------
    fidelity: float
        The entanglement fidelity of target_unitary.dag * unitary.

    """
    if type(unitary) == np.ndarray:
        unitary = matrix.DenseOperator(unitary)
    if type(target_unitary) == np.ndarray:
        target_unitary = matrix.DenseOperator(target_unitary)
    d = target_unitary.shape[0]
    if computational_states is None:
        trace = (target_unitary.dag() * unitary).tr()
    else:
        trace = (target_unitary.dag() * unitary.truncate_to_subspace(
            computational_states,
            map_to_closest_unitary=map_to_closest_unitary)).tr()
    return (np.abs(trace) ** 2) / d / d


def derivative_entanglement_fidelity_with_du(
        target_unitary: matrix.OperatorMatrix,
        forward_propagators: List[matrix.OperatorMatrix],
        unitary_derivatives: List[List[matrix.OperatorMatrix]],
        reversed_propagators: List[matrix.OperatorMatrix],
        computational_states: Optional[List[int]] = None,
        map_to_closest_unitary: bool = False
) -> np.ndarray:
    """
    Derivative of the entanglement fidelity using the derivatives of the
    propagators.

    Parameters
    ----------
    forward_propagators: List[ControlMatrix], len: num_t +1
        The forward propagators calculated in the systems simulation.

    unitary_derivatives: List[List[ControlMatrix]],
                         shape: [[] * num_t] * num_ctrl
        Frechet derivatives of the propagators by the control amplitudes.

    target_unitary: ControlMatrix
        The target unitary evolution.

    reversed_propagators: List[ControlMatrix]
        The reversed propagators calculated in the systems simulation.

    computational_states: Optional[List[int]]
        If set, the entanglement fidelity is only calculated for the specified
        subspace.

    map_to_closest_unitary: bool
        If True, then the final propagator is mapped to the closest unitary
        before the infidelity is evaluated.

    Returns
    -------
    derivative_fidelity: np.ndarray, shape: (num_t, num_ctrl)
        The derivatives of the entanglement fidelity.

    """
    target_unitary_dag = target_unitary.dag(copy_=True)
    if computational_states:
        trace = np.conj(((forward_propagators[-1].truncate_to_subspace(
            computational_states, map_to_closest_unitary=map_to_closest_unitary)
                          * target_unitary_dag).tr()))
    else:
        trace = np.conj(((forward_propagators[-1] * target_unitary_dag).tr()))
    num_ctrls = len(unitary_derivatives)
    num_time_steps = len(unitary_derivatives[0])
    d = target_unitary.shape[0]

    derivative_fidelity = np.zeros(shape=(num_time_steps, num_ctrls),
                                   dtype=float)

    for ctrl in range(num_ctrls):
        for t in range(num_time_steps):
            # here we need to take the real part.
            if computational_states:
                derivative_fidelity[t, ctrl] = 2 / d / d * np.real(
                    trace * ((reversed_propagators[::-1][t + 1]
                             * unitary_derivatives[ctrl][t]
                             * forward_propagators[t]).truncate_to_subspace(
                        subspace_indices=computational_states,
                        map_to_closest_unitary=map_to_closest_unitary
                    )
                             * target_unitary_dag).tr())
            else:
                derivative_fidelity[t, ctrl] = 2 / d / d * np.real(
                    trace * (reversed_propagators[::-1][t + 1]
                             * unitary_derivatives[ctrl][t]
                             * forward_propagators[t]
                             * target_unitary_dag).tr())

    return derivative_fidelity


def entanglement_fidelity_super_operator(
        target_unitary: Union[np.ndarray, matrix.OperatorMatrix],
        propagator: Union[np.ndarray, matrix.OperatorMatrix],
        computational_states: Optional[List[int]] = None,
        map_to_closest_unitary: bool = False
) -> np.float64:
    """
    The entanglement fidelity of a propagator in the super operator formalism.

    The entanglement fidelity between a propagator in the super operator
    formalism of dimension d^2 x d^2 and a target unitary operator of dimension
    d x d.

    Parameters
    ----------
    propagator: Union[np.ndarray, ControlMatrix]
        The simulated evolution propagator in the super operator formalism.

    target_unitary: Union[np.ndarray, ControlMatrix]
        The target unitary evolution. (NOT in super operator formalism.)

    computational_states: Optional[List[int]]
        If set, the entanglement fidelity is only calculated for the specified
        subspace.s only calculated for the specified
        subspace.

    map_to_closest_unitary: bool
        If True, then the final propagator is mapped to the closest unitary
        before the infidelity is evaluated.

    Returns
    -------
    fidelity: float
        The entanglement fidelity of target_unitary.dag * unitary.

    """
    if type(propagator) == np.ndarray:
        propagator = matrix.DenseOperator(propagator)
    if type(target_unitary) == np.ndarray:
        target_unitary = matrix.DenseOperator(target_unitary)
    d = target_unitary.shape[0]
    target_super_operator = \
        matrix.convert_unitary_to_super_operator(
            target_unitary.dag())
    if computational_states is None:
        trace = (target_super_operator * propagator).tr().real
    else:
        trace = (target_super_operator * propagator.truncate_to_subspace(
            computational_states,
            map_to_closest_unitary=map_to_closest_unitary)).tr().real
    return trace / d / d


def deriv_entanglement_fid_sup_op_with_du(
        target_unitary: matrix.OperatorMatrix,
        forward_propagators: List[matrix.OperatorMatrix],
        unitary_derivatives: List[List[matrix.OperatorMatrix]],
        reversed_propagators: List[matrix.OperatorMatrix],
        computational_states: Optional[List[int]] = None,
        map_to_closest_unitary: bool = False
):
    """
    Derivative of the entanglement fidelity of a super operator.

    Calculates the derivatives of the entanglement fidelity between a target
    unitary of dimension d x d and a propagator of dimension d^2 x d^2 with
    respect to the control amplitudes.

    Parameters
    ----------
    forward_propagators: List[ControlMatrix], len: num_t +1
        The super operator forward propagators calculated in the systems
        simulation.

    unitary_derivatives: List[List[ControlMatrix]],
                         shape: [[] * num_t] * num_ctrl
        Frechet derivatives of the propagators by the control amplitudes.

    target_unitary: ControlMatrix
        The target unitary evolution.

    reversed_propagators: List[ControlMatrix]
        The reversed propagators calculated in the systems simulation.

    computational_states: Optional[List[int]]
        If set, the entanglement fidelity is only calculated for the specified
        subspace.s only calculated for the specified
        subspace.

    map_to_closest_unitary: bool
        If True, then the final propagator is mapped to the closest unitary
        before the infidelity is evaluated.

    Returns
    -------
    derivative_fidelity: np.ndarray, shape: (num_t, num_ctrl)
        The derivatives of the entanglement fidelity.

    """
    num_ctrls = len(unitary_derivatives)
    num_time_steps = len(unitary_derivatives[0])

    derivative_fidelity = np.zeros(shape=(num_time_steps, num_ctrls),
                                   dtype=float)

    for ctrl in range(num_ctrls):
        for t in range(num_time_steps):
            # here we need to take the real part.
            derivative_fidelity[t, ctrl] = \
                entanglement_fidelity_super_operator(
                    target_unitary=target_unitary,
                    propagator=reversed_propagators[::-1][t + 1] *
                    unitary_derivatives[ctrl][t] *
                    forward_propagators[t],
                    computational_states=computational_states,
                    map_to_closest_unitary=map_to_closest_unitary)
    return derivative_fidelity


class OperationInfidelity(CostFunction):
    """
    Operator fidelity computer.

    Possible fidelity measures are currently only the entanglement fidelity for
    unitary evolutions and for propagators in the super operator formalism.

    Paramters
    ---------
    t_slot_comp: TimeSlotComputer
        The time slot computer simulating the systems dynamics.

    target: ControlMatrix
        Unitary target evolution.

    use_unitary_derivatives: bool
        If True then the derivatives of the propagators calculated by the time
        slot computer are used. Otherwise the grape approximation is applied.

    fidelity_measure: string
        If 'entanglement': the entanglement fidelity is calculated.
        Otherwise an error is raised.

    super_operator_formalism: bool
        If true, the time slot computer is expected to return a propagator in
        the super operator formalism, while the target unitary is not given as
        super operator.
        If false, no super operators are assumed.

    computational_states: Optional[List[int]]
        If set, the chosen fidelity is only calculated for the specified
        subspace.

    map_to_closest_unitary: bool
        If True, then the final propagator is mapped to the closest unitary
        before the infidelity is evaluated.

    Attributes
    ----------
    t_slot_comp: TimeSlotComputer
        The time slot computer simulating the systems dynamics.

    target: ControlMatrix
        Unitary target evolution.

    fidelity_measure: string
        If 'entanglement': the entanglement fidelity is calculated.
        Otherwise an error is raised.

    super_operator_formalism: bool
        If true, the time slot computer is expected to return a propagator in
        the super operator formalism, while the target unitary is not given as
        super operator.
        If false, no super operators are assumed.

    Raises
    ------
    NotImplementedError
        If the fidelity measure is not 'entanglement'.

    Todo:
        * add the average fidelity? or remove the fidelity_measure.

    """
    def __init__(self,
                 t_slot_comp: solver_algorithms.Solver,
                 target: matrix.OperatorMatrix,
                 fidelity_measure: str = 'entanglement',
                 super_operator_formalism: bool = False,
                 index: Optional[List[str]] = None,
                 computational_states: Optional[List[int]] = None,
                 map_to_closest_unitary: bool = False
                 ):
        super().__init__()
        self.t_slot_comp = t_slot_comp
        self.target = target
        self.computational_states = computational_states
        self.map_to_closest_unitary = map_to_closest_unitary
        if fidelity_measure == 'entanglement':
            self.fidelity_measure = fidelity_measure
        else:
            raise NotImplementedError('Only the entanglement fidelity is '
                                      'currently supported.')
        if index is not None:
            self.index = index
        elif fidelity_measure == 'entanglement':
            self.index = ['Entanglement Infidelity', ]
        else:
            self.index = ['Operator Infidelity', ]
        self.super_operator = super_operator_formalism

    def costs(self):
        """Calculates the costs by the selected fidelity measure. """
        final = self.t_slot_comp.forward_propagators[-1]

        if self.fidelity_measure == 'entanglement' and self.super_operator:
            infid = 1 - entanglement_fidelity_super_operator(
                propagator=final,
                target_unitary=self.target,
                computational_states=self.computational_states,
                map_to_closest_unitary=self.map_to_closest_unitary
            )
        elif self.fidelity_measure == 'entanglement':
            infid = 1 - entanglement_fidelity(
                unitary=final,
                target_unitary=self.target,
                computational_states=self.computational_states,
                map_to_closest_unitary=self.map_to_closest_unitary
            )
        else:
            raise NotImplementedError('Only the entanglement fidelity is '
                                      'implemented in this version.')
        return np.real(infid)

    def grad(self):
        """Calculates the derivatives of the selected fidelity measure with
        respect to the control amplitudes. """
        if self.fidelity_measure == 'entanglement' and self.super_operator:
            derivative_fid = deriv_entanglement_fid_sup_op_with_du(
                forward_propagators=self.t_slot_comp.forward_propagators,
                target_unitary=self.target,
                reversed_propagators=self.t_slot_comp.reversed_propagators,
                unitary_derivatives=self.t_slot_comp.frechet_deriv_propagators
            )
        elif self.fidelity_measure == 'entanglement':
            derivative_fid = derivative_entanglement_fidelity_with_du(
                forward_propagators=self.t_slot_comp.forward_propagators,
                target_unitary=self.target,
                reversed_propagators=self.t_slot_comp.reversed_propagators,
                unitary_derivatives=self.t_slot_comp.frechet_deriv_propagators
            )
        else:
            raise NotImplementedError('Only the average and entanglement'
                                      'fidelity is implemented in this '
                                      'version.')
        return -1 * np.real(derivative_fid)


class OperationNoiseInfidelity(CostFunction):
    """
    Averages the operator fidelity over noise traces.

    Parameters
    ----------
    t_slot_comp: TSCompSaveAllNoise
        Time slot computer sub class handling noise sources by explicitly
        sampling the noise sources.

    neglect_systematic_errors: bool
        If true, the mean operator fidelity is calculated with respect to the
        simulated propagator without statistical noise.
        Otherwise the mean operator fidelity is calculated with respect to the
        target propagator.

    computational_states: Optional[List[int]]
        If set, the chosen fidelity is only calculated for the specified
        subspace.

    map_to_closest_unitary: bool
        If True, then the final propagator is mapped to the closest unitary
        before the infidelity is evaluated.

    Attributes
    ----------
    neglect_systematic_errors: bool
        If true, the standard deviation of the operator fidelity is measured.
        Otherwise the mean operator fidelity is calculated with respect to the
        target propagator.


    Todo
        * raise warning when no target and not neglect_systematic_errors

    """
    def __init__(self,
                 t_slot_comp: solver_algorithms.SchroedingerSMonteCarlo,
                 target: Optional[matrix.OperatorMatrix],
                 index: Optional[List[str]] = None,
                 neglect_systematic_errors=True,
                 fidelity_measure='entanglement',
                 computational_states: Optional[List[int]] = None,
                 map_to_closes_unitary: bool = False):
        super().__init__()
        self.t_slot_comp = t_slot_comp
        self.target = target
        self.computational_states = computational_states

        if index is not None:
            self.index = index
        else:
            self.index = ['Operator Noise Infidelity']

        self.fidelity_measure = fidelity_measure

        self.neglect_systematic_errors = neglect_systematic_errors
        if target is None and not neglect_systematic_errors:
            print('The systematic errors must be neglected if no target is '
                  'set!')
            self.neglect_systematic_errors = True

        self.map_to_closest_unitary = map_to_closes_unitary

    def costs(self):
        """See base class. """
        n_traces = self.t_slot_comp.noise_trace_generator.n_traces
        infidelities = np.zeros((n_traces,))

        if self.neglect_systematic_errors:
            if self.computational_states is None:
                target = self.t_slot_comp.forward_propagators[-1]
            else:
                target = self.t_slot_comp.forward_propagators[
                    -1].truncate_to_subspace(
                    self.computational_states,
                    map_to_closest_unitary=self.map_to_closest_unitary
                )
        else:
            target = self.target

        if self.fidelity_measure == 'entanglement':
            for i in range(n_traces):
                final = self.t_slot_comp.forward_propagators_noise[i][-1]

                infidelities[i] = 1 - entanglement_fidelity(
                    unitary=final, target_unitary=target,
                    computational_states=self.computational_states,
                    map_to_closest_unitary=self.map_to_closest_unitary
                )
        else:
            raise NotImplementedError('Only the entanglement fidelity is '
                                      'currently implemented in this class.')

        return np.mean(np.real(infidelities))

    def grad(self):
        """See base class. """
        if self.neglect_systematic_errors:
            target = self.t_slot_comp.forward_propagators[-1]
        else:
            target = self.target

        n_traces = self.t_slot_comp.noise_trace_generator.n_traces
        num_t = len(self.t_slot_comp.tau)
        num_ctrl = len(self.t_slot_comp.h_ctrl)
        derivative = np.zeros((num_t, num_ctrl, n_traces, ))
        for i in range(n_traces):
            temp = derivative_entanglement_fidelity_with_du(
                target_unitary=target,
                forward_propagators=
                self.t_slot_comp.forward_propagators_noise[i],
                unitary_derivatives=
                self.t_slot_comp.frechet_deriv_propagators_noise[i],
                reversed_propagators=
                self.t_slot_comp.reversed_propagators_noise[i],
                computational_states=self.computational_states
                )
            if self.neglect_systematic_errors:
                temp += derivative_entanglement_fidelity_with_du(
                    target_unitary=self.t_slot_comp.forward_propagators_noise[
                        i][-1],
                    forward_propagators=
                    self.t_slot_comp.forward_propagators,
                    unitary_derivatives=
                    self.t_slot_comp.frechet_deriv_propagators,
                    reversed_propagators=
                    self.t_slot_comp.reversed_propagators,
                    computational_states=self.computational_states
                )
                # TODO: This could be calculated more efficiently, when the
                #  calculation of the derivative of the unitary is separated
                #  from the calculation of the derivative of the fidelity.
            derivative[:, :, i] = np.real(temp)
        return np.mean(-derivative, axis=2)


class OperatorFilterFunctionInfidelity(CostFunction):
    """
    Calculates the infidelity with the filter function formalism.

    Parameters
    ----------
    noise_power_spec_density: Union[Sequence[float], Callable]
        The two-sided noise power spectral density in units of inverse
        frequencies as an array of shape (n_omega,), (n_nops, n_omega), or
        (n_nops, n_nops, n_omega). In the first case, the same spectrum is
        taken for all noise operators, in the second, it is assumed that there
        are no correlations between different noise sources and thus there is
        one spectrum for each noise operator. In the third and most general
        case, there may be a spectrum for each pair of noise operators
        corresponding to the correlations between them. n_nops is the number of
        noise operators considered and should be equal to
        ``len(n_oper_identifiers)``.

    omega: Union[Sequence[float], Dict[str, Union[int, str]]]
        The frequencies at which the integration is to be carried out. If
        *test_convergence* is ``True``, a dict with possible keys ('omega_IR',
        'omega_UV', 'spacing', 'n_min', 'n_max', 'n_points'), where all
        entries are integers except for ``spacing`` which should be a string,
        either 'linear' or 'log'. 'n_points' controls how many steps are taken.
        Note that the frequencies are assumed to be symmetric about zero.

    """
    def __init__(self,
                 t_slot_comp: solver_algorithms.Solver,
                 noise_power_spec_density: Union[Sequence[float], Callable],
                 omega: Union[Sequence[float], Dict[str, Union[int, str]]],
                 index=('Infidelity Filter Function', )):
        super().__init__()
        self.t_slot_comp = t_slot_comp
        self.noise_power_spec_density = noise_power_spec_density
        self.omega = omega
        if index is None:
            self.index = ['Infidelity Filter Function', ]
        else:
            self.index = index

    def costs(self) -> Union[float, np.ndarray]:
        """
        The infidelity is calculated with the filter function package. See its
        documentation for more information.

        Returns
        -------
        costs: Union[float, np.ndarray]
            The infidelity.

        """
        if self.t_slot_comp.pulse_sequence is None:
            self.t_slot_comp.create_pulse_sequence()
        infidelity = filter_functions.numeric.infidelity(
            pulse=self.t_slot_comp.pulse_sequence,
            S=self.noise_power_spec_density,
            omega=self.omega)
        return infidelity

    def grad(self):
        """
        Not implemented in the current version.

        Raises
        ------
        NotImplementedError
            This method has not been implemented yet.

        """
        raise NotImplementedError('The gradient calculation is not implemented '
                                  'for the filter functions.')


class LeakageError(CostFunction):
    """This class measures leakage as quantum operation error.

    The resulting infidelity is measured by truncating the leakage states of the
    propagator U yielding the Propagator V on the computational basis. The
    infidelity is then given as the distance from unitarity:
        infid = 1 - trace(V^\dag V) / 4

    Parameters
    ----------
    t_slot_comp : TimeSlotComputer
        The time slot computer computing the propagation of the system.

    computational_states : list of int
        List of indices marking the computational states of the propagator.
        These are all but the leakage states.

    """
    def __init__(self, t_slot_comp: solver_algorithms.Solver,
                 computational_states: List[int],
                 index: Optional[List[str]] = None):
        super().__init__()
        self.t_slot_comp = t_slot_comp
        self.computational_states = computational_states
        if index is None:
            self.index = ["Leakage Error", ]
        else:
            self.index = index

    def costs(self):
        """See base class. """
        final_prop = self.t_slot_comp.forward_propagators[-1]
        clipped_prop = final_prop.truncate_to_subspace(
            self.computational_states)
        temp = clipped_prop.dag(copy_=True)
        temp *= clipped_prop

        return 1 - temp.tr().real / clipped_prop.shape[0]

    def grad(self):
        """See base class. """
        num_ctrls = len(self.t_slot_comp.frechet_deriv_propagators)
        num_time_steps = len(self.t_slot_comp.frechet_deriv_propagators[0])
        d = self.t_slot_comp.propagators[-1].shape[0]

        derivative_fidelity = np.zeros(shape=(num_time_steps, num_ctrls),
                                       dtype=np.float64)

        final = self.t_slot_comp.forward_propagators[-1]
        final_leak_dag = final.dag(copy_=True).truncate_to_subspace(
            self.computational_states)

        for ctrl in range(num_ctrls):
            for t in range(num_time_steps):
                temp = self.t_slot_comp.reversed_propagators[::-1][t + 1] \
                     * self.t_slot_comp.frechet_deriv_propagators[ctrl][t]
                temp *= self.t_slot_comp.forward_propagators[t]
                temp = temp.truncate_to_subspace(self.computational_states)
                temp *= final_leak_dag
                derivative_fidelity[t, ctrl] = -2. / d * temp.tr().real
        return derivative_fidelity


@deprecated
def derivative_entanglement_fidelity(
        control_hamiltonians: List[matrix.OperatorMatrix],
        forward_propagators: List[matrix.OperatorMatrix],
        reversed_propagators: List[matrix.OperatorMatrix],
        delta_t: List[float],
        target_unitary: matrix.OperatorMatrix) -> np.ndarray:
    """
    Derivative of the entanglement fidelity using the grape approximation.

    dU / du = -i delta_t H_ctrl U

    Parameters
    ----------
    control_hamiltonians: List[ControlMatrix], len: num_ctrl
        The control hamiltonians of the simulation.

    forward_propagators: List[ControlMatrix], len: num_t +1
        The forward propagators calculated in the systems simulation.

    reversed_propagators: List[ControlMatrix]
        The reversed propagators calculated in the systems simulation.

    delta_t: List[float], len: num_t
        The durations of the time steps.

    target_unitary: ControlMatrix
        The target unitary evolution.

    Returns
    -------
    derivative_fidelity: np.ndarray, shape: (num_t, num_ctrl)
        The derivatives of the entanglement fidelity.

    """
    target_unitary_dag = target_unitary.dag(copy_=True)
    trace = np.conj(((forward_propagators[-1] * target_unitary_dag).tr()))
    num_ctrls = len(control_hamiltonians)
    num_time_steps = len(delta_t)
    d = target_unitary.shape[0]

    derivative_fidelity = np.zeros(shape=(num_time_steps, num_ctrls),
                                   dtype=complex)

    for ctrl in range(num_ctrls):
        for t in range(num_time_steps):
            # we take the imaginary part because we took a factor of i out
            derivative_fidelity[t, ctrl] = 2 / d / d * delta_t * np.imag(
                trace * (reversed_propagators[::-1][t + 1]
                         * control_hamiltonians[ctrl]
                         * forward_propagators[t + 1]
                         * target_unitary_dag).tr())
    return derivative_fidelity


@needs_refactoring
def averge_gate_fidelity(unitary: matrix.OperatorMatrix,
                         target_unitary: matrix.OperatorMatrix):
    """
    Average gate fidelity.

    Parameters
    ---------
    unitary: ControlMatrix
        The evolution matrix of the system.

    target_unitary: ControlMatrix
        The target unitary to which the evolution is compared.


    Returns
    -------
    fidelity: float
        The average gate fidelity.

    """

    dim = unitary.shape[0]
    orthogonal_operators = default_set_orthorgonal(dim=dim)
    temp = unitary.dag(copy_=True) * target_unitary

    temp = [ort.dag(copy_=True) * temp.dag(copy_=True) * ort * temp
            for ort in orthogonal_operators]

    fidelity = temp[0]
    for i in range(1, dim ** 2):
        fidelity += temp[i]
    fidelity = (fidelity.tr() + dim ** 2) / (dim ** 2 * (dim + 1))
    return fidelity


@needs_refactoring
def default_set_orthorgonal(dim: int) -> List[matrix.OperatorMatrix]:
    """
    Set of orthogonal matrices for the calculation of the average gate fidelity.

    Currently only for two dimensional systems implemented.

    Parameters
    ----------
    dim: int
        The systems dimension.

    Returns
    -------
    orthogonal_operators: List[ControlMatrix]
        Orthogonal control matrices.

    """

    sigma_x = np.asarray([[0, 1], [1, 0]])
    sigma_y = np.asarray([[1, 0], [0, -1]])
    sigma_z = np.asarray([[0, -1j], [1j, 0]])

    if dim == 2:
        orthogonal_operators = [sigma_x, sigma_y, sigma_z, np.eye(2)]
        orthogonal_operators = [matrix.DenseOperator(mat) for mat
                                in orthogonal_operators]
    else:
        raise NotImplementedError("Please implement a set of orthogonal "
                                  "operators for this dimension.")

    return orthogonal_operators


@deprecated
def derivative_average_gate_fidelity(control_hamiltonians, propagators,
                                     propagators_past, delta_t, target_unitary):
    """
    The derivative of the average gate fidelity.
    """
    unity = matrix.DenseOperator(
        np.eye(propagators[0].data.shape[0]))
    propagators_future = [unity]
    for prop in propagators[::-1]:
        propagators_future.append(propagators_future[-1] * prop)
    propagators_future = propagators_future[::-1]
    dim = control_hamiltonians[0, 0].shape[0]
    orthogonal_operators = default_set_orthorgonal(dim=dim)

    num_time_steps, num_ctrls = control_hamiltonians.shape

    derivative_fidelity = np.zeros(shape=control_hamiltonians.shape,
                                   dtype=complex)
    for ctrl in range(num_ctrls):
        for t in range(num_time_steps):
            bkwd_prop_target = propagators_future[t+1].dag() * target_unitary
            temp = 0
            for ort in orthogonal_operators:
                lambda_ = bkwd_prop_target * ort.dag(copy_=True)
                lambda_ *= bkwd_prop_target.dag()
                rho = propagators_past[t+1] * ort
                rho *= propagators_past[t+1].dag()
                # everything rewritten to operate in place
                temp_mat2 = control_hamiltonians[t, ctrl] * rho
                temp_mat2 -= rho * control_hamiltonians[t, ctrl]
                temp_mat = lambda_
                temp_mat *= -1j
                temp_mat *= delta_t
                temp_mat *= temp_mat2
                temp += temp_mat.tr()
                # temp += (lambda_ * -1j * delta_t * (
                #         control_hamiltonians[t, ctrl] * rho
                #         - rho * control_hamiltonians[t, ctrl])).tr()
            derivative_fidelity[t, ctrl] = temp / (dim ** 2 * (dim + 1))
    return derivative_fidelity


@needs_refactoring
def derivative_average_gate_fid_with_du(propagators, propagators_past,
                                        unitary_derivatives, target_unitary):
    unity = matrix.DenseOperator(
        np.eye(propagators[0].data.shape[0]))
    propagators_future = [unity]
    for prop in propagators[::-1]:
        propagators_future.append(propagators_future[-1] * prop)
    propagators_future = propagators_future[::-1]
    dim = propagators[0].shape[0]
    orthogonal_operators = default_set_orthorgonal(dim=dim)

    num_time_steps, num_ctrls = unitary_derivatives.shape

    derivative_fidelity = np.zeros(shape=unitary_derivatives.shape,
                                   dtype=complex)
    for ctrl in range(num_ctrls):
        for t in range(num_time_steps):
            bkwd_prop_target = propagators_future[t + 1].dag() * target_unitary
            temp = 0
            for ort in orthogonal_operators:
                lambda_ = bkwd_prop_target * ort.dag()
                lambda_ *= bkwd_prop_target.dag()
                rho = propagators_past[t] * ort
                rho *= propagators_past[t + 1].dag()
                # everything rewritten to operate in place
                temp_mat2 = unitary_derivatives[t, ctrl] * rho
                # here we assume, that the orthogonal operators are self
                # adjoined
                temp_mat2 += rho.dag() * unitary_derivatives[t, ctrl].dag()
                lambda_ *= temp_mat2
                temp += lambda_.tr()
            derivative_fidelity[t, ctrl] = temp / (dim ** 2 * (dim + 1))
    return derivative_fidelity
