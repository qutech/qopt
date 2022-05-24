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
"""
Cost functions which can be minimised as figure of merit in the control
optimization.

Each `CostFunction` calculates a commonly used quantity for errors occurring
during a quantum gate or algorithm. These include state and gate fidelities or
leakages. These are also implemented in variations for the description of noise
like the averaging in a Monte Carlo method or the compatibility with a
linearized master equation in lindblad form. One cost function interfaces to
the estimation of infidelities by generalized filter functions.


To support gradient based optimization algorithms such as quasi-Newton type
algorithms the classes also calculate the gradients of the cost functions.
(Jacobians in case of vector valued cost functions.)

Classes
-------
:class:`CostFunction`
    Abstract base class of the fidelity computer.

:class:`OperatorMatrixNorm`
    Calculates the cost as matrix norm of the difference between the actual
    evolution and the target.

:class:`OperationInfidelity`
    Calculates the cost as operation infidelity of a propagator.

:class:`OperationNoiseInfidelity`
    Like Operationfidelity but averaged over noise traces.

:class:`OperatorFilterFunctionInfidelity`
    Estimates infidelities with filter functions.

:class:`LeakageError`
    Estimates the leakage of quantum gates.

Functions
---------
:func:`state_fidelity`
    The quantum state fidelity.

:func:`angle_axis_representation`
    Calculates the representation of a 2x2 unitary matrix as rotation axis and
    angle.

:func:`entanglement_fidelity`
    Calculates the entanglement fidelity between a unitary target evolution and
    a simulated unitary evolution.

:func:`deriv_entanglement_fid_sup_op_with_du`
    Calculates the derivatives of the entanglement fidelity with respect to
    the control amplitudes.

:func:`entanglement_fidelity_super_operator`
    Calculates the entanglement fidelity between two propagators in the super
    operator formalism.

:func:`derivative_entanglement_fidelity_with_du`
    Calculates the derivatives of the entanglement fidelity in the super
    operator formalism with respect to the control amplitudes.

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

from abc import ABC, abstractmethod
from typing import Sequence, Union, List, Optional, Callable

import filter_functions.numeric

from qopt import matrix, solver_algorithms
from qopt.util import needs_refactoring, deprecated
from qopt.matrix import ket_vectorize_density_matrix, \
    convert_ket_vectorized_density_matrix_to_square, \
    convert_unitary_to_super_operator


class CostFunction(ABC):
    r"""
    Abstract base class of the fidelity computer.

    Attributes
    ----------
    solver : `Solver`
        Object that compute the forward/backward evolution and propagator.

    label: list of str
        The label serves as internal name of the cost function values. The
        DataContainer class uses the label to distinct cost functions when
        storing the data.

    """

    def __init__(self, solver: solver_algorithms.Solver,
                 label: Optional[List[str]] = None):
        self.solver = solver
        if label is None:
            self.label = ["Unspecified Cost Function"]
        else:
            self.label = label

    @abstractmethod
    def costs(self) -> Union[float, np.ndarray]:
        """Evaluates the cost function.

        Returns
        -------
        costs : np.array or float
            Result of the cost function's evaluation.

        """
        pass

    @abstractmethod
    def grad(self) -> np.ndarray:
        """Calculates the gradient or Jacobian of the cost function.

        Returns
        -------
        gradient : np.array
            shape: (num_t, num_ctrl, num_f) where num_t is the number of time
            slices, num_ctrl the number of control parameters and num_f the
            number of values returned by the cost function. Derivatives of
            the cost function by the control amplitudes.

        """
        pass


def angle_axis_representation(u: np.ndarray) \
        -> (float, np.ndarray):
    """
    Calculates the representation of a 2x2 unitary matrix by a rotational axis
    and a rotation angle.

    Parameters
    ----------
    u: np.ndarray
        A unitary 2x2 matrix.

    Returns
    -------
    beta, n: float, np.ndarray
        beta is the angle of the rotation and n the rotational axis.

    """
    if type(u) == matrix.DenseOperator:
        u = np.copy(u.data)

    # check if u is unitary
    ident = u @ np.conjugate(np.transpose(u))
    is_unitary = np.isclose(ident[0, 0], 1) \
                 and np.isclose(ident[1, 0], 0) \
                 and np.isclose(ident[0, 1], 0) \
                 and np.isclose(ident[0, 0], 1)

    if not is_unitary:
        raise ValueError("Your input matrix must be unitary to calculate a "
                         "angle axis representation!")

    # there is an unphysical global phase alpha
    cos_alpha = .5 * (u[0, 0] + u[1, 1])
    # beta in [0, pi) so sin in [0, 1]
    sin = np.sqrt(1 - np.abs(cos_alpha) ** 2)
    if np.isclose(0, sin, atol=1e-6):
        return 0, np.array([1, 0, 0])
    n_1_alpha = (u[0, 1] + u[1, 0]) / 1j / sin / 2
    n_2_alpha = (u[0, 1] - u[1, 0]) / sin / 2
    n_3_alpha = (u[0, 0] - u[1, 1]) / 1j / sin / 2
    if not np.isclose(n_1_alpha, 0):
        alpha = n_1_alpha / np.abs(n_1_alpha)
    elif not np.isclose(n_2_alpha, 0):
        alpha = n_2_alpha / np.abs(n_2_alpha)
    elif not np.isclose(n_3_alpha, 0):
        alpha = n_3_alpha / np.abs(n_3_alpha)
    else:
        return 0, np.array([1, 0, 0])

    beta = np.arccos(np.real(cos_alpha / alpha)) * 2
    n_1, n_2, n_3 = n_1_alpha / alpha, n_2_alpha / alpha, n_3_alpha / alpha
    assert np.isclose(np.linalg.norm(np.array([n_1, n_2, n_3])), 1, atol=1e-5)

    # to make this representation unique, we request that beta in [0, pi]
    if beta > np.pi:
        beta = 2 * np.pi - beta
        n_1, n_2, n_3 = -1 * n_1, -1 * n_2, -1 * n_3
    return beta, np.array([np.real(n_1), np.real(n_2), np.real(n_3)])


class OperatorMatrixNorm(CostFunction):
    """
    Computes the fidelity as difference between the propagator and a target.

    A global phase difference is ignored. The result can be returned as
    absolute value or vector. If the result shall be returned as absolute value
    it is calculated in a matrix norm.

    Parameters
    ----------
    solver: TimeSlotComputer
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
    def __init__(self, solver: solver_algorithms.Solver,
                 target: matrix.OperatorMatrix, mode: str = 'scalar',
                 label: Optional[List[str]] = None):
        super().__init__()
        self.solver = solver
        self.target = target
        self.mode = mode
        if label is not None:
            self.label = label
        elif mode == 'scalar':
            self.label = ['Matrix Norm Distance']
        elif mode == 'vector':
            dim = target.shape[0]
            self.label = ['redu' + str(i) + str(j)
                          for i in range(1, dim + 1)
                          for j in range(1, dim + 1)] + [
                             'imdu' + str(i) + str(j)
                             for i in range(1, dim + 1)
                             for j in range(1, dim + 1)]
        elif mode == 'rotation_axis':
            self.label = ['n1 * phi', 'n2', 'n3']
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
        final = self.solver.forward_propagators[-1]

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
        propagators = self.solver.propagators
        forward_prop_cumulative = self.solver.forward_propagators
        # reversed_prop_cumulative = self.t_slot_comp.reversed_prop_cumulative
        unity = matrix.DenseOperator(
            np.eye(propagators[0].data.shape[0]))
        propagators_future = [unity]
        for prop in propagators[:0:-1]:
            propagators_future.append(propagators_future[-1] * prop)
        propagators_future = propagators_future[::-1]

        if isinstance(self.solver.transferred_time, list):
            tau = self.solver.transferred_time[0]
        elif isinstance(self.solver.transferred_time, float):
            tau = self.solver.transferred_time
        else:
            raise NotImplementedError

        num_t = len(self.solver.transferred_time)
        num_ctrl = len(self.solver.h_ctrl)
        jacobian_complex_full = np.zeros(
            shape=[self.target.data.size, num_t,
                   num_ctrl]).astype(complex)
        final = self.solver.forward_propagators[-1]
        exp_iphi = final[0, 0] / np.abs(final[0, 0])
        # * 2 for the seperation of imaginary and real part

        for j in range(num_ctrl):
            for i, (prop, fwd_prop, future_prop) in enumerate(
                    zip(propagators, forward_prop_cumulative,
                        propagators_future)):
                # here i applied the grape approximations
                complex_jac = (
                        -1j * tau * future_prop * self.solver.h_ctrl[j]
                        * fwd_prop).flatten()
                jacobian_complex_full[:, i, j] = complex_jac.data

        dphi_by_du = (
                             np.imag(jacobian_complex_full[0, :, :]) * np.real(
                         final[0, 0]) -
                             np.real(jacobian_complex_full[0, :, :]) * np.imag(
                         final[0, 0])
                     ) / ((np.abs(final[0, 0])) ** 2)
        final.flatten()

        dphi_by_du_times_u = np.concatenate \
            ([np.reshape(dphi_by_du, (1, dphi_by_du.shape[0],
                                      dphi_by_du.shape[1])) * fin
              for fin in final.data])

        jacobian_complex_full = (jacobian_complex_full -
                                 1j * dphi_by_du_times_u) * (1 / exp_iphi)

        # The result must be corrected by a sign depending on the angle phi

        jacobian = np.concatenate([np.real(jacobian_complex_full),
                                   np.imag(jacobian_complex_full)], axis=0)
        return jacobian


def state_fidelity(
        target: Union[np.ndarray, matrix.OperatorMatrix],
        propagated_state: Union[np.ndarray, matrix.OperatorMatrix],
        computational_states: Optional[List[int]] = None,
        rescale_propagated_state: bool = False
) -> np.float64:
    r"""
    Quantum state fidelity.

    The quantum state fidelity between two quantum states is calculated as
    square norm of the wave function overlap as

    .. math::

        F = \vert \langle \psi_1 \vert \psi_2 \rangle \vert^2

    Parameters
    ----------
    target: numpy array or operator matrix of shape (1, d)
        The target state is assumed to be given as bra-vector.

    propagated_state: numpy array or operator matrix of shape (d, 1)
        The target state is assumed to be given as ket-vector.

    computational_states: Optional[List[int]]
        If set, the entanglement fidelity is only calculated for the specified
        subspace.

    rescale_propagated_state: bool
        If True, then the propagated state vector is rescaled to a norm of 1.

    Returns
    -------
    quantum_state_fidelity: float
        The quantum state fidelity between the propagated and the target state.

    TODO:
        * functions should not change type of input arrays

    """
    if type(target) == np.ndarray:
        target = matrix.DenseOperator(target)
    if type(propagated_state) == np.ndarray:
        propagated_state = matrix.DenseOperator(propagated_state)

    if computational_states is not None:
        scalar_prod = target * propagated_state.truncate_to_subspace(
            computational_states,
            map_to_closest_unitary=rescale_propagated_state
        )
    else:
        scalar_prod = target * propagated_state

    if scalar_prod.shape != (1, 1):
        raise ValueError('The scalar product is not a scalar. This means that'
                         'either the target is not a bra vector or the the '
                         'propagated state not a ket, or both!')
    scalar_prod = scalar_prod[0, 0]
    abs_sqr = scalar_prod.real ** 2 + scalar_prod.imag ** 2
    return abs_sqr


def derivative_state_fidelity(
        target: matrix.OperatorMatrix,
        forward_propagators: List[matrix.OperatorMatrix],
        propagator_derivatives: List[List[matrix.OperatorMatrix]],
        reversed_propagators: List[matrix.OperatorMatrix],
        computational_states: Optional[List[int]] = None,
        rescale_propagated_state: bool = False
) -> np.ndarray:
    """
    Derivative of the state fidelity.

    Leakage states can be defined and the propagator is projected onto the
    computational states.

    Parameters
    ----------
    target: OperatorMatrix
        The target state as bra vector.

    forward_propagators: list of OperatorMatrix
        Forward propagated initial state.

    propagator_derivatives: list of OperatorMatrix
        Frechet derivatives of the matrix exponential used to calculate the
        propagators.

    reversed_propagators: list of OperatorMatrix
        Backward passed propagators.

    computational_states: list of int
        States used for the qubit implementation. If this is not None, then all
        other inides are eliminated, by projection into the computational
        space.

    rescale_propagated_state: bool
        If set to Ture, then the propagated state is rescaled after leakage
        states are eliminated.

    Returns
    -------
    Derivative: numpy array, shape: (num_time_steps, num_ctrls)
        The derivatives of the state fidelity by the

    """

    if computational_states is not None:
        scalar_prod = target * forward_propagators[-1].truncate_to_subspace(
            subspace_indices=computational_states,
            map_to_closest_unitary=rescale_propagated_state
        )
    else:
        scalar_prod = target * forward_propagators[-1]

    scalar_prod = np.conj(scalar_prod)

    num_ctrls = len(propagator_derivatives)
    num_time_steps = len(propagator_derivatives[0])

    derivative_fidelity = np.zeros(shape=(num_time_steps, num_ctrls),
                                   dtype=float)
    for ctrl in range(num_ctrls):
        for t in range(num_time_steps):
            # here we need to take the real part.
            if computational_states:
                derivative_fidelity[t, ctrl] = 2 * np.real(
                    (scalar_prod * (
                            target * (
                            reversed_propagators[::-1][t + 1]
                            * propagator_derivatives[ctrl][t]
                            * forward_propagators[t]
                    ).truncate_to_subspace(
                        subspace_indices=computational_states,
                        map_to_closest_unitary=rescale_propagated_state
                    )
                    ))[0, 0])
            else:
                derivative_fidelity[t, ctrl] = 2 * np.real(
                    (scalar_prod * (target * reversed_propagators[::-1][t + 1]
                                    * propagator_derivatives[ctrl][t]
                                    * forward_propagators[t]))[0, 0])

    return derivative_fidelity


def state_fidelity_subspace(
        target: Union[np.ndarray, matrix.OperatorMatrix],
        propagated_state: Union[np.ndarray, matrix.OperatorMatrix],
        dims: List[int],
        remove: List[int]
) -> np.float64:
    r"""
    Quantum state fidelity on a subspace.

    We assume that the target state is defined only on a subspace of the total
    simulated hilbert space. Thus we calculate the partial trace over our
    simulated state, rendering it into the density matrix of a potentially
    mixed state.

    The quantum state fidelity between a pure $\psi$ and a mixed quantum state
    $\rho$ is calculated as

    .. math::

        F =  \langle \psi \vert \rho \vert \psi \rangle

    Parameters
    ----------
    target: numpy array or operator matrix of shape (1, d)
        The target state is assumed to be given as bra-vector.

    propagated_state: numpy array or operator matrix
        The target state is assumed to be given as density matrix of
        shape(d, d) or shape (d^2, 1), or as ket-vector of shape (d, 1).

    dims: list of int,
        The dimensions of the subspaces. (Compare to the ptrace function of
        the MatrixOperator class.)

    remove: list of int,
        The indices of the dims list corresponding to the subspaces that are to
        be eliminated. (Compare to the ptrace function of the MatrixOperator
        class.)

    Returns
    -------
    quantum_state_fidelity: float
        The quantum state fidelity between the propagated and the target state.

    TODO:
        * functions should not change type of input arrays

    """
    if type(target) == np.ndarray:
        local_target = matrix.DenseOperator(target)
    else:
        local_target = target
    if type(propagated_state) == np.ndarray:
        local_propagated_state = matrix.DenseOperator(propagated_state)
    else:
        local_propagated_state = propagated_state

    leakage_total_dimension = 1
    for ind in remove:
        leakage_total_dimension *= dims[ind]

    if local_propagated_state.data.size >= \
            (local_target.data.size * leakage_total_dimension) ** 2:
        # propagated state given as density matrix
        if local_propagated_state.shape[1] == 1:
            # the density matrix is vectorized
            local_propagated_state = \
                convert_ket_vectorized_density_matrix_to_square(
                    local_propagated_state)

    rho = local_propagated_state.ptrace(dims=dims, remove=remove)

    scalar_prod = local_target * rho * local_target.dag()

    if scalar_prod.shape != (1, 1):
        raise ValueError('The scalar product is not a scalar. This means that'
                         'either the target is not a bra vector or the the '
                         'propagated state not a ket, or both!')
    scalar_prod = scalar_prod[0, 0]
    scalar_prod_real = scalar_prod.real
    assert np.abs(scalar_prod - scalar_prod_real) < 1e-5
    return scalar_prod_real


def derivative_state_fidelity_subspace(
        target: matrix.OperatorMatrix,
        forward_propagators: List[matrix.OperatorMatrix],
        propagator_derivatives: List[List[matrix.OperatorMatrix]],
        reversed_propagators: List[matrix.OperatorMatrix],
        dims: List[int],
        remove: List[int]
) -> np.ndarray:
    """
    Derivative of the state fidelity on a subspace.

    The unused subspace is traced out.

    Parameters
    ----------
    target: OperatorMatrix
        The target state as bra vector.

    forward_propagators: list of OperatorMatrix
        Forward propagated initial state.

    propagator_derivatives: list of OperatorMatrix
        Frechet derivatives of the matrix exponential used to calculate the
        propagators.

    reversed_propagators: list of OperatorMatrix
        Backward passed propagators.

    dims: list of int,
        The dimensions of the subspaces. (Compare to the ptrace function of
        the MatrixOperator class.)

    remove: list of int,
        The indices of the dims list corresponding to the subspaces that are to
        be eliminated. (Compare to the ptrace function of the MatrixOperator
        class.)

    Returns
    -------
    Derivative: numpy array, shape: (num_time_steps, num_ctrls)
        The derivatives of the state fidelity by the

    """

    num_ctrls = len(propagator_derivatives)
    num_time_steps = len(propagator_derivatives[0])

    derivative_fidelity = np.zeros(shape=(num_time_steps, num_ctrls),
                                   dtype=float)
    final_state_dag = forward_propagators[-1].dag()
    for ctrl in range(num_ctrls):
        for t in range(num_time_steps):
            # here we need to take the real part.
            derivative_fidelity[t, ctrl] = 2 * np.real((
                target * (
                        reversed_propagators[::-1][t + 1]
                        * propagator_derivatives[ctrl][t]
                        * forward_propagators[t]
                        * final_state_dag
                ).ptrace(dims=dims, remove=remove) * target.dag())[0, 0]
            )

    return derivative_fidelity


def entanglement_fidelity(
        target: Union[np.ndarray, matrix.OperatorMatrix],
        propagator: Union[np.ndarray, matrix.OperatorMatrix],
        computational_states: Optional[List[int]] = None,
        map_to_closest_unitary: bool = False
) -> np.float64:
    """
    The entanglement fidelity between a simulated Propagator and target
    propagator.

    Parameters
    ----------
    propagator: Union[np.ndarray, ControlMatrix]
        The simulated propagator.

    target: Union[np.ndarray, ControlMatrix]
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
    if type(propagator) == np.ndarray:
        propagator = matrix.DenseOperator(propagator)
    if type(target) == np.ndarray:
        target = matrix.DenseOperator(target)
    d = target.shape[0]
    if computational_states is None:
        trace = (target.dag() * propagator).tr()
    else:
        trace = (target.dag() * propagator.truncate_to_subspace(
            computational_states,
            map_to_closest_unitary=map_to_closest_unitary)).tr()
    return (np.abs(trace) ** 2) / d / d


def derivative_entanglement_fidelity_with_du(
        target: matrix.OperatorMatrix,
        forward_propagators: List[matrix.OperatorMatrix],
        propagator_derivatives: List[List[matrix.OperatorMatrix]],
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
        forward_propagators[i] is the ordered sum of the propagators i..0 in
        descending order.

    propagator_derivatives: List[List[ControlMatrix]],
                         shape: [[] * num_t] * num_ctrl
        Frechet derivatives of the propagators by the control amplitudes.

    target: ControlMatrix
        The target propagator.

    reversed_propagators: List[ControlMatrix]
        The reversed propagators calculated in the systems simulation.
        reversed_propagators[i] is the ordered sum of the propagators n-i..n in
        ascending order where n is the total number of time steps.

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
    target_unitary_dag = target.dag(do_copy=True)
    if computational_states:
        trace = np.conj(
            ((forward_propagators[-1].truncate_to_subspace(
                computational_states,
                map_to_closest_unitary=map_to_closest_unitary)
              * target_unitary_dag).tr())
        )
    else:
        trace = np.conj(((forward_propagators[-1] * target_unitary_dag).tr()))
    num_ctrls = len(propagator_derivatives)
    num_time_steps = len(propagator_derivatives[0])
    d = target.shape[0]

    derivative_fidelity = np.zeros(shape=(num_time_steps, num_ctrls),
                                   dtype=float)

    for ctrl in range(num_ctrls):
        for t in range(num_time_steps):
            # here we need to take the real part.
            if computational_states:
                derivative_fidelity[t, ctrl] = 2 / d / d * np.real(
                    trace * ((reversed_propagators[::-1][t + 1]
                              * propagator_derivatives[ctrl][t]
                              * forward_propagators[t]).truncate_to_subspace(
                        subspace_indices=computational_states,
                        map_to_closest_unitary=map_to_closest_unitary
                    )
                             * target_unitary_dag).tr())
            else:
                derivative_fidelity[t, ctrl] = 2 / d / d * np.real(
                    trace * (reversed_propagators[::-1][t + 1]
                             * propagator_derivatives[ctrl][t]
                             * forward_propagators[t]
                             * target_unitary_dag).tr())

    return derivative_fidelity


def entanglement_fidelity_super_operator(
        target: Union[np.ndarray, matrix.OperatorMatrix],
        propagator: Union[np.ndarray, matrix.OperatorMatrix],
        computational_states: Optional[List[int]] = None,
) -> np.float64:
    """
    The entanglement fidelity between a simulated Propagator and target
    propagator in the super operator formalism.

    The entanglement fidelity between a propagator in the super operator
    formalism of dimension d^2 x d^2 and a target unitary operator of dimension
    d x d. If the system incorporates leakage states, the propagator is
    projected onto the computational space [1].

    Parameters
    ----------
    propagator: Union[np.ndarray, ControlMatrix]
        The simulated evolution propagator in the super operator formalism.

    target: Union[np.ndarray, ControlMatrix]
        The target unitary evolution. (NOT as super operator.)

    computational_states: Optional[List[int]]
        If set, the entanglement fidelity is only calculated for the specified
        subspace.

    Returns
    -------
    fidelity: float
        The entanglement fidelity of target_unitary.dag * unitary.

    Notes
    -----
    [1] Quantification and characterization of leakage errors, Christopher J.
    Wood and Jay M. Gambetta, Phys. Rev. A 97, 032306 - Published 8 March 2018

    """
    if type(propagator) == np.ndarray:
        propagator = matrix.DenseOperator(propagator)
    if type(target) == np.ndarray:
        target = matrix.DenseOperator(target)
    dim_comp = target.shape[0]

    if computational_states is None:
        target_super_operator_inv = \
            matrix.convert_unitary_to_super_operator(
                target.dag())
        trace = (target_super_operator_inv * propagator).tr().real
    else:
        # Here we assume that the full Hilbertspace is the outer sum of a
        # computational and a leakage space.

        # Thus the dimension of the propagator is (d_comp + d_leak) ** 2
        d_leakage = int(np.sqrt(propagator.shape[0])) - dim_comp

        # We fill zeros to the target on the leakage space. We will project
        # onto the computational space anyway.

        target_inv = target.dag()
        target_inv_full_space = matrix.DenseOperator(
            np.zeros((d_leakage + dim_comp, d_leakage + dim_comp)))

        for i, row in enumerate(computational_states):
            for k, column in enumerate(computational_states):
                target_inv_full_space[row, column] = target_inv[i, k]

        # this seems to be wrong
        # target_inv_full_space = matrix.DenseOperator(np.eye(d_leakage)).kron(
        #    target.dag()
        #    )

        # Then convert the target unitary into Liouville space.
        target_super_operator_inv = \
            matrix.convert_unitary_to_super_operator(
                target_inv_full_space)

        # We start the projector with a zero matrix of dimension
        # (d_comp + d_leak).
        projector_comp_state = 0 * target_inv_full_space.identity_like()
        for state in computational_states:
            projector_comp_state[state, state] = 1

        # Then convert the projector into liouville space.
        projector_comp_state = matrix.convert_unitary_to_super_operator(
            projector_comp_state
        )

        trace = (
                projector_comp_state * target_super_operator_inv * propagator
        ).tr().real
    return trace / dim_comp / dim_comp


def deriv_entanglement_fid_sup_op_with_du(
        target: matrix.OperatorMatrix,
        forward_propagators: List[matrix.OperatorMatrix],
        unitary_derivatives: List[List[matrix.OperatorMatrix]],
        reversed_propagators: List[matrix.OperatorMatrix],
        computational_states: Optional[List[int]] = None
):
    """
    Derivative of the entanglement fidelity of a super operator.

    Calculates the derivatives of the entanglement fidelity between a target
    unitary of dimension d x d and a propagator of dimension d^2 x d^2 with
    respect to the control amplitudes.

    Parameters
    ----------
    forward_propagators: List[ControlMatrix], len: num_t +1
        The forward propagators calculated in the systems simulation.
        forward_propagators[i] is the ordered sum of the propagators i..0 in
        descending order.

    unitary_derivatives: List[List[ControlMatrix]],
                         shape: [[] * num_t] * num_ctrl
        Frechet derivatives of the propagators by the control amplitudes.

    target: ControlMatrix
        The target unitary evolution.

    reversed_propagators: List[ControlMatrix]
        The reversed propagators calculated in the systems simulation.
        reversed_propagators[i] is the ordered sum of the propagators n-i..n in
        ascending order where n is the total number of time steps.

    computational_states: Optional[List[int]]
        If set, the entanglement fidelity is only calculated for the specified
        subspace.s only calculated for the specified
        subspace.

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
                    target=target,
                    propagator=reversed_propagators[::-1][t + 1] *
                    unitary_derivatives[ctrl][t] *
                    forward_propagators[t],
                    computational_states=computational_states)
    return derivative_fidelity


class StateInfidelity(CostFunction):
    """Quantum state infidelity.

    TODO:
        * support super operator formalism
        * handle leakage states?
    """

    def __init__(self,
                 solver: solver_algorithms.Solver,
                 target: matrix.OperatorMatrix,
                 label: Optional[List[str]] = None,
                 computational_states: Optional[List[int]] = None,
                 rescale_propagated_state: bool = False
                 ):
        if label is None:
            label = ['State Infidelity', ]
        super().__init__(solver=solver, label=label)
        # assure target is a bra vector

        if target.shape[0] > target.shape[1]:
            self.target = target.dag()
        else:
            self.target = target

        self.computational_states = computational_states
        self.rescale_propagated_state = rescale_propagated_state

    def costs(self) -> np.float64:
        """See base class. """
        final = self.solver.forward_propagators[-1]
        infid = 1. - state_fidelity(
            target=self.target,
            propagated_state=final,
            computational_states=self.computational_states,
            rescale_propagated_state=self.rescale_propagated_state
        )
        return infid

    def grad(self) -> np.ndarray:
        """See base class. """
        derivative_fid = derivative_state_fidelity(
            forward_propagators=self.solver.forward_propagators,
            target=self.target,
            reversed_propagators=self.solver.reversed_propagators,
            propagator_derivatives=self.solver.frechet_deriv_propagators,
            computational_states=self.computational_states,
            rescale_propagated_state=self.rescale_propagated_state
        )
        return -1. * np.real(derivative_fid)


class StateInfidelitySubspace(CostFunction):
    """ Quantum state infidelity on a subspace.

    Assume that the simulated system operates on a product space and the
    target states is described only on a subspace. This class then calculates
    the partial derivative over the neglected subspace.

    Parameters
    ----------
    dims: list of int,
        The dimensions of the subspaces. (Compare to the ptrace function of
        the MatrixOperator class.)

    remove: list of int,
        The indices of the dims list corresponding to the subspaces that are to
        be eliminated. (Compare to the ptrace function of the MatrixOperator
        class.)

    TODO:
        * support super operator formalism
        * handle leakage states?
        * Docstring
    """

    def __init__(self,
                 solver: solver_algorithms.Solver,
                 target: matrix.OperatorMatrix,
                 dims: List[int],
                 remove: List[int],
                 label: Optional[List[str]] = None
                 ):
        if label is None:
            label = ['State Infidelity', ]
        super().__init__(solver=solver, label=label)
        # assure target is a bra vector

        if target.shape[0] > target.shape[1]:
            self.target = target.dag()
        else:
            self.target = target

        self.dims = dims
        self.remove = remove

    def costs(self) -> np.float64:
        """See base class. """
        final = self.solver.forward_propagators[-1]
        infid = 1. - state_fidelity_subspace(
            target=self.target,
            propagated_state=final,
            dims=self.dims,
            remove=self.remove
        )
        return infid

    def grad(self) -> np.ndarray:
        """See base class. """
        derivative_fid = derivative_state_fidelity_subspace(
            forward_propagators=self.solver.forward_propagators,
            target=self.target,
            reversed_propagators=self.solver.reversed_propagators,
            propagator_derivatives=self.solver.frechet_deriv_propagators,
            dims=self.dims,
            remove=self.remove
        )
        return -1. * derivative_fid


class StateNoiseInfidelity(CostFunction):
    """ Averages the state infidelity over noise traces.

    TODO:
        * support super operator formalism
        * implement gradient
        * docstring
    """

    def __init__(self,
                 solver: solver_algorithms.SchroedingerSMonteCarlo,
                 target: matrix.OperatorMatrix,
                 label: Optional[List[str]] = None,
                 computational_states: Optional[List[int]] = None,
                 rescale_propagated_state: bool = False,
                 neglect_systematic_errors: bool = True
                 ):
        if label is None:
            label = ['State Infidelity', ]
        super().__init__(solver=solver, label=label)
        self.solver = solver

        # assure target is a bra vector
        if target.shape[0] > target.shape[1]:
            self.target = target.dag()
        else:
            self.target = target

        self.computational_states = computational_states
        self.rescale_propagated_state = rescale_propagated_state

        self.neglect_systematic_errors = neglect_systematic_errors
        if target is None and not neglect_systematic_errors:
            print('The systematic errors must be neglected if no target is '
                  'set!')
            self.neglect_systematic_errors = True

    def costs(self) -> np.float64:
        """See base class. """
        n_traces = self.solver.noise_trace_generator.n_traces
        infidelities = np.zeros((n_traces,))

        if self.neglect_systematic_errors:
            if self.computational_states is None:
                target = self.solver.forward_propagators[-1]
            else:
                target = self.solver.forward_propagators[
                    -1].truncate_to_subspace(
                    self.computational_states,
                    map_to_closest_unitary=self.rescale_propagated_state
                )
            target = target.dag()
        else:
            target = self.target

        for i in range(n_traces):
            final = self.solver.forward_propagators_noise[i][-1]
            infidelities[i] = 1. - state_fidelity(
                target=target,
                propagated_state=final,
                computational_states=self.computational_states,
                rescale_propagated_state=self.rescale_propagated_state
            )
        return np.mean(infidelities)

    def grad(self) -> np.ndarray:
        """See base class. """
        raise NotImplementedError


class OperationInfidelity(CostFunction):
    """Calculates the infidelity of a quantum channel.

    The infidelity of a quantum channel described by a unitary evolution or
    propagator in the master equation formalism.

    Parameters
    ----------
    solver: `Solver`
        The time slot computer simulating the systems dynamics.

    target: `ControlMatrix`
        Unitary target evolution.

    label: list of str
        Indices of the returned infidelities for distinction in the analysis.

    fidelity_measure: string, optional
        If 'entanglement': the entanglement fidelity is calculated.
        Otherwise an error is raised.

    super_operator_formalism: bool, optional
        If true, the time slot computer is expected to return a propagator in
        the super operator formalism, while the target unitary is not given as
        super operator.
        If false, no super operators are assumed.

    computational_states: list of int, optional
        If set, the chosen fidelity is only calculated for the specified
        subspace.

    map_to_closest_unitary: bool, optional
        If True, then the final propagator is mapped to the closest unitary
        before the infidelity is evaluated.

    Attributes
    ----------
    solver: TimeSlotComputer
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
        * gradient does not truncate to the subspace.

    """

    def __init__(self,
                 solver: solver_algorithms.Solver,
                 target: matrix.OperatorMatrix,
                 fidelity_measure: str = 'entanglement',
                 super_operator_formalism: bool = False,
                 label: Optional[List[str]] = None,
                 computational_states: Optional[List[int]] = None,
                 map_to_closest_unitary: bool = False
                 ):
        if label is None:
            if fidelity_measure == 'entanglement':
                label = ['Entanglement Infidelity', ]
            else:
                label = ['Operator Infidelity', ]

        super().__init__(solver=solver, label=label)
        self.target = target
        self.computational_states = computational_states
        self.map_to_closest_unitary = map_to_closest_unitary

        if fidelity_measure == 'entanglement':
            self.fidelity_measure = fidelity_measure
        else:
            raise NotImplementedError('Only the entanglement fidelity is '
                                      'currently supported.')

        self.super_operator = super_operator_formalism

    def costs(self) -> float:
        """Calculates the costs by the selected fidelity measure. """
        final = self.solver.forward_propagators[-1]

        if self.fidelity_measure == 'entanglement' and self.super_operator:
            infid = 1 - entanglement_fidelity_super_operator(
                propagator=final,
                target=self.target,
                computational_states=self.computational_states
            )
        elif self.fidelity_measure == 'entanglement':
            infid = 1 - entanglement_fidelity(
                propagator=final,
                target=self.target,
                computational_states=self.computational_states,
                map_to_closest_unitary=self.map_to_closest_unitary
            )
        else:
            raise NotImplementedError('Only the entanglement fidelity is '
                                      'implemented in this version.')
        return np.real(infid)

    def grad(self) -> np.ndarray:
        """Calculates the derivatives of the selected fidelity measure with
        respect to the control amplitudes. """
        if self.fidelity_measure == 'entanglement' and self.super_operator:
            derivative_fid = deriv_entanglement_fid_sup_op_with_du(
                forward_propagators=self.solver.forward_propagators,
                target=self.target,
                reversed_propagators=self.solver.reversed_propagators,
                unitary_derivatives=self.solver.frechet_deriv_propagators,
                computational_states=self.computational_states,
            )
        elif self.fidelity_measure == 'entanglement':
            derivative_fid = derivative_entanglement_fidelity_with_du(
                forward_propagators=self.solver.forward_propagators,
                target=self.target,
                reversed_propagators=self.solver.reversed_propagators,
                propagator_derivatives=self.solver.frechet_deriv_propagators,
                computational_states=self.computational_states,
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
    solver: `Solver`
        The time slot computer simulating the systems dynamics.

    target: `ControlMatrix`
        Unitary target evolution.

    label: list of str
        Indices of the returned infidelities for distinction in the analysis.

    fidelity_measure: string, optional
        If 'entanglement': the entanglement fidelity is calculated.
        Otherwise an error is raised.

    computational_states: list of int, optional
        If set, the chosen fidelity is only calculated for the specified
        subspace.

    map_to_closest_unitary: bool, optional
        If True, then the final propagator is mapped to the closest unitary
        before the infidelity is evaluated.

    neglect_systematic_errors: bool
        If true, the mean operator fidelity is calculated with respect to the
        simulated propagator without statistical noise.
        Otherwise the mean operator fidelity is calculated with respect to the
        target propagator.

    Attributes
    ----------
    neglect_systematic_errors: bool
        If true, the standard deviation of the operator fidelity is measured.
        Otherwise the mean operator fidelity is calculated with respect to the
        target propagator.

    """

    def __init__(self,
                 solver: solver_algorithms.SchroedingerSMonteCarlo,
                 target: Optional[matrix.OperatorMatrix],
                 label: Optional[List[str]] = None,
                 fidelity_measure: str = 'entanglement',
                 computational_states: Optional[List[int]] = None,
                 map_to_closest_unitary: bool = False,
                 neglect_systematic_errors: bool = True):
        if label is None:
            label = ['Operator Noise Infidelity']
        super().__init__(solver=solver, label=label)
        self.solver = solver
        self.target = target

        self.computational_states = computational_states
        self.map_to_closest_unitary = map_to_closest_unitary
        self.fidelity_measure = fidelity_measure

        self.neglect_systematic_errors = neglect_systematic_errors
        if target is None and not neglect_systematic_errors:
            print('The systematic errors must be neglected if no target is '
                  'set!')
            self.neglect_systematic_errors = True

    def _to_comp_space(self,
                       dynamic_target: matrix.OperatorMatrix) -> matrix.OperatorMatrix:
        """Map an operator to the computational space"""
        if self.computational_states is not None:
            return dynamic_target.truncate_to_subspace(
                subspace_indices=self.computational_states,
                map_to_closest_unitary=self.map_to_closest_unitary,
            )
        else:
            return dynamic_target

    def _effective_target(self) -> matrix.OperatorMatrix:
        if self.neglect_systematic_errors:
            return self._to_comp_space(self.solver.forward_propagators[-1])
        else:
            return self.target

    def costs(self):
        """See base class. """
        n_traces = self.solver.noise_trace_generator.n_traces
        infidelities = np.zeros((n_traces,))

        target = self._effective_target()

        if self.fidelity_measure == 'entanglement':
            for i in range(n_traces):
                final = self.solver.forward_propagators_noise[i][-1]

                infidelities[i] = 1 - entanglement_fidelity(
                    propagator=final, target=target,
                    computational_states=self.computational_states,
                    map_to_closest_unitary=self.map_to_closest_unitary
                )
        else:
            raise NotImplementedError('Only the entanglement fidelity is '
                                      'currently implemented in this class.')

        return np.mean(np.real(infidelities))

    def grad(self):
        """See base class. """
        target = self._effective_target()

        n_traces = self.solver.noise_trace_generator.n_traces
        num_t = len(self.solver.transferred_time)
        num_ctrl = len(self.solver.h_ctrl)
        derivative = np.zeros((num_t, num_ctrl, n_traces,))
        for i in range(n_traces):
            temp = derivative_entanglement_fidelity_with_du(
                target=target,
                forward_propagators=self.solver.forward_propagators_noise[i],
                propagator_derivatives=
                self.solver.frechet_deriv_propagators_noise[i],
                reversed_propagators=self.solver.reversed_propagators_noise[i],
                computational_states=self.computational_states
            )
            if self.neglect_systematic_errors:
                temp_target = self._to_comp_space(
                    self.solver.forward_propagators_noise[i][-1])

                temp += derivative_entanglement_fidelity_with_du(
                    target=temp_target,
                    forward_propagators=self.solver.forward_propagators,
                    propagator_derivatives=
                    self.solver.frechet_deriv_propagators,
                    reversed_propagators=self.solver.reversed_propagators,
                    computational_states=self.computational_states
                )
            derivative[:, :, i] = np.real(temp)
        return np.mean(-derivative, axis=2)


class LiouvilleMonteCarloEntanglementInfidelity(CostFunction):
    """
    Entanglement infidelity for a combination of Monte Carlo and Liouville
    description.

    The propagators are first mapped to the super operator formalism in
    Liouville space. Next, they are averaged and finally we calculate the
    entanglement infidelity.

    Systematic errors cannot be neglected in the current formulation.

    Parameters
    ----------
    solver: `Solver`
        The time slot computer simulating the systems dynamics.

    target: `ControlMatrix`
        Unitary target evolution.

    label: list of str
        Indices of the returned infidelities for distinction in the analysis.

    computational_states: list of int, optional
        If set, the chosen fidelity is only calculated for the specified
        subspace.

    """

    def __init__(self,
                 solver: solver_algorithms.SchroedingerSMonteCarlo,
                 target: Optional[matrix.OperatorMatrix],
                 label: Optional[List[str]] = None,
                 computational_states: Optional[List[int]] = None):
        if label is None:
            label = ['Super Operator M.C. Ent. Infidelity']
        super().__init__(solver=solver, label=label)
        self.solver = solver
        self.target = target

        self.computational_states = computational_states

    def costs(self):
        """See base class. """
        n_traces = len(self.solver.forward_propagators_noise)
        dim = self.solver.forward_propagators_noise[0][0].shape[0]
        propagator = type(
            self.solver.forward_propagators_noise[0][0])(
            np.zeros([dim ** 2, dim ** 2]))
        for propagators_by_trace in \
                self.solver.forward_propagators_noise:
            propagator += convert_unitary_to_super_operator(
                propagators_by_trace[-1])
        propagator *= (1 / n_traces)

        infid = 1 - entanglement_fidelity_super_operator(
            propagator=propagator,
            target=self.target,
            computational_states=self.computational_states
        )
        return infid

    def grad(self):
        """See base class. """
        raise NotImplementedError('The derivative of the cost function '
                                  'LiouvilleMonteCarloEntanglementInfidelity'
                                  ' has not been implemented'
                                  'yet.')


class OperatorFilterFunctionInfidelity(CostFunction):
    """
    Calculates the infidelity with the filter function formalism.

    Parameters
    ----------
    solver: `Solver`
        The time slot computer simulating the systems dynamics.

    label: list of str
        Indices of the returned infidelities for distinction in the analysis.

    noise_power_spec_density: Callable
        The noise power spectral density in units of inverse frequencies that
        returns an array of shape (n_omega,) or (n_nops, n_omega). In the first
        case, the same spectrum is taken for all noise operators, in the
        second, it is assumed that there are no correlations between different
        noise sources and thus there is one spectrum for each noise operator.
        The order of the noise terms must correspond to the order defined in
        the solver by filter_function_h_n.

    omega: Sequence[float]
        The frequencies at which the integration is to be carried out.

    """

    def __init__(self,
                 solver: solver_algorithms.Solver,
                 noise_power_spec_density: Callable,
                 omega: Sequence[float],
                 label: Optional[List[str]] = None):
        if label is None:
            label = ['Infidelity Filter Function', ]
        super().__init__(solver=solver, label=label)
        self.noise_power_spec_density = noise_power_spec_density
        self._omega = omega

    @property
    def omega(self):
        if self._omega is None:
            if self.solver.pulse_sequence is None:
                self.solver.create_pulse_sequence()

            self._omega = filter_functions.util.get_sample_frequencies(
                pulse=self.solver.pulse_sequence,
                n_samples=200,
                spacing='log',
            )
        return self._omega

    def costs(self) -> Union[float, np.ndarray]:
        """
        The infidelity is calculated with the filter function package. See its
        documentation for more information.

        Returns
        -------
        costs: Union[float, np.ndarray]
            The infidelity.

        """
        if self.solver.pulse_sequence is None:
            self.solver.create_pulse_sequence()
        infidelity = filter_functions.numeric.infidelity(
            pulse=self.solver.pulse_sequence,
            spectrum=self.noise_power_spec_density(self.omega),
            omega=self.omega,
            cache_intermediates=True,
            n_oper_identifiers=self.solver.filter_function_n_oper_identifiers,
        )
        return infidelity

    def grad(self):
        """
        The gradient of the infidelity is calculated with the filter function
        package. See its documentation for more information.

        Raises
        ------
        NotImplementedError
            This method has not been implemented yet.

        """
        if self.solver.pulse_sequence is None:
            self.solver.create_pulse_sequence()

        c_id = ['Control' + str(i) for i in range(len(self.solver.h_ctrl))]

        derivative = filter_functions.gradient.infidelity_derivative(
            pulse=self.solver.pulse_sequence,
            spectrum=self.noise_power_spec_density(self.omega),
            omega=self.omega,
            control_identifiers=c_id,
            n_oper_identifiers=self.solver.filter_function_n_oper_identifiers,
            n_coeffs_deriv=self.solver.filter_function_n_coeffs_deriv_vals
        )
        # what comes from ff:
        # num_noise_contribution, num_t, num_ctrls_direction
        # need to return: (num_t, num_f, num_ctrl)
        derivative = derivative.transpose(1, 0, 2)
        return derivative


class LeakageError(CostFunction):
    r"""This class measures leakage as quantum operation error.

    The resulting infidelity is measured by truncating the leakage states of
    the propagator U yielding the Propagator V on the computational basis. The
    infidelity is then given as the distance from unitarity:
    infid = 1 - trace(V^\dag V) / d

    Parameters
    ----------
    solver : TimeSlotComputer
        The time slot computer computing the propagation of the system.

    computational_states : list of int
        List of indices marking the computational states of the propagator.
        These are all but the leakage states.

    label: list of str
        Indices of the returned infidelities for distinction in the analysis.

    """

    def __init__(self, solver: solver_algorithms.Solver,
                 computational_states: List[int],
                 label: Optional[List[str]] = None):
        if label is None:
            label = ["Leakage Error", ]
        super().__init__(solver=solver, label=label)
        self.computational_states = computational_states

    def costs(self):
        """See base class. """
        final_prop = self.solver.forward_propagators[-1]
        clipped_prop = final_prop.truncate_to_subspace(
            self.computational_states)
        temp = clipped_prop.dag(do_copy=True)
        temp *= clipped_prop

        # the result should always be positive within numerical accuracy
        return max(0, 1 - temp.tr().real / clipped_prop.shape[0])

    def grad(self):
        """See base class. """
        num_ctrls = len(self.solver.frechet_deriv_propagators)
        num_time_steps = len(self.solver.frechet_deriv_propagators[0])

        derivative_fidelity = np.zeros(shape=(num_time_steps, num_ctrls),
                                       dtype=np.float64)

        final = self.solver.forward_propagators[-1]
        final_leak_dag = final.dag(do_copy=True).truncate_to_subspace(
            self.computational_states)
        d = final_leak_dag.shape[0]

        for ctrl in range(num_ctrls):
            for t in range(num_time_steps):
                temp = self.solver.reversed_propagators[::-1][t + 1] \
                       * self.solver.frechet_deriv_propagators[ctrl][t]
                temp *= self.solver.forward_propagators[t]
                temp = temp.truncate_to_subspace(self.computational_states)
                temp *= final_leak_dag
                derivative_fidelity[t, ctrl] = -2. / d * temp.tr().real
        return derivative_fidelity


class IncoherentLeakageError(CostFunction):
    r"""This class measures leakage as quantum operation error.

    The resulting infidelity is measured by truncating the leakage states of
    the propagator U yielding the Propagator V on the computational basis. The
    infidelity is then given as the distance from unitarity:
    infid = 1 - trace(V^\dag V) / d

    Parameters
    ----------
    solver : TimeSlotComputer
        The time slot computer computing the propagation of the system.

    computational_states : list of int
        List of indices marking the computational states of the propagator.
        These are all but the leakage states.

    label: list of str
        Indices of the returned infidelities for distinction in the analysis.

    TODO:
        * adjust docstring

    """

    def __init__(self, solver: solver_algorithms.SchroedingerSMonteCarlo,
                 computational_states: List[int],
                 label: Optional[List[str]] = None):
        if label is None:
            label = ["Incoherent Leakage Error", ]
        super().__init__(solver=solver, label=label)
        self.solver = solver
        self.computational_states = computational_states

    def costs(self):
        """See base class. """
        final_props = [
            props[-1] for props in self.solver.forward_propagators_noise
        ]
        clipped_props = [
            prop.truncate_to_subspace(self.computational_states,
                                      map_to_closest_unitary=False)
            for prop in final_props
        ]
        temp = [
            c_prop.dag(do_copy=True) * c_prop
            for c_prop in clipped_props
        ]
        result = [
            1 - product.tr().real / len(self.computational_states)
            for product in temp
        ]
        result = np.mean(np.asarray(result))
        return result

    def grad(self):
        """See base class. """
        raise NotImplementedError('Derivatives only implemented for the '
                                  'coherent leakage.')


class LeakageLiouville(CostFunction):
    r"""This class measures leakage in Liouville space.

    The leakage is calculated in Liouville space as matrix element. In pseudo
    Code:

        L = < projector leakage space | Propagator | projector comp. space >

    Parameters
    ----------
    solver : TimeSlotComputer
        The time slot computer computing the propagation of the system.

    computational_states : list of int
        List of indices marking the computational states of the propagator.
        These are all but the leakage states.

    label: list of str
        Indices of the returned infidelities for distinction in the analysis.

    verbose: int
        Additional printed output for debugging.

    input_unitary: bool
        If True, then the input is assumed to be formulated in the standard
        Hilbert space and thus expressed as unitary propagator. This propagator
        is then expressed as superoperator.

    monte_carlo: bool
        If True, then we make a monte carlo simulation and average over the
        propagators.

    """

    def __init__(self, solver: solver_algorithms.Solver,
                 computational_states: List[int],
                 label: Optional[List[str]] = None,
                 verbose: int = 0,
                 input_unitary: bool = False,
                 monte_carlo: bool = False):
        if label is None:
            label = ["Leakage Error Lindblad", ]
        super().__init__(solver=solver, label=label)
        self.computational_states = computational_states
        dim = self.solver.h_ctrl[0].shape[0]
        self.dim_comp = len(self.computational_states)
        self.verbose = verbose
        operator_class = type(self.solver.h_ctrl[0])

        # create projectors
        projector_comp = operator_class(
            np.diag(np.ones([dim, ], dtype=complex)))
        projector_leakage = operator_class(
            np.diag(np.ones([dim, ], dtype=complex)))

        for state in computational_states:
            projector_leakage[state, state] = 0
        projector_comp -= projector_leakage

        # vectorize projectors
        self.projector_leakage_bra = ket_vectorize_density_matrix(
            projector_leakage).transpose()

        self.projector_comp_ket = ket_vectorize_density_matrix(projector_comp)

        self.input_unitary = input_unitary
        self.monte_carlo = monte_carlo

    def costs(self):
        """See base class. """
        if self.input_unitary:
            if self.monte_carlo:
                n_traces = len(self.solver.forward_propagators_noise)
                dim = self.solver.forward_propagators_noise[0][0].shape[0]
                propagator = type(
                    self.solver.forward_propagators_noise[0][0])(
                    np.zeros([dim ** 2, dim ** 2]))
                for propagators_by_trace in \
                    self.solver.forward_propagators_noise:
                    propagator += convert_unitary_to_super_operator(
                        propagators_by_trace[-1])
                propagator *= (1 / n_traces)
            else:
                propagator = convert_unitary_to_super_operator(
                    self.solver.forward_propagators[-1])
        else:
            if self.monte_carlo:
                n_traces = len(self.solver.forward_propagators_noise)
                dim = self.solver.forward_propagators_noise[0][0].shape[0]
                propagator = type(
                    self.solver.forward_propagators_noise[0][0])(
                    np.zeros([dim, dim]))
                for propagators_by_trace in \
                    self.solver.forward_propagators_noise:
                    propagator += propagators_by_trace[-1]
                propagator *= (1 / n_traces)
            else:
                propagator = self.solver.forward_propagators[-1]

        leakage = (1 / self.dim_comp) * (
                self.projector_leakage_bra
                * propagator
                * self.projector_comp_ket
        )

        if self.verbose > 0:
            print('leakage:')
            print(leakage[0, 0])

        # the result should always be positive within numerical accuracy
        return leakage.data.real[0]

    def grad(self):
        """See base class. """
        raise NotImplementedError('The derivative of the cost function '
                                  'LeakageLiouville has not been implemented'
                                  'yet.')


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
    target_unitary_dag = target_unitary.dag(do_copy=True)
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
    ----------
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
    temp = unitary.dag(do_copy=True) * target_unitary

    temp = [ort.dag(do_copy=True) * temp.dag(do_copy=True) * ort * temp
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
                                     propagators_past, delta_t,
                                     target_unitary):
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
            bkwd_prop_target = propagators_future[t + 1].dag() * target_unitary
            temp = 0
            for ort in orthogonal_operators:
                lambda_ = bkwd_prop_target * ort.dag(do_copy=True)
                lambda_ *= bkwd_prop_target.dag()
                rho = propagators_past[t + 1] * ort
                rho *= propagators_past[t + 1].dag()
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
