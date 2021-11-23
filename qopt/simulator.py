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
"""The `Simulator` class provides the interface between the optimizer and the
actual simulation.

The clean interface between the simulation and the optimization algorithm
allows qopt to interface with a wide class of optimization algorithms. A
special focus is set on gradient based optimization algorithms and analytic
gradients are also wrapped by the `Simulator` class.

The correct setup of the entire simulation might contain the
implementation of functions and their derivatives. This a common place for
mistakes and therefore the `Simulator` class offers convenience functions to
check the integrated calculation of gradients based on analytic results with
finite differences.

Classes
-------
:class:`Simulator`
    Base class.


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

from typing import Optional, Sequence
import numpy as np
import time

from qopt import cost_functions, performance_statistics, solver_algorithms

from qopt.util import needs_refactoring


class Simulator(object):
    """
    The Dynamics class provides the interface for the Optimizer class.

    It wraps the cost functions and optionally the gradient of the infidelity.

    Parameters
    ----------
    solvers: Solver
        This object calculates the evolution of the system under
        consideration.

    cost_funcs: List[FidelityComputer]
        These are the parameters which are optimized.

    optimization_parameters: numpy array, optional
        The initial pulse of shape (N_t, N_c) where N_t is the
        number of time steps and N_c the number of controlled parameters.

    num_ctrl: int, optional
        The number of controlled parameters N_c.

    times: numpy array or list, optional
        A one dimensional numpy array of the discrete time steps.

    num_times: int, optional
        The number of time steps N_t. Mainly for consistency checks.

    record_performance_statistics: bool
        If True, then the evaluation times of the cost functions and their
        gradients are stored.

    Attributes
    ----------
    solvers: list of `Solver`
        Instances of the time slot computers used by the cost functions.

    cost_funcs: list of `CostFunction`
        Instances of the cost functions which are to be optimized.

    stats: Stats
        Performance statistics.

    TODO:
        * properly implement check method as parser
        * flags controlling how much data is saved
        * is the pulse attribute useful?
        * check attributes for duplication: should times, num_ctrl and
            num_times be saved at this level?

    """

    def __init__(
            self,
            solvers: Optional[Sequence[solver_algorithms.Solver]],
            cost_funcs: Optional[Sequence[cost_functions.CostFunction]],
            optimization_parameters=None,
            num_ctrl=None,
            times=None,
            num_times=None,
            record_performance_statistics: bool = True,
            numeric_jacobian: bool = False
    ):
        self._num_ctrl = num_ctrl
        self._num_times = num_times
        self._optimization_parameteres = optimization_parameters
        self._times = times

        self.solvers = solvers
        self.cost_funcs = cost_funcs

        self.stats = (performance_statistics.PerformanceStatistics()
                      if record_performance_statistics else None)

        self.numeric_jacobian = numeric_jacobian

    @property
    def pulse(self):
        """Optimization parameters. """
        return self._optimization_parameteres

    @pulse.setter
    def pulse(self, new_pulse):
        """Sets the pulse and the corresponding attributes accordingly. """
        if new_pulse is not None:
            self._num_times, self._num_ctrl = self._optimization_parameteres.shape
        self._optimization_parameteres = new_pulse

    @needs_refactoring
    def check(self):
        """ Verifies the shape of the time steps and the pulse. """
        if self._times.size != self._num_times:
            raise ValueError(
                'There must be self.num_times values in self.times!')

        if self._optimization_parameteres.shape != (self._num_times, self._num_ctrl):
            raise ValueError(
                'The shape of self.pulse does not fit to the number of times'
                ' and control amplitudes!')

    @property
    def cost_indices(self):
        """Indices of cost functions. """
        cost_indices = []
        for cost_func in self.cost_funcs:
            cost_indices += cost_func.label
        return cost_indices

    def wrapped_cost_functions(self, pulse=None):
        """
        Wraps the cost functions of the fidelity computer.

        This function coordinates the complete simulation including the
        application of the transfer function, the execution of the time
        slot computer and the evaluation of the actual cost functions.

        Parameters
        ----------
        pulse: numpy array optional
            If no pulse is specified the cost function is evaluated for the
            attribute pulse.

        Returns
        -------
        costs: numpy array, shape (n_fun)
            Array of costs (i.e. infidelities).

        costs_indices: list of str
            Names of the costs.

        """
        if pulse is None:
            pulse = self.pulse

        for solver in self.solvers:
            solver.set_optimization_parameters(pulse)

        costs = []

        if self.stats:
            self.stats.cost_func_eval_times.append([])
            for i, cost_func in enumerate(self.cost_funcs):
                t_start = time.time()
                cost = cost_func.costs()
                t_end = time.time()
                self.stats.cost_func_eval_times[-1].append(t_end - t_start)

                # reimplement the block below
                costs.append(np.asarray(cost).flatten())

                """
                I do not understand this block anymore. The cost can be an 
                array or a scalar, but the scalar can not be reshaped.
                if hasattr(cost, "__len__"):
                    costs.append(cost)
                else:
                    costs.append(cost.reshape(1))
                """
            costs = np.concatenate(costs, axis=0)
        else:
            for i, cost_func in enumerate(self.cost_funcs):
                cost = cost_func.costs()

                costs.append(np.asarray(cost).flatten())
                """
                if hasattr(cost, "__len__"):
                    costs.append(cost)
                else:
                    costs.append(cost.reshape(1))
                """
            costs = np.concatenate(costs, axis=0)

        return np.asarray(costs)

    def wrapped_jac_function(self, pulse=None):
        """
        Wraps the gradient calculation functions of the fidelity computer.

        Parameters
        ----------
        pulse: numpy array, optional
            shape: (num_t, num_ctrl) If no pulse is specified the cost function
            is evaluated for the attribute pulse.

        Returns
        -------
        jac: numpy array
            Array of gradients of shape (num_t, num_func, num_amp).
        """

        if self.numeric_jacobian:
            return self.numeric_gradient(pulse=pulse)

        if pulse is None:
            pulse = self.pulse

        for solver in self.solvers:
            solver.set_optimization_parameters(pulse)

        jacobians = []

        record_evaluation_times = bool(self.stats)

        if record_evaluation_times:
            self.stats.grad_func_eval_times.append([])

        for i, cost_func in enumerate(self.cost_funcs):
            if record_evaluation_times:
                t_start = time.time()
            jac_u = cost_func.grad()

            # if the cost function is scalar, an extra dimension is inserted
            if len(jac_u.shape) == 2:
                jac_u = np.expand_dims(jac_u, axis=1)

            # apply the chain rule to the derivatives
            jac_x = cost_func.solver.amplitude_function.derivative_by_chain_rule(
                jac_u, cost_func.solver.transfer_function(pulse))
            jac_x_transferred = \
                cost_func.solver.transfer_function.gradient_chain_rule(
                    jac_x
                )
            jacobians.append(jac_x_transferred)
            if record_evaluation_times:
                t_end = time.time()
                self.stats.grad_func_eval_times[-1].append(t_end - t_start)

        # two dimensional form as required by scipy solvers
        total_jac = np.concatenate(jacobians, axis=1)

        return total_jac

    def compare_numeric_to_analytic_gradient(
            self, pulse: Optional[np.ndarray] = None,
            delta_eps: float = 1e-8,
            symmetric: bool = False
    ):
        """
        This function compares the numerical to the analytical gradient in order
        to serve as a consistency check.

        Parameters
        ----------
        pulse: array
            The pulse at which the gradient is evaluated.

        delta_eps: float
            The finite difference.

        symmetric: bool
            If True, then the finite differences are evaluated symmetrically
            around the pulse. Otherwise by forward finite differences.

        Returns
        -------
        gradient_difference_norm: float
            The matrix norm of the difference between the numeric and analytic
            gradient.

        gradient_difference_relative: float
            The relation of the aforementioned norm of the difference matrix
            and the average norm of the numeric and analytic gradient.

        """
        numeric_gradient = self.numeric_gradient(pulse=pulse,
                                                 delta_eps=delta_eps,
                                                 symmetric=symmetric)
        analytic_gradient = self.wrapped_jac_function(pulse=pulse)

        diff_norm = np.linalg.norm(numeric_gradient - analytic_gradient)
        relative_difference = 2 * diff_norm \
            / (np.linalg.norm(numeric_gradient)
               + np.linalg.norm(analytic_gradient))
        return diff_norm, relative_difference

    def numeric_gradient(
            self, pulse: Optional[np.ndarray] = None,
            delta_eps: float = 1e-8,
            symmetric: bool = False
    ) -> np.ndarray:
        """
        This function calculates the gradient numerically and analytically
        in order to serve as a consistency check.

        Parameters
        ----------
        pulse: array
            The pulse at which the gradient is evaluated.

        delta_eps: float
            The finite difference.

        symmetric: bool
            If True, then the finite differences are evaluated symmetrically
            around the pulse. Otherwise by forward finite differences.

        Returns
        -------
        gradients: array
            The gradients as numpy array of shape (n_time, n_func, n_opers).

        """
        if pulse is None:
            test_pulse = self.pulse
        else:
            test_pulse = pulse

        central_costs = self.wrapped_cost_functions(pulse=test_pulse)

        n_times, n_operators = test_pulse.shape
        n_cost_funcs = len(central_costs)

        gradients = np.zeros((n_times, n_cost_funcs, n_operators))

        for n_time in range(n_times):
            for n_operator in range(n_operators):
                delta = np.zeros_like(test_pulse, dtype=float)
                delta[n_time, n_operator] = delta_eps
                fwd_val = self.wrapped_cost_functions(test_pulse + delta)
                if symmetric:
                    bck_val = self.wrapped_cost_functions(test_pulse - delta)
                    gradients[n_time, :, n_operator] = (fwd_val - bck_val) / (
                            2 * delta_eps)
                else:
                    gradients[n_time, :, n_operator] = \
                        (fwd_val - central_costs) / delta_eps

        return gradients
