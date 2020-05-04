# -*- coding: utf-8 -*-
# =============================================================================
#     filter_functions
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
Optimization algorithms.

This module implements the optimization algorithms for the optimal control
problem.

Currently supported are:

    LS-TRF - Least squares, Trust Region Reflective

Classes
-------
:class:`Optimizer`
    Base class optimizer.

:class:`LeastSquaresOptimizer`
    An interface to scipy's least squares optimizer.

:class:`WallTimeExceeded`
    Exception for exceeding the optimization's time limit.

Notes
-----
The implementation was inspired by the optimal control package of QuTiP [1]_
(Quantum Toolbox in Python)

References
----------
.. [1] J. R. Johansson, P. D. Nation, and F. Nori: "QuTiP 2: A Python framework
 for the dynamics of open quantum systems.", Comp. Phys. Comm. 184, 1234 (2013)
[DOI: 10.1016/j.cpc.2012.11.019].

"""

import numpy as np
import scipy
import scipy.optimize
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable

from qsim import optimization_data, solver_algorithms, simulator
import simanneal

default_termination_conditions = {
    "min_gradient_norm": 1e-7,
    "min_cost_gain": 1e-7,
    "max_wall_time": 10 * 60.0,
    "max_cost_func_calls": 1e6,
    "max_iterations": 10000,
    "min_amplitude_change": 1e-8
}


class WallTimeExceeded(Exception):
    """Raised when the time limit for the optimization is exceeded. """
    pass


class Optimizer(ABC):
    """ Abstract base class for the optimizer.

    Attributes
    ----------
    dynamics : Dynamics
        The dynamics is the interface to the simulation.

    pulse_shape : Tuple of int
        The shape of the control amplitudes is saved and used for the
        cost functions while the optimization function might need them flatted.

    TODO:
        * implement termination conditions such as wall time!
    """

    def __init__(
            self,
            dynamics: Optional[solver_algorithms.Solver] = None,
            termination_cond: Optional[Dict] = None,
            save_intermediary_steps: bool = False):
        self.dynamics = dynamics
        if termination_cond is None:
            self.termination_conditions = default_termination_conditions
        else:
            self.termination_conditions = termination_cond

        self.optim_iter_summary = None
        self.pulse_shape = ()

        self._opt_start_time = 0
        self._last_costs = None
        self._last_jac = None
        self._last_par = None
        self._n_cost_fkt_eval = 0
        self._n_jac_fkt_eval = 0

        # flags:
        self.save_intermediary_steps = save_intermediary_steps

    def cost_fktn_wrapper(self, optimization_parameters):
        """Wraps the cost function given by the dynamics class.

        The relevant information for the analysis is saved.

        Parameters
        ----------
        optimization_parameters: np.array
            Raw optimization parameters in a linear array.

        Returns
        -------
        costs: np.array
            Cost values.

        """
        if (time.time() - self._opt_start_time) \
                > self.termination_conditions['max_wall_time']:
            raise WallTimeExceeded

        costs = self.dynamics.wrapped_cost_functions(
            optimization_parameters.reshape(self.pulse_shape[::-1]).transfer_matrix)

        if self.save_intermediary_steps:
            self.optim_iter_summary.iter_num += 1
            self.optim_iter_summary.costs._append(costs)
            self.optim_iter_summary.parameters._append(
                optimization_parameters.reshape(self.pulse_shape[::-1]).transfer_matrix
            )

        self._last_costs = costs
        self._last_par = optimization_parameters.reshape(self.pulse_shape[::-1]).transfer_matrix
        self._n_cost_fkt_eval += 1
        return costs

    def cost_jacobian_wrapper(self, optimization_parameters):
        """Wraps the cost Jacobian function given by the dynamics class.

        The relevant information for the analysis is saved.

        Parameters
        ----------
        optimization_parameters: np.array
            Raw optimization parameters in a linear array.

        Returns
        -------
        jacobian: np.array
            Jacobian of the cost functions.

        """
        jacobian = self.dynamics.wrapped_jac_function(
            optimization_parameters.reshape(self.pulse_shape[::-1]).transfer_matrix)

        if self.save_intermediary_steps:
            self.optim_iter_summary.gradients._append(jacobian)

        # jacobian shape (num_t, num_f, num_ctrl) -> (num_f, num_t * num_ctrl)
        jacobian = jacobian.transpose([1, 2, 0])
        jacobian = jacobian.reshape(
            (jacobian.shape[0], jacobian.shape[1] * jacobian.shape[2]))

        self._last_jac = jacobian
        self._n_jac_fkt_eval += 1
        return jacobian

    @abstractmethod
    def run_optimization(self, initial_control_amplitudes: np.ndarray) \
            -> optimization_data.OptimizationResult:
        """Runs the optimization of the control amplitudes.

        Parameters
        ----------
        initial_control_amplitudes : array
            shape (num_t, num_ctrl)

        Returns
        -------
        optimization_result : `OptimizationResult`
            The resulting data of the simulation.

        """
        pass

    def prepare_optimization(self, initial_optimization_parameters: np.ndarray):
        """Prepare for the next optimization.

        Parameters
        ----------
        initial_optimization_parameters : array
            shape (num_t, num_ctrl)

        Data stored in this class might be overwritten.
        """
        self._last_costs = None
        self._last_jac = None
        self._last_par = None
        self._n_cost_fkt_eval = 0
        self._n_jac_fkt_eval = 0
        self.pulse_shape = initial_optimization_parameters.shape
        if self.save_intermediary_steps:
            self.optim_iter_summary = \
                optimization_data.OptimizationSummary(
                    indices=self.dynamics.cost_indices
                )
        self._opt_start_time = time.time()
        if self.dynamics.stats is not None:
            self.dynamics.stats.start_t_opt = float(self._opt_start_time)
            self.dynamics.stats.indices = self.dynamics.cost_indices


class LeastSquaresOptimizer(Optimizer):
    """
    Uses the scipy least squares method for optimization.

    Parameters
    ----------
    system_simulator: `Simulator`
        The systems simulator.

    termination_cond: dict
        Termination conditions.

    save_intermediary_steps: bool, optional
        If False, only the simulation result is stored. Defaults to False.

    method: str, optional
        The optimization method used. Currently implemented are:
        - 'trf': A trust region optimization algorithm. This is the default.

    bounds: array of boundaries, optional
        The boundary conditions for the pulse optimizations. If none are given
        then the pulse is assumed to take any real value.

    use_jacobian_function: bool, optional
        If set to true, then the jacobians are calculated analytically.


    TODO:
        * in the handling of a wall time exceeded message, the last parameters
            * are stored instead of the best ones.

    """

    def __init__(
            self,
            system_simulator: Optional[simulator.Simulator] = None,
            termination_cond: Optional[Dict] = None,
            save_intermediary_steps: bool = False,
            method: str = 'trf',
            bounds: Optional[np.ndarray] = None,
            use_jacobian_function=True):
        super().__init__(dynamics=system_simulator,
                         termination_cond=termination_cond,
                         save_intermediary_steps=save_intermediary_steps)
        self.method = method
        self.bounds = bounds
        self.use_jacobian_function = use_jacobian_function

    def run_optimization(self, initial_control_amplitudes: np.array) \
            -> optimization_data.OptimizationResult:
        """See base class. """
        super().prepare_optimization(
            initial_optimization_parameters=initial_control_amplitudes)

        try:
            if self.use_jacobian_function:
                result = scipy.optimize.least_squares(
                    fun=super().cost_fktn_wrapper,
                    x0=initial_control_amplitudes.T.flatten(),
                    jac=super().cost_jacobian_wrapper,
                    bounds=self.bounds,
                    method=self.method,
                    ftol=self.termination_conditions["min_cost_gain"],
                    xtol=self.termination_conditions["min_amplitude_change"],
                    gtol=self.termination_conditions["min_gradient_norm"],
                    max_nfev=self.termination_conditions["max_iterations"]
                )
            else:
                result = scipy.optimize.least_squares(
                    fun=super().cost_fktn_wrapper,
                    x0=initial_control_amplitudes.T.flatten(),
                    bounds=self.bounds,
                    method=self.method,
                    ftol=self.termination_conditions["min_cost_gain"],
                    xtol=self.termination_conditions["min_amplitude_change"],
                    gtol=self.termination_conditions["min_gradient_norm"],
                    max_nfev=self.termination_conditions["max_iterations"]
                )

            if self.dynamics.stats is not None:
                self.dynamics.stats.end_t_opt = time.time()

            optim_result = optimization_data.OptimizationResult(
                final_cost=result.fun,
                indices=self.dynamics.cost_indices,
                final_parameters=result.x.reshape(
                    self.pulse_shape[::-1]).transfer_matrix,
                final_grad_norm=np.linalg.norm(result.grad),
                num_iter=result.nfev,
                termination_reason=result.message,
                status=result.status,
                optimizer=self,
                optim_summary=self.optim_iter_summary,
                optimization_stats=self.dynamics.stats
            )
        except WallTimeExceeded:
            if self.dynamics.stats is not None:
                self.dynamics.stats.end_t_opt = time.time()

            if self._last_jac is None:
                jac_norm = 0
            else:
                jac_norm = np.linalg.norm(self._last_jac)

            optim_result = optimization_data.OptimizationResult(
                final_cost=self._last_costs,
                indices=self.dynamics.cost_indices,
                final_parameters=self._last_par.reshape(
                    self.pulse_shape[::-1]).transfer_matrix,
                final_grad_norm=jac_norm,
                num_iter=self._n_cost_fkt_eval,
                termination_reason='Maximum Wall Time Exceeded',
                status=5,
                optimizer=self,
                optim_summary=self.optim_iter_summary,
                optimization_stats=self.dynamics.stats
            )

        return optim_result


class PulseAnnealer(simanneal.Annealer):
    """
    Simulated annealer for the discrete optimization of pulses.

    The state is the pulse.

    Parameters
    ----------
    state
    bounds
    energy_function
    step_size
    step_ratio
    Tmax
    Tmin
    steps
    updates

    """
    def __init__(
            self,
            state,
            bounds,
            energy_function: Callable,
            step_size: int = 1,
            step_ratio: float = 1.,
            Tmax = 1.,
            Tmin = 1e-8,
            steps: int = 100,
            updates: Optional[int] = None
    ):
        super().__init__(initial_state=state)
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.steps = steps
        if updates is not None:
            self.updates = updates
        else:
            self.updates = self.steps

        self.bounds = bounds
        self.step_size = step_size
        self.step_ratio = step_ratio
        self.energy_function = energy_function

    def move(self):
        """Moving into a random direction. """
        pulse = self.state

        if type(self.step_size) != int:
            raise ValueError("The step size must be integer! But it is: "
                             + str(self.step_size))

        if self.step_size == 0:
            raise ValueError("The step size has been set to 0.")

        random_step = np.random.randint(
            low=-1 * self.step_size,
            high=self.step_size + 1,
            size=pulse.shape
        )

        # The update mask decides randomly which directions are neglected
        update_mask = np.random.rand(*random_step.shape)
        random_step[update_mask > self.step_ratio] = 0

        new_pulse = pulse + random_step

        # if a limit is exceeded, set the value to the limit
        lower_limit_exceeded = new_pulse < self.bounds[0]
        upper_limit_exceeded = new_pulse > self.bounds[1]

        new_pulse[lower_limit_exceeded] = self.bounds[0][lower_limit_exceeded]
        new_pulse[upper_limit_exceeded] = self.bounds[1][upper_limit_exceeded]

        self.state = new_pulse

    def energy(self):
        """The energy or cost function of the annealer. """
        return np.linalg.norm(self.energy_function(self.state.transfer_matrix.flatten()))


class SimulatedAnnealing(Optimizer):
    """
    This class uses simulated annealing for discrete optimization.

    Parameters
    ----------
    temperature: float
        Initial temperature for the annealing algorithm.

    step_size: int
        Initial stepsize.

    interval: int
        Number of optimization iterations before the step size is reduced.

    bounds: array of boundaries, shape: (2, num_t, num_ctrl)
        The boundary conditions for the pulse optimizations. bounds[0] should be
        the lower bounds, and bounds[1] the upper ones.

    """

    def __init__(
            self,
            dynamics: Optional[dynamics.Dynamics] = None,
            termination_cond: Optional[Dict] = None,
            save_intermediary_steps: bool = False,
            initial_temperature: float = 1.,
            final_temperature: float = 1e-6,
            step_size: int = 1,
            step_ratio: float = 1.,
            bounds: Optional[np.ndarray] = None,
            updates: Optional[int] = None
    ):
        super().__init__(
            dynamics=dynamics,
            termination_cond=termination_cond,
            save_intermediary_steps=save_intermediary_steps
        )

        self.annealer = PulseAnnealer(
            state=0,
            bounds=bounds,
            energy_function=self.cost_fktn_wrapper,
            step_size=step_size,
            step_ratio=step_ratio,
            Tmax=initial_temperature,
            Tmin=final_temperature,
            steps=termination_cond["max_iterations"],
            updates=updates
        )

    def run_optimization(self, initial_control_amplitudes: np.ndarray):
        """See base class. """

        self.prepare_optimization(
            initial_optimization_parameters=initial_control_amplitudes)

        pulse, costs = self.annealer.anneal()

        if self.dynamics.stats is not None:
            self.dynamics.stats.end_t_opt = time.time()

        optim_result = optimization_data.OptimizationResult(
            final_cost=costs,
            indices=self.dynamics.cost_indices,
            final_parameters=pulse,
            optimizer=self,
            optim_summary=self.optim_iter_summary,
            optimization_stats=self.dynamics.stats
        )

        return optim_result

    def prepare_optimization(self, initial_optimization_parameters: np.ndarray):
        super().prepare_optimization(
            initial_optimization_parameters=initial_optimization_parameters)
        self.annealer.state = initial_optimization_parameters


class SimulatedAnnealingScipy(Optimizer):
    """
    This class uses simulated annealing for discrete optimization.

    Parameters
    ----------
    temperature: float
        Initial temperature for the annealing algorithm.

    step_size: int
        Initial stepsize.

    interval: int
        Number of optimization iterations before the step size is reduced.

    bounds: array of boundaries, shape: (2, num_t, num_ctrl)
        The boundary conditions for the pulse optimizations. bounds[0] should be
        the lower bounds, and bounds[1] the upper ones.

    """

    def __init__(
            self,
            dynamics: Optional[dynamics.Dynamics] = None,
            termination_cond: Optional[Dict] = None,
            save_intermediary_steps: bool = False,
            temperature: float = 1.,
            step_size: int = 1,
            interval: int = 50,
            bounds: Optional[np.ndarray] = None
    ):
        super().__init__(
            dynamics=dynamics,
            termination_cond=termination_cond,
            save_intermediary_steps=save_intermediary_steps
        )
        self.temperature = temperature
        self.step_size = step_size
        self.interval = interval
        self.bounds = bounds

    def run_optimization(self, initial_control_amplitudes: np.ndarray):
        """See base class. """

        super().prepare_optimization(
            initial_optimization_parameters=initial_control_amplitudes)

        try:
            result = scipy.optimize.basinhopping(
                func=self.cost_fktn_wrapper,
                x0=initial_control_amplitudes.T.flatten(),
                niter=self.termination_conditions["max_iterations"],
                T=self.temperature,
                stepsize=self.step_size,
                take_step=self._take_step,
                callback=None,
                interval=self.interval,
                disp=True
            )

            if self.dynamics.stats is not None:
                self.dynamics.stats.end_t_opt = time.time()

            optim_result = optimization_data.OptimizationResult(
                final_cost=result.fun,
                indices=self.dynamics.cost_indices,
                final_parameters=result.x.reshape(self.pulse_shape[::-1]).transfer_matrix,
                num_iter=result.nfev,
                termination_reason=result.message,
                status=result.status,
                optimizer=self,
                optim_summary=self.optim_iter_summary,
                optimization_stats=self.dynamics.stats
            )

        except WallTimeExceeded:
            if self.dynamics.stats is not None:
                self.dynamics.stats.end_t_opt = time.time()

            optim_result = optimization_data.OptimizationResult(
                final_cost=self._last_costs,
                indices=self.dynamics.cost_indices,
                final_parameters=self._last_par.reshape(
                    self.pulse_shape[::-1]).transfer_matrix,
                num_iter=self._n_cost_fkt_eval,
                termination_reason='Maximum Wall Time Exceeded',
                status=5,
                optimizer=self,
                optim_summary=self.optim_iter_summary,
                optimization_stats=self.dynamics.stats
            )

        return optim_result

    def _take_step(self, current_pulse: np.ndarray) -> np.ndarray:
        """
        This function applies a random discrete variation to the pulse.

        Parameters
        ----------
        current_pulse: array of int
            The pulse before the application of the take step function.

        Returns
        -------
        new_pulse: array of int
            The pulse initial pulse plus a random variation.

        """
        pulse = current_pulse.reshape(self.pulse_shape[::-1]).transfer_matrix

        if type(self.step_size) != int:
            raise ValueError("The step size must be integer! But it is: "
                             + str(self.step_size))

        if self.step_size == 0:
            raise ValueError("The step size has been set to 0.")

        random_step = np.random.randint(
            low=-1 * self.step_size,
            high=self.step_size + 1,
            size=pulse.shape
        )

        new_pulse = pulse + random_step

        # if a limit is exceeded, set the value to the limit
        lower_limit_exceeded = new_pulse < self.bounds[0]
        upper_limit_exceeded = new_pulse > self.bounds[1]

        new_pulse[lower_limit_exceeded] = self.bounds[0][lower_limit_exceeded]
        new_pulse[upper_limit_exceeded] = self.bounds[1][upper_limit_exceeded]

        return new_pulse.transfer_matrix.flatten()
