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
This module contains interfaces to optimization algorithms or the algorithms
themselves.

Currently it supports interfacing to least squares and minimization algorithms
of the scipy package. Conditionally, a simulated annealing is supported, if the
simanneal package is included in the environment. (See installation
instructions on https://github.com/qutech/qopt) The classes for simulated
annealing are in an experimental state and not well tested!

The `Optimizer` class uses the `Simulator` as interface to the cost functions
evaluated after the simulation.


Classes
-------
:class:`Optimizer`
    Base class optimizer.

:class:`LeastSquaresOptimizer`
    An interface to scipy's least squares optimizer.

:class:`ScalarMinimizingOptimizer`
    An interface to scipy's minimize functions.

:class:`WallTimeExceeded`
    Exception for exceeding the optimization's time limit.

:class:`PulseAnnealer`
    Helper class for `SimulatedAnnealing`.

:class:`SimulatedAnnealing`
    Simulated annealing as optimization. Experimental implementation. Not well
    tested!

:class:`SimulatedAnnealingScipy`
    Simulated annealing based on scipy functions. Experimental implementation.
    Not well tested!

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
import scipy
import scipy.optimize
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable, List, Union, Sequence
from unittest import mock
from warnings import warn

from qopt import optimization_data, simulator, performance_statistics


try:
    import simanneal
except ImportError:
    warn('simanneal not installed. '
         'SimulatedAnnealing is not available')
    simanneal = mock.Mock()


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

    Parameters
    ----------
    system_simulator : Simulator
        The simulator is the interface to the simulation.

    termination_cond: dict, optional
        The termination conditions of the optimization.

    save_intermediary_steps: bool, optional
        If True, then the results from intermediary steps are stored. Defaults
        to True.

    cost_func_weights: list of float, optional
        The cost functions are multiplied with these weights during the
        optimisation.

    use_jacobian_function: bool, optional
        If set to true, then the jacobians are calculated analytically.
        Defaults to True.

    store_optimizer: bool, optional
        If True, then the optimizer stores itself in the result class.
        Defaults to False


    Attributes
    ----------
    system_simulator : Simulator
        The simulator is the interface to the simulation.

    pulse_shape : Tuple of int
        The shape of the control amplitudes is saved and used for the
        cost functions while the optimization function might need them flatted.

    cost_func_weights: list of float, optional
        The cost functions are multiplied with these weights during the
        optimisation.

    use_jacobian_function: bool, optional
        If set to true, then the jacobians are calculated analytically.

    store_optimizer: bool, optional
        If True, then the optimizer stores itself in the result class.
        Defaults to False

    """

    def __init__(
            self,
            system_simulator: Optional[simulator.Simulator] = None,
            termination_cond: Optional[Dict] = None,
            save_intermediary_steps: bool = True,
            cost_func_weights: Optional[Sequence[float]] = None,
            use_jacobian_function=True,
            store_optimizer: bool = False
    ):
        self.system_simulator = system_simulator
        self.use_jacobian_function = use_jacobian_function
        self.termination_conditions = default_termination_conditions
        if termination_cond is not None:
            self.termination_conditions.update(**termination_cond)

        self.optim_iter_summary = None
        self.pulse_shape = ()

        self._opt_start_time = 0
        self._min_costs = np.inf
        self._min_costs_par = None
        self._n_cost_fkt_eval = 0
        self._n_jac_fkt_eval = 0

        # flags:
        self.save_intermediary_steps = save_intermediary_steps
        self.store_optimizer = store_optimizer

        self.cost_func_weights = cost_func_weights

        if self.cost_func_weights is not None:
            self.cost_func_weights = np.asarray(
                self.cost_func_weights).flatten()
            if len(self.cost_func_weights) == 0:
                self.cost_func_weights = None

    def cost_func_wrapper(self, optimization_parameters):
        """Wraps the cost function given by the simulator class.

        The relevant information for the analysis is saved.

        Parameters
        ----------
        optimization_parameters: np.array
            Raw optimization parameters in a linear array.

        Returns
        -------
        costs: np.array, shape (n_fun)
            Cost values.

        """
        if (time.time() - self._opt_start_time) \
                > self.termination_conditions['max_wall_time']:
            raise WallTimeExceeded

        costs = self.system_simulator.wrapped_cost_functions(
            optimization_parameters.reshape(self.pulse_shape[::-1]).T)

        if self.save_intermediary_steps:
            self.optim_iter_summary.iter_num += 1
            self.optim_iter_summary.costs.append(costs)
            self.optim_iter_summary.parameters.append(
                optimization_parameters.reshape(self.pulse_shape[::-1]).T
            )
        if np.linalg.norm(costs) < np.linalg.norm(self._min_costs):
            self._min_costs = costs
            self._min_costs_par = optimization_parameters.reshape(
                self.pulse_shape[::-1]).T

        # apply the cost function weights after saving the values.
        if self.cost_func_weights is not None:
            costs *= self.cost_func_weights

        self._n_cost_fkt_eval += 1
        return costs

    def cost_jacobian_wrapper(self, optimization_parameters):
        """Wraps the cost Jacobian function given by the simulator class.

        The relevant information for the analysis is saved.

        Parameters
        ----------
        optimization_parameters: np.array
            Raw optimization parameters in a linear array.

        Returns
        -------
        jacobian: np.array, shape (num_func, num_t * num_amp)
            Jacobian of the cost functions.

        """
        jacobian = self.system_simulator.wrapped_jac_function(
            optimization_parameters.reshape(self.pulse_shape[::-1]).T)

        if self.save_intermediary_steps:
            self.optim_iter_summary.gradients.append(jacobian)

        # jacobian shape (num_t, num_f, num_ctrl) -> (num_f, num_t * num_ctrl)
        jacobian = jacobian.transpose([1, 2, 0])
        jacobian = jacobian.reshape(
            (jacobian.shape[0], jacobian.shape[1] * jacobian.shape[2]))

        # apply the cost function weights after saving the values.
        if self.cost_func_weights is not None:
            jacobian = np.einsum('ab, a -> ab', jacobian,
                                 self.cost_func_weights)

        self._n_jac_fkt_eval += 1
        return jacobian

    @abstractmethod
    def run_optimization(self, initial_control_amplitudes: np.ndarray,
                         verbose) \
            -> optimization_data.OptimizationResult:
        """Runs the optimization of the control amplitudes.

        Parameters
        ----------
        initial_control_amplitudes : array
            shape (num_t, num_ctrl)
        verbose
            Verbosity of the run. Depends on which optimizer is used.

        Returns
        -------
        optimization_result : `OptimizationResult`
            The resulting data of the simulation.

        """
        pass

    def prepare_optimization(self,
                             initial_optimization_parameters: np.ndarray):
        """Prepare for the next optimization.

        Parameters
        ----------
        initial_optimization_parameters : array
            shape (num_t, num_ctrl)

        Data stored in this class might be overwritten.
        """
        self._min_costs = np.inf
        self._min_costs_par = None
        self._n_cost_fkt_eval = 0
        self._n_jac_fkt_eval = 0
        self.pulse_shape = initial_optimization_parameters.shape
        if self.save_intermediary_steps:
            self.optim_iter_summary = \
                optimization_data.OptimizationSummary(
                    indices=self.system_simulator.cost_indices
                )
        self._opt_start_time = time.time()
        if self.system_simulator.stats is not None:
            # If the system simulator wants to write down statistics, then
            # initialise a fresh instance
            self.system_simulator.stats = \
                performance_statistics.PerformanceStatistics()
            self.system_simulator.stats.start_t_opt = float(
                self._opt_start_time)
            self.system_simulator.stats.indices = \
                self.system_simulator.cost_indices

    def write_state_to_result(self):
        """ Writes the current state into an instance of 'OptimizationResult'.

        Intended for saving progress when terminating the optimization in an
        unexpected way.

        Returns
        -------
        result: optimization_data.OptimizationResult
            The current result of the optimization.

        """
        if self.system_simulator.stats is not None:
            self.system_simulator.stats.end_t_opt = time.time()

        if self.use_jacobian_function:
            jac_norm = np.linalg.norm(
                self.cost_jacobian_wrapper(self._min_costs_par))
        else:
            jac_norm = 0

        if self.store_optimizer:
            storage_opt = self
        else:
            storage_opt = None

        optim_result = optimization_data.OptimizationResult(
            final_cost=self._min_costs,
            indices=self.system_simulator.cost_indices,
            final_parameters=self._min_costs_par,
            final_grad_norm=jac_norm,
            num_iter=self._n_cost_fkt_eval,
            termination_reason='Maximum Wall Time Exceeded',
            status=5,
            optimizer=storage_opt,
            optim_summary=self.optim_iter_summary,
            optimization_stats=self.system_simulator.stats
        )
        return optim_result


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

    bounds: list of array-like, optional
        Attention: The boundary format can vary between optimizers!
        The boundary conditions for the pulse optimizations. If none are given
        then the pulse is assumed to take any real value. The boundaries are
        given as a list of two arrays. The first array specifies the upper
        and the second array specifies the lower bounds. Single parameters
        can be excepted by using np.inf with appropriate sign.

    """

    def __init__(
            self,
            system_simulator: Optional[simulator.Simulator] = None,
            termination_cond: Optional[Dict] = None,
            save_intermediary_steps: bool = True,
            method: str = 'trf',
            bounds: Union[np.ndarray, List, None] = None,
            use_jacobian_function=True,
            cost_func_weights: Optional[Sequence[float]] = None,
            store_optimizer: bool = False):
        super().__init__(system_simulator=system_simulator,
                         termination_cond=termination_cond,
                         save_intermediary_steps=save_intermediary_steps,
                         cost_func_weights=cost_func_weights,
                         use_jacobian_function=use_jacobian_function,
                         store_optimizer=store_optimizer)
        self.method = method
        self.bounds = bounds

    def run_optimization(self, initial_control_amplitudes: np.array,
                         verbose: int = 0) -> optimization_data.OptimizationResult:
        """See base class. """
        super().prepare_optimization(
            initial_optimization_parameters=initial_control_amplitudes)

        if self.use_jacobian_function:
            jac = super().cost_jacobian_wrapper
        else:
            jac = '2-point'

        try:
            result = scipy.optimize.least_squares(
                fun=super().cost_func_wrapper,
                x0=initial_control_amplitudes.T.flatten(),
                jac=jac,
                bounds=self.bounds,
                method=self.method,
                ftol=self.termination_conditions["min_cost_gain"],
                xtol=self.termination_conditions["min_amplitude_change"],
                gtol=self.termination_conditions["min_gradient_norm"],
                max_nfev=self.termination_conditions["max_iterations"],
                verbose=verbose
            )

            if self.system_simulator.stats is not None:
                self.system_simulator.stats.end_t_opt = time.time()

            if self.store_optimizer:
                storage_opt = self
            else:
                storage_opt = None

            optim_result = optimization_data.OptimizationResult(
                final_cost=result.fun,
                indices=self.system_simulator.cost_indices,
                final_parameters=result.x.reshape(
                    self.pulse_shape[::-1]).T,
                final_grad_norm=np.linalg.norm(result.grad),
                num_iter=result.nfev,
                termination_reason=result.message,
                status=result.status,
                optimizer=storage_opt,
                optim_summary=self.optim_iter_summary,
                optimization_stats=self.system_simulator.stats
            )
        except WallTimeExceeded:
            optim_result = self.write_state_to_result()

        return optim_result


class ScalarMinimizingOptimizer(Optimizer):
    """ Interfaces to the minimize functions of the optimization package in
    scipy.

    Parameters
    ----------
    method: string
        Takes methods implemented by scipy.optimize.minimize.

    bounds: sequence, optional
        Attention: The boundary format can vary between optimizers!
        The boundary conditions for the pulse optimizations. If none are given
        then the pulse is assumed to take any real value. The boundaries are
        given as a sequence of (min, max) pairs for each element. Defaults to
        None.

    """
    def __init__(
            self,
            system_simulator: Optional[simulator.Simulator] = None,
            termination_cond: Optional[Dict] = None,
            save_intermediary_steps: bool = True,
            method: str = 'L-BFGS-B',
            bounds: Union[np.ndarray, List, None] = None,
            use_jacobian_function=True,
            cost_func_weights: Optional[Sequence[float]] = None,
            store_optimizer: bool = False
    ):
        super().__init__(system_simulator=system_simulator,
                         termination_cond=termination_cond,
                         save_intermediary_steps=save_intermediary_steps,
                         cost_func_weights=cost_func_weights,
                         use_jacobian_function=use_jacobian_function,
                         store_optimizer=store_optimizer)
        self.method = method
        self.bounds = bounds

    def cost_func_wrapper(self, optimization_parameters):
        """ Evalutes the cost function.

         The total cost function is defined as the sum of cost functions.

         """
        costs = super().cost_func_wrapper(optimization_parameters)
        scalar_costs = np.sum(costs)
        return scalar_costs

    def cost_jacobian_wrapper(self, optimization_parameters):
        """ The Jacobian reduced to the gradient.

        The gradient is calculated by summation over the Jacobian along the
        function axis, because the total cost function is defined as the sum
        of cost functions.

        Returns
        -------
        gradient: numpy array, shape (num_t * num_amp)
            The gradient of the costs in the 2 norm.

        """
        jac = super().cost_jacobian_wrapper(optimization_parameters)
        grad = (np.sum(jac, axis=0))
        return grad

    def run_optimization(self, initial_control_amplitudes: np.array,
                         verbose: bool = False) -> optimization_data.OptimizationResult:
        super().prepare_optimization(
            initial_optimization_parameters=initial_control_amplitudes)

        if self.use_jacobian_function:
            jac = self.cost_jacobian_wrapper
        else:
            jac = None

        if self.method == 'L-BFGS-B':
            try:
                result = scipy.optimize.minimize(
                    fun=self.cost_func_wrapper,
                    x0=initial_control_amplitudes.T.flatten(),
                    jac=jac,
                    bounds=self.bounds,
                    method=self.method,
                    options={
                        'ftol': self.termination_conditions["min_cost_gain"],
                        'gtol': self.termination_conditions["min_gradient_norm"],
                        'maxiter': self.termination_conditions["max_iterations"],
                        'disp': verbose
                    }
                )

                if self.store_optimizer:
                    storage_opt = self
                else:
                    storage_opt = None

                optim_result = optimization_data.OptimizationResult(
                    final_cost=result.fun,
                    indices=self.system_simulator.cost_indices,
                    final_parameters=result.x.reshape(
                        self.pulse_shape[::-1]).T,
                    final_grad_norm=np.linalg.norm(result.jac),
                    num_iter=result.nfev,
                    termination_reason=result.message,
                    status=result.status,
                    optimizer=storage_opt,
                    optim_summary=self.optim_iter_summary,
                    optimization_stats=self.system_simulator.stats
                )
            except WallTimeExceeded:
                optim_result = self.write_state_to_result()

        elif self.method == 'Nelder-Mead':
            try:
                result = scipy.optimize.minimize(
                    fun=self.cost_func_wrapper,
                    x0=initial_control_amplitudes.T.flatten(),
                    bounds=self.bounds,
                    method=self.method,
                    options={
                        'maxiter': self.termination_conditions[
                            "max_iterations"]},
                )

                if self.store_optimizer:
                    storage_opt = self
                else:
                    storage_opt = None

                optim_result = optimization_data.OptimizationResult(
                    final_cost=result.fun,
                    indices=self.system_simulator.cost_indices,
                    final_parameters=result.x.reshape(
                        self.pulse_shape[::-1]).T,
                    num_iter=result.nfev,
                    termination_reason=result.message,
                    status=result.status,
                    optimizer=storage_opt,
                    optim_summary=self.optim_iter_summary,
                    optimization_stats=self.system_simulator.stats
                )
            except WallTimeExceeded:
                optim_result = self.write_state_to_result()

        else:
            try:
                result = scipy.optimize.minimize(
                    fun=self.cost_func_wrapper,
                    x0=initial_control_amplitudes.T.flatten(),
                    bounds=self.bounds,
                    method=self.method
                )

                optim_result = optimization_data.OptimizationResult(
                    final_cost=result.fun,
                    indices=self.system_simulator.cost_indices,
                    final_parameters=result.x.reshape(
                        self.pulse_shape[::-1]).T,
                    num_iter=result.nfev,
                    termination_reason=result.message,
                    status=result.status,
                    optimizer=self,
                    optim_summary=self.optim_iter_summary,
                    optimization_stats=self.system_simulator.stats
                )
            except WallTimeExceeded:
                optim_result = self.write_state_to_result()

        if self.system_simulator.stats is not None:
            self.system_simulator.stats.end_t_opt = time.time()

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
        return np.linalg.norm(self.energy_function(self.state.T.flatten()))


class SimulatedAnnealing(Optimizer):
    """
    This class uses simulated annealing for discrete optimization.

    The use of this class requires the installation of the simanneal package
    from pypi.

    Parameters
    ----------
    initial_temperature: float
        Initial temperature for the annealing algorithm.

    step_size: int
        Initial stepsize.

    interval: int
        Number of optimization iterations before the step size is reduced.

    bounds: array of boundaries, shape: (2, num_t, num_ctrl)
        The boundary conditions for the pulse optimizations. bounds[0] should
        be the lower bounds, and bounds[1] the upper ones.

    """

    def __init__(
            self,
            system_simulator: Optional[simulator.Simulator] = None,
            termination_cond: Optional[Dict] = None,
            save_intermediary_steps: bool = False,
            store_optimizer: bool = False,
            initial_temperature: float = 1.,
            final_temperature: float = 1e-6,
            step_size: int = 1,
            step_ratio: float = 1.,
            bounds: Optional[np.ndarray] = None,
            updates: Optional[int] = None
    ):

        try:
            import simanneal
        except ImportError as err:
            raise RuntimeError(
                'Requirements not fulfilled. Please install simanneal'
            ) from err

        super().__init__(
            system_simulator=system_simulator,
            termination_cond=termination_cond,
            save_intermediary_steps=save_intermediary_steps,
            store_optimizer=store_optimizer
        )

        self.annealer = PulseAnnealer(
            state=0,
            bounds=bounds,
            energy_function=self.cost_func_wrapper,
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

        if self.system_simulator.stats is not None:
            self.system_simulator.stats.end_t_opt = time.time()

        if self.store_optimizer:
            storage_opt = self
        else:
            storage_opt = None

        optim_result = optimization_data.OptimizationResult(
            final_cost=costs,
            indices=self.system_simulator.cost_indices,
            final_parameters=pulse,
            optimizer=storage_opt,
            optim_summary=self.optim_iter_summary,
            optimization_stats=self.system_simulator.stats
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
        The boundary conditions for the pulse optimizations. bounds[0] should
        be the lower bounds, and bounds[1] the upper ones.

    """

    def __init__(
            self,
            system_simulator: Optional[simulator.Simulator] = None,
            termination_cond: Optional[Dict] = None,
            save_intermediary_steps: bool = False,
            store_optimizer: bool = False,
            temperature: float = 1.,
            step_size: int = 1,
            interval: int = 50,
            bounds: Optional[np.ndarray] = None
    ):
        super().__init__(
            system_simulator=system_simulator,
            termination_cond=termination_cond,
            save_intermediary_steps=save_intermediary_steps,
            store_optimizer=store_optimizer
        )
        self.temperature = temperature
        self.step_size = step_size
        self.interval = interval
        self.bounds = bounds

    def run_optimization(self, initial_control_amplitudes: np.ndarray,
                         verbose: bool = False):
        """See base class. """

        super().prepare_optimization(
            initial_optimization_parameters=initial_control_amplitudes)

        if self.store_optimizer:
            storage_opt = self
        else:
            storage_opt = None

        try:
            result = scipy.optimize.basinhopping(
                func=self.cost_func_wrapper,
                x0=initial_control_amplitudes.T.flatten(),
                niter=self.termination_conditions["max_iterations"],
                T=self.temperature,
                stepsize=self.step_size,
                take_step=self._take_step,
                callback=None,
                interval=self.interval,
                disp=verbose
            )

            if self.system_simulator.stats is not None:
                self.system_simulator.stats.end_t_opt = time.time()

            optim_result = optimization_data.OptimizationResult(
                final_cost=result.fun,
                indices=self.system_simulator.cost_indices,
                final_parameters=result.x.reshape(self.pulse_shape[::-1]).T,
                num_iter=result.nfev,
                termination_reason=result.message,
                status=result.status,
                optimizer=storage_opt,
                optim_summary=self.optim_iter_summary,
                optimization_stats=self.system_simulator.stats
            )

        except WallTimeExceeded:
            if self.system_simulator.stats is not None:
                self.system_simulator.stats.end_t_opt = time.time()

            optim_result = optimization_data.OptimizationResult(
                final_cost=self._min_costs,
                indices=self.system_simulator.cost_indices,
                final_parameters=self._min_costs_par,
                num_iter=self._n_cost_fkt_eval,
                termination_reason='Maximum Wall Time Exceeded',
                status=5,
                optimizer=storage_opt,
                optim_summary=self.optim_iter_summary,
                optimization_stats=self.system_simulator.stats
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
        pulse = current_pulse.reshape(self.pulse_shape[::-1]).T

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

        return new_pulse.T.flatten()
