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
"""The `Analyser` class operates on an instance of the `DataContainer` class
and offers various convenience functions for the visualization and analysis
of data acquired during the optimizations.

These features include the plotting of cost functions, either for a single
optimization run or for a multitude. Also functions for the calculation of
computational time for the various cost functions and if given their gradients
are included.

Classes
-------
:class:`Analyser`
    Holds convenience functions to visualize the optimization.

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
import matplotlib.pyplot as plt

from qopt import data_container


class Analyser:
    """Holds convenience functions to visualize the optimization.
    
    The Analyser class can be used to make plots of the optimization data
    stored in an instance of the DataContainer class. This can be useful to
    judge the performance of optimization algorithms and investigate how fast
    the convergence is and whether the algorithm has fully converged.

    """
    def __init__(self, data: data_container.DataContainer):
        self.data = data
        self.infidelities = None

    @property
    def n_least_square(self) -> int:
        """Returns the number of the optimization run which yields the smallest
        total costs.

        The total cost is measured as squared sum of the final cost function
        values.

        Returns
        -------
        n_least_square : int
            Number of optimization run with smallest final costs.

        """
        final_costs = np.asarray(self.data.final_costs)
        if len(final_costs.shape) == 1:
            final_costs = np.reshape(
                final_costs, (len(self.data.final_costs), 1))
        squared_sum = np.sum(final_costs**2, axis=1)
        return int(np.argmin(squared_sum, axis=0))

    def plot_costs(self, n=0, log_y=True, ax=None) -> None:
        """Plots the absolute cost values as function of optimization
        iteration.

        Parameters
        ----------
        n: int, optional
            Number of the optimization run. Defaults to 0.

        log_y: bool, optional
            If True then the costs are plotted logarithmically. Defaults to
            True.

        ax: matplotlib.pyplot.axes
            Axes element in which the data is plotted. If not specified, a new
            one will be created.

        Returns
        -------
        ax: matplotlib.pyplot.axes
            Axes with the plot.

        """
        if ax is None:
            _, ax = plt.subplots()

        for cost, index in zip(np.asarray(self.data.costs[n]).T, self.data.indices):
            ax.plot(cost, label=index)
        ax.set_ylabel('Costs')
        ax.set_xlabel('Iteration')
        ax.legend()
        if log_y:
            ax.set_yscale('log')

        return ax

    def absolute_costs(self) -> np.ndarray:
        """
        Calculates for each optimization run and each iteration in the
        optimization algorithm the sum of the costs.

        Returns
        -------
        costs: numpy array, shape (n_runs, n_iter)
            The sum of the costs.

        """
        n_steps = np.max(list(map(len, self.data.costs)))

        # shape: (num_runs, num_step, num_cost_fkt)
        costs = np.empty(
            (len(self.data.costs), n_steps, len(self.data.costs[0][0])))
        costs[:] = np.nan

        for i, run in enumerate(self.data.costs):
            num_steps = len(run)
            costs[i, :num_steps, :] = np.stack(run, axis=0)

        costs = np.sqrt(np.sum(costs ** 2, axis=2))
        return costs

    def plot_absolute_costs(self) -> (plt.Figure, plt.Axes):
        """Plots the absolute costs. """
        costs = self.absolute_costs()
        fig, axes = plt.subplots(1, 1)
        ax = axes
        ax.plot(costs.T)
        ax.set_yscale('log')
        ax.set_title('Sum of Infidelitites')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Infidelity')
        return fig, ax

    def integral_cost_fkt_times(self, n: int = 0) -> np.ndarray:
        """Sum of the time required for the evaluation of the cost
        function.

        Parameters
        ----------
        n: int, optional
            Number of the optimization run. Defaults to 0.

        Returns
        -------
        integral_times: np.array
            Integrated time required for the cost function evaluation.

        """
        times = self.data.optimization_statistics[n].cost_func_eval_times
        integral_times = np.sum(np.asarray(times), axis=0)
        return integral_times

    def integral_grad_fkt_times(self, n: int = 0):
        """Sum of the time required for the evaluation of the cost
        function gradients.

        Parameters
        ----------
        n: int, optional
            Number of the optimization run. Defaults to 0.

        Returns
        -------
        integral_times: np.array
            Integrated time required for the cost function gradient evaluation.

        """
        times = self.data.optimization_statistics[n].grad_func_eval_times
        integral_times = np.sum(np.asarray(times), axis=0)
        return integral_times

    def opt_times(self):
        """Total optimization times.

        Returns
        -------
        total_times: np.array
            Time required per optimization run.

        """
        total_times = np.zeros((len(self.data.optimization_statistics)))
        for i in range(len(self.data.optimization_statistics)):
            t_start = self.data.optimization_statistics[i].start_t_opt
            t_end = self.data.optimization_statistics[i].end_t_opt
            total_times[i] = t_end - t_start
        return total_times

    def total_cost_fkt_time(self):
        """Total time of cost function evaluation.

        Returns
        -------
        total_t: np.array
            Total times for the evaluation of cost functions.

        """
        total_t = 0
        for n in range(len(self.data.optimization_statistics)):
            total_t += np.sum(self.integral_cost_fkt_times(n))
        return total_t

    def total_grad_fkt_time(self):
        """Total time of cost function gradient calculation.

        Returns
        -------
        total_t: np.array
            Total times for the calculation of cost functions gradients.

        """
        total_t = 0
        for n in range(len(self.data.optimization_statistics)):
            total_t += np.sum(self.integral_grad_fkt_times(n))
        return total_t

    def time_share_cost_fkt(self):
        """Time share of the cost function evaluation. """
        return self.total_cost_fkt_time() / np.sum(self.opt_times())

    def time_share_grad_fkt(self):
        """Time share of the cost function gradient calculation. """
        return self.total_grad_fkt_time() / np.sum(self.opt_times())
