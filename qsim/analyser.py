"""This file contains some functions for the analysis of the optimization.

Classes
-------
:class:`Analyser`
    Holds convenience functions to visualize the optimization.

"""

import pandas as pd
import numpy as np

from qsim import data_container


class Analyser:
    """

    """
    def __init__(self,
                 data: data_container.DataContainer):
        self.data = data
        self.infidelities = None

    @property
    def n_least_square(self):
        final_costs = np.asarray(self.data.final_costs)
        squared_sum = np.sum(final_costs**2, axis=1)
        return np.argmin(squared_sum, axis=0)

    def plot_costs(self, n=0):
        df = pd.DataFrame(data=np.abs(np.asarray(self.data.costs[n]).T),
                          index=self.data.indices)
        df.T.plot(logy=True)

    def absolute_costs(self):
        n_steps = np.max(list(map(len, self.data.costs)))

        # shape: (num_runs, num_step, num_cost_fkt)
        costs = np.empty(
            (len(self.data.costs), n_steps, len(self.data.costs[0][0])))

        for i, run in enumerate(self.data.costs):
            num_steps = len(run)
            costs[i, :num_steps, :] = np.stack(run, axis=0)

        costs = np.sum(costs, axis=2)
        df = pd.DataFrame(costs, index=range(costs.shape[0]),
                          columns=range(costs.shape[1]))
        df.T.plot(logy=True)

    def integral_cost_fkt_times(self, n: int = 0):
        times = self.data.optimization_statistics[n].cost_func_eval_times
        integral_times = np.sum(np.asarray(times), axis=0)
        return integral_times

    def integral_grad_fkt_times(self, n: int = 0):
        times = self.data.optimization_statistics[n].grad_func_eval_times
        integral_times = np.sum(np.asarray(times), axis=0)
        return integral_times

    def opt_times(self):
        total_times = np.zeros((len(self.data.optimization_statistics)))
        for i in range(len(self.data.optimization_statistics)):
            t_start = self.data.optimization_statistics[i].start_t_opt
            t_end = self.data.optimization_statistics[i].end_t_opt
            total_times[i] = t_end - t_start
        return total_times

    def total_cost_fkt_time(self):
        total_t = 0
        for n in range(len(self.data.optimization_statistics)):
            total_t += np.sum(self.integral_cost_fkt_times(n))
        return total_t

    def total_grad_fkt_time(self):
        total_t = 0
        for n in range(len(self.data.optimization_statistics)):
            total_t += np.sum(self.integral_grad_fkt_times(n))
        return total_t

    def time_share_cost_fkt(self):
        return self.total_cost_fkt_time() / np.sum(self.opt_times())

    def time_share_grad_fkt(self):
        return self.total_grad_fkt_time() / np.sum(self.opt_times())
