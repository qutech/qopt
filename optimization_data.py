"""
Class containing the results of the pulse optimisation
"""

import numpy as np
from typing import Dict, List


class OptimResult(object):
    """
    Resulting data of the optimization.

    Attributes
    ----------
    termination_reason : string
        Description of the reason for terminating the optimisation

    status : None or int
        The termination_reason as integer. Like in scipy.OptimizeResult
        None if the optimization has not started.
        -1: improper input parameters status
        0: the maximum number of function evaluations is exceeded.
        1: gradient norm termination condition is satisfied.
        2: cost function termination condition is satisfied.
        3: minimal step size termination condition is satisfied.
        4: Both 2 and 3 termination conditions are satisfied.

    final_cost : float
        final (normalised) fidelity that was achieved

    final_grad_norm : float
        Final value of the sum of the squares of the (normalised) fidelity
        error gradients

    num_iter : integer
        Number of iterations of the optimisation algorithm completed

    init_parameters : array[num_tslots, n_ctrls]
        The amplitudes at the start of the optimisation

    final_parameteres : array[num_tslots, n_ctrls]
        The optimization parameters at the end of the optimisation

    optimizer : OptimizerOld
        Instance of the OptimizerOld used to generate the result

    optim_iter_summary : OptimIterSummary, optional
        None if no intermediary results are saved. Otherwise the infidelity
        during the optimization.

    """

    def __init__(self, final_cost=None, indices=None, final_parameters=None,
                 final_grad_norm=None, init_parameteres=None, num_iter=None,
                 termination_reason="not started yet", status=None,
                 optimization_stats=None,
                 optimizer=None, optim_iter_summary=None):
        self.final_cost = final_cost
        self.indices = indices
        self.final_parameteres = final_parameters
        self.final_grad_norm = final_grad_norm

        self.init_parameters = init_parameteres

        self.num_iter = num_iter
        self.termination_reason = termination_reason
        self.status = status

        self.optimizer = optimizer
        self.optimization_stats = optimization_stats
        self.optim_iter_summary = optim_iter_summary

    @classmethod
    def reset(cls):
        return cls()

    def to_dict(self):
        return {'final_cost': self.final_cost,
                'indices': self.indices,
                'final_amps': self.final_parameteres,
                'final_grad_norm': self.final_grad_norm,
                'init_parameters': self.init_parameters,
                'num_iter': self.num_iter,
                'termination_reason': self.termination_reason,
                'optimizer': self.optimizer,
                'optimization_stats': self.optimization_stats,
                'optim_iter_summary': self.optim_iter_summary
                }

    @classmethod
    def from_dict(cls, data_dict: Dict):
        return cls(**data_dict)


class OptimIterSummary(object):
    """A summary of the most recent iteration of the pulse optimisation

    Attributes
    ----------
    iter_num : int
        Number of iterations stored. Serves as checksum to verify that full
        data has been stored.

    costs : List[float]
        Evaluation results of the cost functions. The dictionary is sorted by
        cost function indices. The lists hold one entry for each evaluation.

    indices : List[str]
        The indices of the cost functions.

    gradients : List[array]
        Gradients of the cost functions. The dictionary is again sorted by cost
        function indices and the lists hold one entry per evaluation.

    parameters : List[array]
        Optimization parameters during the optimization.

    """

    def __init__(self, indices=None, iter_num=0, costs=None, gradients=None,
                 parameters=None):
        self.indices = indices
        self.iter_num = iter_num
        if costs is None:
            self.costs = []
        else:
            self.costs = costs
        if gradients is None:
            self.gradients = []
        else:
            self.gradients = gradients
        if parameters is None:
            self.parameters = []
        else:
            self.parameters = parameters
