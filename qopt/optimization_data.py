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
"""This module stores information about the optimization and its result.

The `OptimizationResult` is generated with the final properties and initial
optimization parameter values of each optimization run. The
`OptimizationSummary` is only created when requested and stores the properties
of each step in the optimization algorithm. This information is valuable for
the choice of the best optimization algorithm.

Classes
-------
:class:`OptimizationResult`
    Describes the information gained by an optimization run.

:class:`OptimizationSummary`
    Describes the whole information gained during an optimization run.

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

from typing import Dict, List


class OptimizationResult(object):
    """
    Resulting data of the optimization.

    An instance of this class is returned by the `Optimizer` after the
    optimization has terminated. It holds the results of the optimization and
    can also contain an instance of `OptimizationSummary` to describe the
    optimization run itself, for example its convergence.
    The parameters of the initialization method are all optional. This class is
    intended to be initialized empty or loaded from a dictionary by the class
    method :meth:`from_dict`.

    Attributes
    ----------
    termination_reason : string
        Reason for the termination as string.

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
        Value of the cost functions after the optimization.

    final_grad_norm : float
        Norm of the gradient after the optimization.

    num_iter : integer
        Number of iterations in the optimization algorithm.

    init_parameters : array, shape: (n_t, n_par)
        The amplitudes at the start of the optimisation, where n_t is
        the number of time steps simulated and n_par the number of
        optimization parameters.

    final_parameters : array, shape: (n_t, n_par)
        The optimization parameters at the end of the optimisation, where n_t
        is the number of time steps simulated and n_par the number of
        optimization parameters.

    optimizer : `Optimizer`
        Instance of the `Optimizer` used to generate the result

    optim_summary : `OptimizationSummary`
        None if no intermediary results are saved. Otherwise the infidelity
        during the optimization.

    """

    def __init__(self, final_cost=None, indices=None, final_parameters=None,
                 final_grad_norm=None, init_parameters=None, num_iter=None,
                 termination_reason="not started yet", status=None,
                 optimization_stats=None,
                 optimizer=None, optim_summary=None):
        self.final_cost = final_cost
        self.indices = indices
        self.final_parameters = final_parameters
        self.final_grad_norm = final_grad_norm

        self.init_parameters = init_parameters

        self.num_iter = num_iter
        self.termination_reason = termination_reason
        self.status = status

        self.optimizer = optimizer
        self.optimization_stats = optimization_stats
        self.optim_summary = optim_summary

    def to_dict(self):
        """Writes the information held by this instance to a dictionary.

        Returns
        -------
        dictionary: dict
            The information stored in a class instance as dictionary.

        """
        return {'final_cost': self.final_cost,
                'indices': self.indices,
                'final_amps': self.final_parameters,
                'final_grad_norm': self.final_grad_norm,
                'init_parameters': self.init_parameters,
                'num_iter': self.num_iter,
                'termination_reason': self.termination_reason,
                'optimizer': self.optimizer,
                'optimization_stats': self.optimization_stats,
                'optim_summary': self.optim_summary
                }

    @classmethod
    def from_dict(cls, data_dict: Dict):
        """Initialize the class with the information held in a dictionary.

        Parameters
        ----------
        data_dict: dict
            Class information.

        Returns
        -------
        optim_result: OptimizationResult
            Class instance.

        """
        return cls(**data_dict)


class OptimizationSummary(object):
    """A summary of an optimization run.

    This class saves the state of the optimization for each iteration. All
    parameters for the initialization are optimal. The class is intended to be
    either initialized empty.

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
