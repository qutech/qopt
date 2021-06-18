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
"""Statistics of the use of computational resources.

The class `PerformanceStatistics` gathers information about the wall time
spend for the calculation of each cost function and its gradient. It can be
used to evaluate the consumption of computational resources.

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


class PerformanceStatistics(object):
    """Stores performance statistics.

    Attributes
    ----------
    start_t_opt: float or None
        Time of the optimizations start. None if it has not been set yet.

    end_t_opt: float or None
        Time of the optimizations end. None if it has not been set yet.

    indices : List[str]
        The indices of the cost functions.

    cost_func_eval_times: list of float
        List of durations of the evaluation of the cost functions.

    grad_func_eval_times: list of float
        List of durations of the evaluation of the gradients.

    """
    def __init__(self):
        self.start_t_opt = None
        self.end_t_opt = None
        self.indices = None
        self.cost_func_eval_times = []
        self.grad_func_eval_times = []
