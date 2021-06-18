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
This module contains functions for the support of multiprocessing.

The function `run_optimization_parallel` can be used to perform the
optimization for multiple initial conditions in parallel.

Caution! The solver class `SchroedingerSMonteCarlo` offers a functionality for
the parallel execution of the simulation vor various noise samples. These
features are not compatible. The program can only be parallelized once.

Functions
---------
:func:`run_optimization`
    Executes the run_optimization method of an optimizer.

:func:`run_optimization_parallel`
    Parallel execution of the run_optimization Method of the
    Optimizer.

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

from multiprocessing import Pool
import copy

from qopt.data_container import DataContainer


def run_optimization(optimizer, initial_pulse):
    """ Executes the run_optimization method of an optimizer.

    Parameters
    ----------
    optimizer: Optimizer
        The Optimizer.

    initial_pulse: numpy array, shape (num_t, num_ctrl)
        The initial pulse.

    Returns
    -------
    result: OptimizationResult
        The result of the optimization.

    """
    return optimizer.run_optimization(initial_pulse)


def run_optimization_parallel(optimizer, initial_pulses, processes=None):
    """ Parallel execution of the run_optimization Method of the
    Optimizer.

    Parameters
    ----------
    optimizer: Optimizer
        The Optimizer.

    initial_pulses: numpy array, shape (num_init, num_t, num_ctrl)
        The initial pulse. Where num_init is the number of initial pulses.

    processes: int, optional
        If an integer is given, then the propagation is calculated in
        this number of parallel processes. If 1 then no parallel
        computing is applied. If None then cpu_count() is called to use
        all cores available. Defaults to None.

    Returns
    -------
    data: DataContainer
        A DataContainer in which the OptimizationResults are saved.

    """
    optimizers = [copy.deepcopy(optimizer) for _ in initial_pulses]
    with Pool(processes=processes) as pool:
        results = pool.starmap(
            run_optimization, zip(optimizers, initial_pulses))
    data = DataContainer()
    for result in results:
        data.append_optim_result(result)
    return data
