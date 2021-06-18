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
"""Implements data storage.

The `DataContainer` class stores the contend of multiple `Result` class
instances. Each 'Result' class instance holds the information gathered in
an optimization run.

The `DataContainer` interfaces to the `Analyser` class, which visualizes the
stored data. It has also the functionalities for writing data to and loading
it from the hard drive.

Classes
-------
:class:`DataContainer`
    Data storage class.

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

import pickle
import os
import copy

from typing import Optional, List

from qopt import optimization_data, performance_statistics


class DataContainer:

    """Stores data of the optimization.

    This class gathers the information stored in multiple objects of the
    class `OptimResult`.

    Parameters
    ----------
    storage_path : string
        The path were this instance of DataContainer is to be stored or from
        where is shall be loaded.

    file_name : string
        The file name will be appended to the path for storage and loading. The
        default value is an empty string assuming that the storage path already
        contains the file name.

    indices : list of str
        The indices of the costs.

    final_costs : list
        The final values of the cost function.

    init_parameters : list
        The initial optimization parameters.

    final_parameters : list
        The final optimization parameters.

    costs : list of list
        All values of the cost functions during the optimization.

    parameters : list of list
        All parameters for which the cost functions were evaluated during the
        optimization.

    status : list of None or int
        The termination_reason as integer. Like in scipy.OptimizeResult
        None if the optimization has not started.
        -1: improper input parameters status
        0: the maximum number of function evaluations is exceeded.
        1: gradient norm termination condition is satisfied.
        2: cost function termination condition is satisfied.
        3: minimal step size termination condition is satisfied.
        4: Both 2 and 3 termination conditions are satisfied.
        5: Wall time exceeded.

    optimization_stats : list
        Optimization statistics, which have been appended to the data.

    append_time_to_path : bool
        If True, the current time is appended to the file name.

    """
    def __init__(self,
                 storage_path: Optional[str] = None,
                 file_name: str = 'Temp File',
                 indices: Optional[List[str]] = None,
                 final_costs: Optional[List] = None,
                 init_parameters: Optional[List] = None,
                 final_parameters: Optional[List] = None,
                 costs: Optional[List[List]] = None,
                 parameters: Optional[List[List]] = None,
                 status: Optional[List] = None,
                 optimization_stats: Optional[List] = None,
                 append_time_to_path=True):

        storage_path = os.path.join(
            __file__, r"..\..\temp"
        ) if storage_path is None else storage_path

        self.final_costs = [] if final_costs is None else final_costs
        self.indices = [] if indices is None else indices
        self.init_parameters = (
            [] if init_parameters is None else init_parameters)
        self.final_parameters = (
            [] if final_parameters is None else final_parameters)
        self.costs = [] if costs is None else costs
        self.parameters = [] if parameters is None else parameters
        self.status = [] if status is None else status
        self.optimization_statistics = (
            [] if optimization_stats is None else optimization_stats)

        self.check_length()

        self.storage_path = storage_path
        self.append_time_to_path = append_time_to_path

        self._asyncrone_writer = None
        self.file_name = file_name

    def __len__(self):
        """Number of optimization runs in the data.

        Returns
        -------
        len: int
            Number of optimization runs in the data.

        """
        if self.costs is None:
            return 0
        else:
            return len(self.costs)

    @property
    def index(self):
        """Indices of the cost functions. """
        return self.final_parameters.index

    def check_length(self):
        pass

    def append_optim_result(
            self,
            optim_result: optimization_data.OptimizationResult):
        """Appends an instance of `OptimizationResult` to the stored data.

        The Information gained in an optimization run is extracted and
        appended to the various lists of the `DataContainer`.

        Parameters
        ----------
        optim_result: `OptimizationResult`
            Result of an optimization run.

        """
        if optim_result.optim_summary is None:
            costs = []
            parameters = []
        else:
            costs = optim_result.optim_summary.costs
            parameters = optim_result.optim_summary.parameters

        self._append(final_costs=optim_result.final_cost,
                     indices=optim_result.indices,
                     init_parameters=optim_result.init_parameters,
                     final_parameters=optim_result.final_parameters,
                     costs=costs,
                     parameters=parameters,
                     status=optim_result.status,
                     optimization_stats=optim_result.optimization_stats
                     )

    def _append(self, final_costs: List, indices: List[str],
                init_parameters: List, final_parameters: List,
                costs: List, parameters: List, status: int,
                optimization_stats: Optional[
                   performance_statistics.PerformanceStatistics]):
        if len(self) == 0:
            self.indices = indices
        else:
            assert self.indices == indices

        self.final_costs.append(final_costs)
        self.init_parameters.append(init_parameters)
        self.final_parameters.append(final_parameters)
        self.costs.append(costs)
        self.parameters.append(parameters)
        self.status.append(status)
        self.optimization_statistics.append(optimization_stats)
        self.check_length()

    def __deepcopy__(self):
        cpyobj = type(self)(
            final_costs=copy.deepcopy(self.final_costs),
            indices=copy.deepcopy(self.indices),
            init_parameters=copy.deepcopy(self.init_parameters),
            final_parameters=copy.deepcopy(self.final_parameters),
            costs=copy.deepcopy(self.costs),
            parameters=copy.deepcopy(self.parameters),
            status=copy.deepcopy(self.status),
            storage_path=copy.deepcopy(self.storage_path),
            file_name=copy.deepcopy(self.file_name),
            optimization_stats=copy.deepcopy(
                self.optimization_statistics),
            append_time_to_path=copy.deepcopy(self.append_time_to_path))
        return cpyobj

    def to_pickle(self, filename=None):
        """Dumps the class to pickle.

        Parameters
        ----------
        filename : str
            Name of the file to which the class is pickled.

        """
        if filename is None:
            if self.file_name is not None:
                filename = os.path.join(self.storage_path, self.file_name)
            else:
                filename = self.storage_path
        infile = open(filename, 'wb')
        pickle.dump(self._to_dict(), infile)
        infile.close()

    @classmethod
    def from_pickle(cls, filename):
        """Read class from pickled file.

        Parameters
        ----------
        filename : str
            The name of the file which is loaded.

        """
        outfile = open(filename, 'rb')
        data_dict = pickle.load(outfile)
        outfile.close()
        return cls._from_dict(data_dict=data_dict)

    def _to_dict(self):
        return dict(final_costs=self.final_costs,
                    indices=self.indices,
                    init_parameters=self.init_parameters,
                    final_parameters=self.final_parameters,
                    costs=self.costs,
                    parameters=self.parameters,
                    status=self.status,
                    storage_path=self.storage_path,
                    file_name=self.file_name,
                    append_time_to_path=self.append_time_to_path,
                    optimization_stats=self.optimization_statistics)

    @classmethod
    def _from_dict(cls, data_dict):
        return cls(**data_dict)
