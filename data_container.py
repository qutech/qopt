import logging
import pickle
import os
import copy

from typing import Optional, List

import optimization_data
import stats


# TODO: rewrite as serializable
# TODO: add time tag
# class DataContainer(metaclass=qutil.storage.HDF5Serializable):
class DataContainer:

    """ This class gathers the information stored in multiple objects of the
    class OptimResult.

    Its primary functionality is the storage of data.

    Parameters
    ----------
    storage_path : string
        The path were this instance of DataContainer is to be stored or from
        where is shall be loaded.

    file_name : string
        The file name will be appended to the path for storage and loading. The
        default value is an empty string assuming that the storage path already
        contains the file name.

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

    append_time_to_path : bool
        If True, the current time is appended to the file name.

    """
    def __init__(self, storage_path: str, file_name: str = '',
                 final_costs: Optional[List] = None,
                 indices: Optional[List[str]] = None,
                 init_parameters: Optional[List] = None,
                 final_parameters: Optional[List] = None,
                 costs: Optional[List[List]] = None,
                 parameters: Optional[List[List]] = None,
                 status: Optional[List] = None,
                 optimization_statistics: Optional[List] = None,
                 append_time_to_path=True):
        if final_costs is None:
            self.final_costs = []
        else:
            self.final_costs = final_costs

        if indices is None:
            self.indices = []
        else:
            self.indices = indices

        if init_parameters is None:
            self.init_parameters = []
        else:
            self.init_parameters = init_parameters

        if final_parameters is None:
            self.final_parameters = []
        else:
            self.final_parameters = final_parameters

        if costs is None:
            self.costs = []
        else:
            self.costs = costs

        if parameters is None:
            self.parameters = []
        else:
            self.parameters = parameters

        if status is None:
            self.status = []
        else:
            self.status = status

        if optimization_statistics is None:
            self.optimization_statistics = []
        else:
            self.optimization_statistics = optimization_statistics

        self.check_length()

        self.storage_path = storage_path
        self.append_time_to_path = append_time_to_path

        self._asyncrone_writer = None
        self.file_name = file_name

    def __len__(self):
        if self.costs is None:
            return 0
        else:
            return len(self.costs)

    """
    @property
    def asyncrone_writer(self):
        if self._asyncrone_writer is None:
            self._asyncrone_writer = qutil.storage.AsynchronousHDF5Writer(
                reserved=dict(),
                multiprocess=False)
        return self._asyncrone_writer
    """

    @property
    def logger(self):
        return logging.getLogger('DataContainer')

    @property
    def index(self):
        return self.final_parameters.index

    """
    def write_to_hdf5(self):
        if self.storage_path:
            if not os.path.isdir(self.storage_path):
                os.makedirs(self.storage_path)
            if self.append_time_to_path:
                full_path = os.path.join(self.storage_path,
                                         self.file_name +
                                         qutil.storage.time_string() + ".hdf5")
            else:
                full_path = os.path.join(self.storage_path, self.file_name +
                                         ".hdf5")
            self.asyncrone_writer.write(
                self,
                file_name=full_path,
                name='DataContainer')

        else:
            logging.warning('The Data could not be saved because no storage'
                            'path was determined.')
    """
    def check_length(self):
        pass

    def append_optim_result(
            self,
            optim_result: optimization_data.OptimResult):

        self.append(final_costs=optim_result.final_cost,
                    indices=optim_result.indices,
                    init_parameters=optim_result.init_parameters,
                    final_parameters=optim_result.final_parameteres,
                    costs=optim_result.optim_iter_summary.costs,
                    parameters=optim_result.optim_iter_summary.parameters,
                    status=optim_result.status,
                    optimization_stats=optim_result.optimization_stats
                    )

    def append(self, final_costs: List, indices: List[str],
               init_parameters: List, final_parameters: List,
               costs: List, parameters: List, status: int,
               optimization_stats: Optional[stats.Stats]):
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
            optimization_statistics=copy.deepcopy(self.optimization_statistics),
            append_time_to_path=copy.deepcopy(self.append_time_to_path))
        return cpyobj

    def to_hdf5(self):
        return self.to_dict()

    def to_pickle(self, filename=None):
        if filename is None:
            if self.file_name is not None:
                filename = os.path.join(self.storage_path, self.file_name)
            else:
                filename = self.storage_path
        infile = open(filename, 'wb')
        pickle.dump(self.to_dict(), infile)
        infile.close()

    @classmethod
    def from_pickle(cls, filename):
        outfile = open(filename, 'rb')
        data_dict = pickle.load(outfile)
        outfile.close()
        return cls.from_dict(data_dict=data_dict)

    def to_dict(self):
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
                    optimization_statistics=self.optimization_statistics)

    @classmethod
    def from_dict(cls, data_dict):
        return cls(**data_dict)
