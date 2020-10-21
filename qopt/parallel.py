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
    """ Parallelizes the execution of the run_optimization Method of the
    Optimizer.

    Parameters
    ----------
    optimizer: Optimizer
        The Optimizer.

    initial_pulses: numpy array, shape (num_t, num_ctrl)
        The initial pulse.

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
