from multiprocessing import Pool
import copy

from qopt.data_container import DataContainer


def run_optimization(optimizer, initial_pulse):
    return optimizer.run_optimization(initial_pulse)


def run_optimization_parallel(optimizer, initial_pulses, processes=None):
    optimizers = [copy.deepcopy(optimizer) for _ in initial_pulses]
    with Pool(processes=processes) as pool:
        results = pool.starmap(
            run_optimization, zip(optimizers, initial_pulses))
    data = DataContainer()
    for result in results:
        data.append_optim_result(result)
    return data
