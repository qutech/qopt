from multiprocessing import Pool
import copy


class ParallelOptimizer(object):

    def __init__(self, simulator, optimizer, processes = None):
        self.simulator = simulator
        self.optimizer = optimizer
        self.processes = processes

    def parallelize_optimization(self, initial_pulses):
        with Pool(processes=self.processes) as pool:
            results = pool.map(self.optimizer.run_optimization, initial_pulses)
        return results


def run_optimization(optimizer, initial_pulse):
    return optimizer.run_optimization(initial_pulse)


def run_optimization_parallel(optimizer, initial_pulses, processes=None):
    optimizers = [copy.deepcopy(optimizer) for _ in initial_pulses]
    with Pool(processes=processes) as pool:
        results = pool.starmap(
            run_optimization, zip(optimizers, initial_pulses))
    return results
