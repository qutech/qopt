"""
Statistics of the use of computational resources.

"""


class OptimizationStatistics(object):
    """Stores and processes performance statistics.

    """
    def __init__(self):
        self.start_t_opt = None
        self.end_t_opt = None
        self.indices = None
        self.cost_func_eval_times = []
        self.grad_func_eval_times = []
