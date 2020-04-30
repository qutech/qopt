"""Statistics of the use of computational resources.

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
