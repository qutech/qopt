import tensorflow as tf
import numpy as np


class TensorFlowSimulator(object):
    """
    The gradient tape needs to be started in this class.

    Todo: write a simulation with jacobians and multiple solvers and cost
    functions

    The simulator works entirely with tensorflow objects. If required, then
    they will be converted to numpy objects in the optimizer.
    """

    def __init__(
            self,
            solver,
            cost_func
    ):
        self.solver = solver
        self.cost_func = cost_func

    # @tf.function  # todo: could be possible to decorate,
    #  will check that later
    def value_and_gradient(
            self, opt_pars: tf.Variable
    ) -> (tf.Tensor, tf.Tensor):
        """
        Calculates the cost function values and the corresponding jacobian.

        Parameters
        ----------
        opt_pars: tf.Variable, shape=(n_time_steps, n_ctrl)
            The optimization parameters

        Returns
        -------
        values: tf.Tensor, shape=(n_cost_func), dtype=default float
            Values of the cost functions.

        gradient: tf.Tensor, shape=(n_cost_func, n_time_steps, n_ctrl),
            dtype=default float
            Jacobian of the cost functions.

        """
        with tf.GradientTape() as tape:  # Todo: persistent = proTrue
            tape.watch(opt_pars)
            value = self.cost_func.costs(opt_pars)
            # values = tf.stack(
            #     [c_fun.costs(opt_pars) for c_fun in self.cost_funcs])
            gradient = tape.gradient(value, opt_pars)
        return value, tf.math.real(gradient)
