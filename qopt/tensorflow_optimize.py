import numpy as np
from scipy import optimize
from qopt.tensorflow_simulator import TensorFlowSimulator
from qopt.tensorflow_util import DEFAULT_FLOAT_TYPE
import tensorflow as tf
from qopt.optimize import default_termination_conditions


class TensorFlowScipyOptimizer:
    """
    L-BFGS-B algorithm

    """

    def __init__(
            self,
            simulator: TensorFlowSimulator,
            boundaries: np.array,
            termination_conditions=default_termination_conditions
    ):
        self.simulator = simulator
        self.boundaries = boundaries
        self.termination_conditions = termination_conditions

    def run_optimization(self, initial_opt_pars):

        result = optimize.minimize(
            fun=self.wrapped_value_and_gradient,
            jac=True,
            x0=initial_opt_pars,
            method='L-BFGS-B',
            bounds=self.boundaries,
            options={
                'ftol': self.termination_conditions["min_cost_gain"],
                'gtol': self.termination_conditions["min_gradient_norm"],
                'maxiter': self.termination_conditions["max_iterations"]
            }
        )
        return result

    def wrapped_value_and_gradient(self, opt_pars: np.array):
        """
        This function converts between tensorflow and numpy objects.

        The function builds the interface between a Solver class based on
        tensorflow and the scipy.optimize.minimize function, which is provided
        with numpy arrays as input.

        Parameters
        ----------
        opt_pars: np.array, shape(n_time_steps * n_ctrl, )
            Optimization parameters as required by the scipy optimize
            minimize function.

        Returns
        -------
        value: float
            Function value als float.

        gradient: np.array, shape(n_time_steps * n_ctrl, )
            Gradient as numpy array.

        """
        # bring into the qopt shape
        opt_pars_reshaped = np.reshape(
            opt_pars,
            newshape=(
                self.simulator.solver.n_time_steps,
                self.simulator.solver.n_ctrl_amps
            )
        )
        # Call the cost function
        value, gradient = self.simulator.value_and_gradient(
            tf.Variable(opt_pars_reshaped, dtype=DEFAULT_FLOAT_TYPE)
        )
        gradient = tf.math.real(gradient)
        return np.squeeze(value.numpy()), gradient.numpy().reshape(
            self.simulator.solver.n_time_steps *
            self.simulator.solver.n_ctrl_amps
        )

