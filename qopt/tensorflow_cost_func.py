import numpy as np

from qopt.tensorflow_solver_algs import TensorFlowSolver
import tensorflow as tf
from typing import Union
from qopt.matrix import DenseOperator
from qopt.tensorflow_util import convert_to_constant_tensor, DEFAULT_FLOAT_TYPE


class TensorFlowProcessInfid:

    def __init__(self,
                 solver: TensorFlowSolver,
                 target: Union[DenseOperator, np.array, tf.Tensor],
                 label: str = ("TensorFlow Process Infid", ),
                 monte_carlo_simulation: bool = False
                 ):
        self.solver = solver
        self.label = list(label)
        self.inverse_dim_squared = 1 / solver.h_ctrl.shape[-1] ** 2
        self.target_unitary_adj = convert_to_constant_tensor(target)
        self.target_unitary_adj = tf.linalg.adjoint(self.target_unitary_adj)

        if monte_carlo_simulation:
            self.costs = self.costs_monte_carlo
        else:
            self.costs = self.costs_schroedinger

    # @tf.function
    def costs_monte_carlo(self):
        # I am not sure if this is missing a mean? maybe the trace does the
        # trick, but that's not obvious.
        x = tf.Variable(1, dtype=DEFAULT_FLOAT_TYPE)
        x = x - process_fid_unitary(
            self.solver.forward_propagators[-1, :, :, :],
            target_unitary_adj=self.target_unitary_adj,
            inverse_dim_squared=self.inverse_dim_squared
        )
        return x

    def costs_schroedinger(self, opt_pars: tf.Variable) -> tf.Variable:
        forward_propagators = self.solver.forward_propagators(
            opt_pars=opt_pars)
        x = 1 - process_fid_unitary(
            forward_propagators[-1, :, :],
            target_unitary_adj=self.target_unitary_adj,
            inverse_dim_squared=self.inverse_dim_squared
        )
        return x


def process_fid_unitary(
        propagator,
        target_unitary_adj,
        inverse_dim_squared
) -> tf.Variable:
    """
    Process fidelity between a unitary quantum channel and a target gate.

    Returns
    -------

    """
    x = tf.linalg.trace(target_unitary_adj @ propagator)
    x = inverse_dim_squared * x * tf.math.conj(x)
    return tf.math.real(x)


class TensorFlowStateInfidelity:

    def __init__(self,
                 solver: TensorFlowSolver,
                 target: Union[DenseOperator, np.array, tf.Tensor],
                 label: str = ("TensorFlow State Infid", ),
                 ):
        self.solver = solver
        self.label = list(label)
        self.target_state_adj = convert_to_constant_tensor(target)
        self.target_state_adj = tf.linalg.adjoint(self.target_state_adj)

    def costs(self, opt_pars: tf.Tensor) -> tf.Tensor:
        forward_propagators = self.solver.forward_propagators(
            opt_pars=opt_pars)
        x = 1 - state_fidelity_on_vectors(
            target_state_adj=self.target_state_adj,
            state=forward_propagators[-1, :, :]
        )
        return x


def state_fidelity_on_vectors(
        target_state_adj: tf.Tensor,
        state: tf.Tensor
) -> tf.Tensor:
    """
    State fidelity between two quantum states given as vectors.

    Parameters
    ----------
    target_state_adj: tf.Tensor, shape=(1, n_dim)
        The target state as bra-vector.

    state: tf.Tensor, shape=(n_dim, 1)
        The state vector as ket-vector.

    Returns
    -------
    fidelity: tf.Tensor, shape=()
        The quantum state fidelity.

    """
    fidelity = target_state_adj @ state
    fidelity = tf.abs(fidelity) ** 2
    return fidelity
