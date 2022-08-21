import numpy as np
import tensorflow as tf
from qopt.tensorflow_util import DEFAULT_COMPLEX_TYPE, DEFAULT_FLOAT_TYPE, \
    convert_to_constant_tensor
from qopt import *


class TensorFlowSolver:
    """

    I need to break the previous interface. For the calculation of gradients
    the values must go through the function returns to avoid tensor flow
    'leakage'.

    Todo:
    - add transfer function
    - add value function
    """
    def __init__(
            self,
            h_ctrl,
            h_drift,
            tau: np.array,
            initial: DenseOperator = None
            # transfer_function: Optional[TransferFunction] = None,
            # amplitude_function: Optional[AmplitudeFunction] = None,
    ):

        # Store dimensions:
        # todo: This can change when we include a transfer function
        self.n_time_steps = len(tau)
        self.n_ctrl_amps = len(h_ctrl)
        self.n_dim = h_ctrl[0].shape[0]

        # Store time and control and drift operators:
        self.tau = tf.constant(
            value=tau, dtype=DEFAULT_FLOAT_TYPE, shape=(self.n_time_steps, ),
            name='Time Steps'
        )
        self.h_ctrl = [
            tf.constant(h.data, dtype=DEFAULT_COMPLEX_TYPE, shape=h.shape)
            for h in h_ctrl]
        self.h_ctrl = tf.stack(self.h_ctrl)
        self.h_drift = [
            tf.constant(h.data, dtype=DEFAULT_COMPLEX_TYPE, shape=h.shape)
            for h in h_drift]
        self.h_drift = tf.stack(self.h_drift)

        if initial is None:
            initial = DenseOperator.identity_like(h_ctrl[0])
        self.initial = convert_to_constant_tensor(initial)
        # tf.constant(initial.data, dtype=DEFAULT_COMPLEX_TYPE)

        # Initialize variables and tensors
        # todo: this changes when we include the amplitude function
        self._opt_pars = tf.Variable(
            np.zeros((self.n_time_steps, self.n_ctrl_amps)),
            dtype=DEFAULT_FLOAT_TYPE
        )
        self._ctrl_amps = tf.Variable(
            np.zeros((self.n_time_steps, self.n_ctrl_amps)),
            dtype=DEFAULT_FLOAT_TYPE
        )

        self._prop = tf.constant(
            np.zeros((self.n_time_steps, self.n_dim, self.n_dim)),
            dtype=DEFAULT_COMPLEX_TYPE
        )
        # todo: it might make sense to make this variable for the forward
        # propagation.
        self._fwd_prop = tf.constant(
            np.zeros((self.n_time_steps, self.n_dim, self.n_dim)),
            dtype=DEFAULT_COMPLEX_TYPE
        )

    def set_optimization_parameters(self, y: tf.Variable) -> tf.Variable:
        # todo: the conversion should be in an outer scope. probably in the
        # solver
        # self._opt_pars = convert_to_constant_tensor(y)
        self._opt_pars = y
        self._ctrl_amps = y
        return self._ctrl_amps

    def _create_dyn_gen(self, ctrl_amps: tf.Variable) -> tf.Tensor:
        control_dynamics = tf.einsum(
            't, tc->tc', self.tau, ctrl_amps)
        control_dynamics = tf.einsum(
            'tc,cij->tij', control_dynamics, self.h_ctrl)
        # t: time
        # c: control operator
        # ij: indices on the control matrix
        hamiltonian = control_dynamics + self.h_drift
        self.dyn_gen = -1j * hamiltonian
        return self.dyn_gen

    def _compute_propagation(self, dyn_gen) -> tf.Tensor:
        self._prop = tensor_matrix_exponentials(dyn_gen)
        return self._prop

    def forward_propagators(self, opt_pars) -> tf.Tensor:
        """
        Let's change the philosophy. This is no langer a property checking if
        the object is already calculated. I assume that there will only be
        a single call with each parameter set and always recalculate
        everything.

        Parameters
        ----------
        opt_pars

        Returns
        -------

        """
        ctrl_amps = self.set_optimization_parameters(y=opt_pars)
        dyn_gen = self._create_dyn_gen(ctrl_amps=ctrl_amps)
        propagators = self._compute_propagation(dyn_gen=dyn_gen)
        self._fwd_prop = tensor_forward_pass(
            initial=self.initial, propagators=propagators,
            num_t=self.n_time_steps)
        return self._fwd_prop

    def _compute_propagation_derivatives(self) -> None:
        pass


@tf.function
def tensor_matrix_exponentials(dyn_gen):
    propagators = tf.linalg.expm(
        input=dyn_gen,
        name='matrix_exponential'
    )
    return propagators


@tf.function
def tensor_forward_pass(initial, propagators, num_t):
    propagator_list = tf.unstack(propagators, num=num_t, axis=0)

    forward_pass = [initial]
    for prop in propagator_list:
        forward_pass.append(prop @ forward_pass[-1])
    return tf.stack(forward_pass)
