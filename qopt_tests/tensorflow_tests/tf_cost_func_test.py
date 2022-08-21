import unittest
import tensorflow as tf
import numpy as np
import qopt.tensorflow_cost_func
import qopt.tensorflow_util
from qopt import *


class TestProcessInfid(unittest.TestCase):
    def test_process_infidelity_unitary(self):
        target_matrix = (.5 * DenseOperator.pauli_x()).exp(-1j * np.pi)
        target_matrix_adj = target_matrix.dag()
        target_matrix_adj_tf = tf.constant(
            target_matrix_adj.data,
            dtype=qopt.tensorflow_util.DEFAULT_COMPLEX_TYPE
        )

        sample_matrices = []
        for phi in np.linspace(0, 2 * np.pi, 15):
            for psi in np.linspace(0, 2 * np.pi, 15):
                sample_matrices.append(
                    (.5 * (
                            np.cos(psi) * DenseOperator.pauli_x() +
                            np.sin(psi) * DenseOperator.pauli_y())
                     ).exp(1j * phi)
                )

        sample_matrices_tf = [
            tf.constant(sm.data, dtype=qopt.tensorflow_util.DEFAULT_FLOAT_TYPE)
            for sm in sample_matrices
        ]

        for i in range(len(sample_matrices)):
            infid_tf = qopt.tensorflow_cost_func.process_fid_unitary(
                    propagator=sample_matrices_tf[i],
                    target_unitary_adj=target_matrix_adj_tf,
                    inverse_dim_squared=.25
                )

            infid_np = entanglement_fidelity(
                target=target_matrix,
                propagator=sample_matrices[i]
            )

            self.assertAlmostEqual(infid_np, infid_tf.numpy(), delta=1e-10)
