import math
import unittest
from qopt import matrix, cost_functions as q_fc, solver_algorithms, \
    transfer_function
import numpy as np

sig_0 = matrix.DenseOperator(np.eye(2))
sig_x = matrix.DenseOperator(np.asarray([[0, 1], [1, 0]]))
sig_y = matrix.DenseOperator(np.asarray([[0, -1j], [1j, 0]]))
sig_z = matrix.DenseOperator(np.asarray([[1, 0], [0, -1]]))


class TestEntanglementFidelity(unittest.TestCase):
    def test_entanglement_average_fidelity(self):
        a = 1 - q_fc.averge_gate_fidelity(sig_x, sig_y)
        b = 1 - q_fc.averge_gate_fidelity(sig_x, sig_x)
        c = 1 - q_fc.averge_gate_fidelity(
            sig_x, 1 / np.sqrt(2) * (sig_y + sig_x))

        a_e = 1 - q_fc.entanglement_fidelity(sig_x, sig_y)
        b_e = 1 - q_fc.entanglement_fidelity(sig_x, sig_x)
        c_e = 1 - q_fc.entanglement_fidelity(
            sig_x, 1 / np.sqrt(2) * (sig_y + sig_x))

        self.assertAlmostEqual(a * 3 / 2, a_e)
        self.assertAlmostEqual(b * 3 / 2, b_e)
        self.assertAlmostEqual(c * 3 / 2, c_e)

        e_sig_x = sig_x.exp(tau=.25j * math.pi)
        e_sig_y = sig_y.exp(tau=.25j * math.pi)

        c = q_fc.averge_gate_fidelity(e_sig_x, e_sig_y)
        d = q_fc.averge_gate_fidelity(e_sig_x, e_sig_x)

        self.assertAlmostEqual(c, .5)
        self.assertAlmostEqual(d, 1)

        # super operator formalism
        a_s = 1 - q_fc.entanglement_fidelity_super_operator(
            sig_x,
            matrix.convert_unitary_to_super_operator(sig_y))
        b_s = 1 - q_fc.entanglement_fidelity_super_operator(
            sig_x,
            matrix.convert_unitary_to_super_operator(sig_x))
        c_s = 1 - q_fc.entanglement_fidelity_super_operator(
            sig_x,
            matrix.convert_unitary_to_super_operator(
                1 / np.sqrt(2) * (sig_y + sig_x)))

        self.assertEqual(a_e, a_s)
        self.assertEqual(b_e, b_s)
        self.assertEqual(c_e, c_s)

    def test_gradient_calculation(self):
        # constants

        num_x = 12
        num_ctrl = 2

        over_sample_rate = 8
        bound_type = ("n", 5)
        num_u = over_sample_rate * num_x + 2 * bound_type[1]

        tau = [100e-9 for _ in range(num_x)]
        lin_freq_rel = 5.614e-4 * 1e6 * 1e3

        h_ctrl = [.5 * 2 * math.pi * sig_x,
                  .5 * 2 * math.pi * sig_y]
        h_drift = [
            matrix.DenseOperator(np.zeros((2, 2), dtype=complex))
            for _ in range(num_u)]

        # trivial transfer function

        T = np.diag(num_x * [lin_freq_rel])
        T = np.expand_dims(T, 2)
        linear_transfer_function = \
            transfer_function.CustomTF(T, num_ctrls=2)
        exponential_saturation_transfer_function = \
            transfer_function.ExponentialTF(
                awg_rise_time=.2 * tau[0],
                oversampling=over_sample_rate,
                bound_type=bound_type,
                num_ctrls=2
            )
        concatenated_tf = transfer_function.ConcatenateTF(
            tf1=linear_transfer_function,
            tf2=exponential_saturation_transfer_function
        )
        concatenated_tf.set_times(np.asarray(tau))

        # t_slot_comp
        seed = 1
        np.random.seed(seed)

        initial_pulse = 4 * np.random.rand(num_x, num_ctrl) - 2
        initial_ctrl_amps = concatenated_tf(initial_pulse)

        initial_state = matrix.DenseOperator(
            np.eye(2, dtype=complex))

        tau_u = [100e-9 / over_sample_rate for _ in range(num_u)]

        t_slot_comp = solver_algorithms.SchroedingerSolver(
            h_drift=h_drift,
            h_ctrl=h_ctrl,
            initial_state=initial_state,
            tau=tau_u,
            ctrl_amps=initial_ctrl_amps,
            calculate_propagator_derivatives=False)

        target = matrix.DenseOperator(
            (1j * sig_x + np.eye(2, dtype=complex)) * (1 / np.sqrt(2)))

        fidelity_computer = q_fc.OperationInfidelity(
            solver=t_slot_comp,
            target=target,
            fidelity_measure='entanglement')

        initial_cost = fidelity_computer.costs()
        cost = np.zeros(shape=(3, num_ctrl, num_x))

        delta_amp = 1e-3
        for i in range(num_x):
            for j in range(num_ctrl):
                cost[1, j, i] = np.real(initial_cost)
                delta = np.zeros(shape=initial_pulse.shape)
                delta[i, j] = -1 * delta_amp
                t_slot_comp.set_optimization_parameters(
                    concatenated_tf(initial_pulse + delta))
                cost[0, j, i] = fidelity_computer.costs()
                delta[i, j] = delta_amp
                t_slot_comp.set_optimization_parameters(
                    concatenated_tf(initial_pulse + delta))
                cost[2, j, i] = fidelity_computer.costs()

        numeric_gradient = np.gradient(cost, delta_amp, axis=0)
        t_slot_comp.set_optimization_parameters(concatenated_tf(initial_pulse))
        grad = fidelity_computer.grad()
        np.expand_dims(grad, 1)
        analytic_gradient = concatenated_tf.gradient_chain_rule(
            np.expand_dims(grad, 1))

        self.assertLess(np.sum(
            np.abs(numeric_gradient[1].T - analytic_gradient.squeeze(1))), 1e-6)

        # super operators
        dissipation_sup_op = [matrix.DenseOperator(
            np.zeros((4, 4)))]
        initial_state_sup_op = matrix.DenseOperator(np.eye(4))
        lindblad_tslot_obj = solver_algorithms.LindbladSolver(
            h_ctrl=h_ctrl, h_drift=h_drift, tau=tau_u,
            initial_state=initial_state_sup_op, ctrl_amps=initial_ctrl_amps,
            initial_diss_super_op=dissipation_sup_op,
            calculate_unitary_derivatives=False
        )

        fid_comp_sup_op = q_fc.OperationInfidelity(
            solver=lindblad_tslot_obj,
            target=target,
            fidelity_measure='entanglement',
            super_operator_formalism=True
        )

        t_slot_comp.set_optimization_parameters(initial_ctrl_amps)
        lindblad_tslot_obj.set_optimization_parameters(initial_ctrl_amps)

        self.assertAlmostEqual(fidelity_computer.costs(),
                               fid_comp_sup_op.costs())
        np.testing.assert_array_almost_equal(fid_comp_sup_op.grad(),
                                             fidelity_computer.grad())


class TestMatrixDistance(unittest.TestCase):
    def test_angle_axis_representation(self):
        beta = .25 * np.pi
        n = np.asarray([1 / np.sqrt(2), .5, .5])
        u = sig_0 * np.cos(beta) + \
            1j * (n[0] * sig_x + n[1] * sig_y + n[2] * sig_z) * np.sin(beta)
        beta_calc, n_calc = q_fc.angle_axis_representation(
            u.data)

        self.assertEqual(2 * beta, beta_calc)
        np.testing.assert_array_almost_equal(n, n_calc)

    """
    def test_gradient_calculation(self):
        # constants

        num_x = 12
        num_ctrl = 2

        over_sample_rate = 8
        bound_type = ("n", 5)
        num_u = over_sample_rate * num_x + 2 * bound_type[1]

        tau = [100e-9 for _ in range(num_x)]
        lin_freq_rel = 5.614e-4 * 1e6 * 1e3

        h_ctrl = [.5 * 2 * math.pi * matrix.OperatorDense(
            qutip.sigmax()),
                  .5 * 2 * math.pi * matrix.OperatorDense(
                      qutip.sigmay())]
        h_drift = [
            matrix.OperatorDense(np.zeros((2, 2), dtype=complex))
            for _ in range(num_u)]

        # trivial transfer function

        T = np.diag(num_x * [lin_freq_rel])
        linear_transfer_function = \
            transfer_function.CustomTF(T)
        exponential_saturation_transfer_function = \
            transfer_function.ExponentialTF(
                awg_rise_time=.2 * tau[0],
                oversampling=over_sample_rate,
                bound_type=bound_type
            )
        concatenated_tf = transfer_function.ConcatenateTF(
            tf1=linear_transfer_function,
            tf2=exponential_saturation_transfer_function
        )
        concatenated_tf.set_times(np.asarray(tau))

        # t_slot_comp
        seed = 1
        np.random.seed(seed)

        initial_pulse = 4 * np.random.rand(num_x, num_ctrl) - 2
        initial_ctrl_amps = concatenated_tf(initial_pulse)

        initial_state = matrix.OperatorDense(
            np.eye(2, dtype=complex))

        tau_u = [100e-9 / over_sample_rate for _ in range(num_u)]

        t_slot_comp = solver_algorithms.SchroedingerSolver(
            h_drift=h_drift,
            h_ctrl=h_ctrl,
            initial_state=initial_state,
            tau=tau_u,
            ctrl_amps=initial_ctrl_amps)

        target = matrix.OperatorDense(
            (np.eye(2, dtype=complex) + 1j * qutip.sigmax()) / np.sqrt(2))

        fidelity_computer = q_fc.OperatorMatrixNorm(
            t_slot_comp=t_slot_comp,
            target=target,
            mode='vector')

        initial_cost = fidelity_computer.costs()
        cost = np.zeros(shape=(3, num_ctrl, num_x))

        delta_amp = 1e-3
        for k in range(8):
            for i in range(num_x):
                for j in range(num_ctrl):
                    cost[1, j, i] = np.real(initial_cost[k])
                    delta = np.zeros(shape=initial_pulse.shape)
                    delta[i, j] = -1 * delta_amp
                    t_slot_comp.set_optimization_parameters(
                        concatenated_tf(initial_pulse + delta))
                    cost[0, j, i] = fidelity_computer.costs()[k]
                    delta[i, j] = delta_amp
                    t_slot_comp.set_optimization_parameters(
                        concatenated_tf(initial_pulse + delta))
                    cost[2, j, i] = fidelity_computer.costs()[k]

            numeric_gradient = np.gradient(cost, delta_amp, axis=0)
            t_slot_comp.set_optimization_parameters(concatenated_tf(initial_pulse))
            grad = fidelity_computer.grad()
            np.expand_dims(grad, 1)
            analytic_gradient = concatenated_tf.gradient_u2x(
                np.expand_dims(grad, 1))

            self.assertLess(np.sum(
                np.abs(
                    numeric_gradient[1].T - analytic_gradient.squeeze(1)[
                                            :, k])), 1e-6)
    """
