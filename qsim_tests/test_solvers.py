from qsim import noise, matrix, solver_algorithms
import qutip
import numpy as np
import math
import unittest


class TestTslots(unittest.TestCase):

    def test_comp_all_save_exp(self):
        h_drift = [
            matrix.DenseOperator(np.zeros((2, 2), dtype=complex))
            for _ in range(4)]
        h_control = [.5 * matrix.DenseOperator(
            qutip.sigmax()),
                     .5 * matrix.DenseOperator(
                          qutip.sigmaz())]

        ctrl_amps = np.asarray([[.5, 0, .25, .25], [0, .5, 0, 0]]).transfer_matrix * 2 * np.pi
        n_t = 4
        tau = [1 for _ in range(4)]
        initial_state = matrix.DenseOperator(np.eye(2)) \
                        * (1 + 0j)
        tslot_obj = solver_algorithms.SchroedingerSolver(
            h_ctrl=h_control, h_drift=h_drift, tau=tau,
            initial_state=initial_state, ctrl_amps=ctrl_amps,
            calculate_propagator_derivatives=True)

        # test the propagators
        # use the exponential identity for pauli operators for verification
        correct_props = [[[0. + 0.j, 0. - 1.j], [0. - 1.j, 0. + 0.j]],
                         [[0. - 1.j, 0. + 0.j], [0. + 0.j, 0. + 1.j]],
                         [[0.70710678 + 0.j, 0. - 0.70710678j],
                          [0. - 0.70710678j, 0.70710678 + 0.j]],
                         [[0.70710678 + 0.j, 0. - 0.70710678j],
                          [0. - 0.70710678j, 0.70710678 + 0.j]]]
        correct_props = list(map(np.asarray, correct_props))
        propagators = tslot_obj.propagators
        for i in range(n_t):
            np.testing.assert_array_almost_equal(propagators[i].data,
                                                 correct_props[i])

        # test the forward propagation
        correct_fwd_prop = [initial_state.data, ]
        for prop in correct_props:
            correct_fwd_prop.append(prop @ correct_fwd_prop[-1])
        forward_propagators = tslot_obj.forward_propagators
        for i in range(n_t + 1):
            np.testing.assert_array_almost_equal(forward_propagators[i].data,
                                                 correct_fwd_prop[i])

        # test the reverse propagation
        # This functionality is currently not used / supported

        correct_rev_prop = [initial_state.data]
        for prop in correct_props[::-1]:
            correct_rev_prop.append(correct_rev_prop[-1] @ prop)
        reverse_propagators = tslot_obj.reversed_propagators
        for i in range(n_t + 1):
            np.testing.assert_array_almost_equal(reverse_propagators[i].data,
                                                 correct_rev_prop[i])

        sx = matrix.DenseOperator(.5 * qutip.sigmax())
        A = sx * -1j * .5 * 2 * math.pi
        B = sx * -1j
        prop, deriv = A.dexp(direction=B, tau=1, compute_expm=True)
        np.testing.assert_array_almost_equal(
            deriv.data, tslot_obj.frechet_deriv_propagators[0][0].data)
        analytic_reference = -.5 * np.eye(2, dtype=complex)
        np.testing.assert_array_almost_equal(
            analytic_reference, tslot_obj.frechet_deriv_propagators[0][0].data)

    def test_save_all_noise(self):
        h_drift = [
            matrix.DenseOperator(np.zeros((2, 2), dtype=complex))
            for _ in range(4)]
        h_control = [.5 * matrix.DenseOperator(
            qutip.sigmax()), ]
        h_noise = [.5 * matrix.DenseOperator(
            qutip.sigmaz()), ]
        ctrl_amps = np.asarray([[.5, 0, .25, .25], ]).transfer_matrix * 2 * math.pi
        tau = [1, 1, 1, 1]
        n_t = len(tau)
        initial_state = matrix.DenseOperator(np.eye(2)) \
                        * (1 + 0j)
        mocked_noise = np.asarray([0, .5, 0, 0]) * 2 * math.pi
        mocked_noise = np.expand_dims(mocked_noise, 0)
        mocked_noise = np.expand_dims(mocked_noise, 0)

        noise_sample_generator = noise.NTGQuasiStatic(
            noise_samples=mocked_noise, n_samples_per_trace=4,
            standard_deviation=[2, ], n_traces=1, always_redraw_samples=False)

        time_slot_noise = solver_algorithms.SchroedingerSMonteCarlo(
            h_drift=h_drift,
            h_ctrl=h_control,
            initial_state=initial_state,
            tau=tau,
            ctrl_amps=ctrl_amps,
            h_noise=h_noise,
            noise_trace_generator=noise_sample_generator
        )

        # test the propagators
        # use the exponential identity for pauli operators for verification
        correct_props = [[[0. + 0.j, 0. - 1.j], [0. - 1.j, 0. + 0.j]],
                         [[0. - 1.j, 0. + 0.j], [0. + 0.j, 0. + 1.j]],
                         [[0.70710678 + 0.j, 0. - 0.70710678j],
                          [0. - 0.70710678j, 0.70710678 + 0.j]],
                         [[0.70710678 + 0.j, 0. - 0.70710678j],
                          [0. - 0.70710678j, 0.70710678 + 0.j]]]
        correct_props = list(map(np.asarray, correct_props))
        propagators = time_slot_noise.propagators_noise[0]
        for i in range(n_t):
            np.testing.assert_array_almost_equal(propagators[i].data,
                                                 correct_props[i])

        # test the forward propagation
        correct_fwd_prop = [initial_state.data, ]
        for prop in correct_props:
            correct_fwd_prop.append(prop @ correct_fwd_prop[-1])
        forward_propagators = time_slot_noise.forward_propagators_noise[0]
        for i in range(n_t + 1):
            np.testing.assert_array_almost_equal(forward_propagators[i].data,
                                                 correct_fwd_prop[i])

        # test the reverse propagation
        # This functionality is currently not used / supported

        correct_rev_prop = [initial_state]
        for prop in correct_props[::-1]:
            correct_rev_prop.append(correct_rev_prop[-1] * prop)
        reverse_propagators = time_slot_noise.reversed_propagators_noise[0]
        for i in range(n_t + 1):
            np.testing.assert_array_almost_equal(reverse_propagators[i].data,
                                                 correct_rev_prop[i].data)

        sx = matrix.DenseOperator(.5 * qutip.sigmax())
        A = sx * -1j * .5 * 2 * math.pi
        B = sx * -1j
        prop, deriv = A.dexp(direction=B, tau=1, compute_expm=True)
        np.testing.assert_array_almost_equal(
            deriv.data,
            time_slot_noise.frechet_deriv_propagators_noise[0][0][0].data)

    def test_lindblad_no_dissipation(self):
        """
        Use the previous test to verify the Lindblad master equation in absence
        of decoherence terms.
        """

        h_drift = [
            matrix.DenseOperator(np.zeros((2, 2), dtype=complex))
            for _ in range(4)]
        h_control = [.5 * matrix.DenseOperator(qutip.sigmax()),
                     .5 * matrix.DenseOperator(qutip.sigmaz())]
        ctrl_amps = np.asarray(
            [[.5, 0, .25, .25], [0, .5, 0, 0]]).transfer_matrix * 2 * math.pi
        tau = [1, 1, 1, 1]

        dissipation_sup_op = [matrix.DenseOperator(
            np.zeros((4, 4)))]
        initial_state = matrix.DenseOperator(np.eye(4))

        lindblad_tslot_obj = solver_algorithms.LindbladSolver(
            h_ctrl=h_control, h_drift=h_drift, tau=tau,
            initial_state=initial_state, ctrl_amps=ctrl_amps,
            initial_diss_super_op=dissipation_sup_op,
            calculate_unitary_derivatives=True
        )

        propagators = lindblad_tslot_obj.propagators

        # test the propagators
        # use the exponential identity for pauli operators for verification
        correct_props = [[[0. + 0.j, 0. - 1.j], [0. - 1.j, 0. + 0.j]],
                         [[0. - 1.j, 0. + 0.j], [0. + 0.j, 0. + 1.j]],
                         [[0.70710678 + 0.j, 0. - 0.70710678j],
                          [0. - 0.70710678j, 0.70710678 + 0.j]],
                         [[0.70710678 + 0.j, 0. - 0.70710678j],
                          [0. - 0.70710678j, 0.70710678 + 0.j]]]
        correct_props = list(map(np.asarray, correct_props))
        correct_props = [np.kron(cp.conj(), cp) for cp in correct_props]

        for i in range(4):
            np.testing.assert_array_almost_equal(correct_props[i],
                                                 propagators[i].data)

        analytic_reference = -.5j * np.eye(2, dtype=complex)
        sx = np.asarray([[0, 1], [1, 0]]) * (1 + 0j)
        analytic_reference = np.kron(sx, analytic_reference) \
            - np.kron(analytic_reference, sx)
        np.testing.assert_array_almost_equal(
            analytic_reference,
            lindblad_tslot_obj.frechet_deriv_propagators[0][0].data)

    def test_lindblad_false_dissipation(self):
        """
        Disguise the dissipation less evolution by using the dissipation
        operators.
        """
        # method 1.
        h_drift = [
            matrix.DenseOperator(np.zeros((2, 2), dtype=complex))
            for _ in range(4)]
        h_control = [.5 * matrix.DenseOperator(qutip.sigmax()),
                     .5 * matrix.DenseOperator(qutip.sigmaz())]
        ctrl_amps = np.asarray(
            [[.5, 0, .25, .25], [0, .5, 0, 0]]).transfer_matrix * 2 * math.pi
        tau = [1, 1, 1, 1]

        def prefactor_function(_):
            return ctrl_amps

        identity = h_control[0].identity_like()
        dissipation_sup_op = [(identity.kron(h) - h.kron(identity)) * -1j
                              for h in h_control]
        h_control = [matrix.DenseOperator(np.zeros((2, 2)))]
        initial_state = matrix.DenseOperator(np.eye(4))

        lindblad_tslot_obj = solver_algorithms.LindbladSolver(
            h_ctrl=h_control, h_drift=h_drift, tau=tau,
            initial_state=initial_state, ctrl_amps=ctrl_amps,
            initial_diss_super_op=dissipation_sup_op,
            calculate_unitary_derivatives=False,
            prefactor_function=prefactor_function
        )

        propagators = lindblad_tslot_obj.propagators

        # test the propagators
        # use the exponential identity for pauli operators for verification
        correct_props = [[[0. + 0.j, 0. - 1.j], [0. - 1.j, 0. + 0.j]],
                         [[0. - 1.j, 0. + 0.j], [0. + 0.j, 0. + 1.j]],
                         [[0.70710678 + 0.j, 0. - 0.70710678j],
                          [0. - 0.70710678j, 0.70710678 + 0.j]],
                         [[0.70710678 + 0.j, 0. - 0.70710678j],
                          [0. - 0.70710678j, 0.70710678 + 0.j]]]
        correct_props = list(map(np.asarray, correct_props))
        correct_props = [np.kron(cp.conj(), cp) for cp in correct_props]

        for i in range(4):
            np.testing.assert_array_almost_equal(correct_props[i],
                                                 propagators[i].data)


        # method 3

        def diss_sup_op_func(_, _2):
            return lindblad_tslot_obj._diss_sup_op

        similar_lindblad_tslot_onj = solver_algorithms.LindbladSolver(
            h_ctrl=h_control, h_drift=h_drift, tau=tau,
            initial_state=initial_state, ctrl_amps=ctrl_amps,
            initial_diss_super_op=dissipation_sup_op,
            super_operator_function=diss_sup_op_func
        )

        propagators = similar_lindblad_tslot_onj.propagators

        for i in range(4):
            np.testing.assert_array_almost_equal(correct_props[i],
                                                 propagators[i].data)

        # method 2

        lindblad = [matrix.DenseOperator(
            np.asarray([[0, 1j], [1, 0]], dtype=complex))]

        lind_lindblad_tslot_obj = solver_algorithms.LindbladSolver(
            h_ctrl=h_control, h_drift=h_drift, tau=tau,
            initial_state=initial_state, ctrl_amps=ctrl_amps,
            initial_diss_super_op=dissipation_sup_op,
            lindblad_operators=lindblad
        )

        diss_sup_op = lind_lindblad_tslot_obj._calc_diss_sup_op()

        li = lindblad[0]
        correct_diss_sup_op = li.conj(True).kron(li) \
            - .5 * identity.kron(li * li.dag()) \
            - .5 * (li.transpose() * li.conj()).kron(identity)

        for i in range(4):
            np.testing.assert_array_almost_equal(diss_sup_op[i].data,
                                                 correct_diss_sup_op.data)
