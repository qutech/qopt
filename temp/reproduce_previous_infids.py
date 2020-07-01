import numpy as np
import unittest
import os

from scipy.io import loadmat
import filter_functions as ff
import temp.ff_testutil as ff_testutil

from qopt.transfer_function import OversamplingTF
from qopt.amplitude_functions import UnaryAnalyticAmpFunc
from qopt.solver_algorithms import SchroedingerSolver, \
    LindbladSolver, SchroedingerSMCControlNoise
from qopt.noise import NTGColoredNoise
from qopt.cost_functions import OperationInfidelity, \
    OperationNoiseInfidelity, OperatorFilterFunctionInfidelity

from temp.GaAsExchangeCoupledCerf import \
    CONSTANTS, OPERATORS, INFIDELITIES, REQUIRED_ACCURACY, \
    create_diss_super_op_fkt
import temp.GaAsExchangeCoupledCerf as GaAs


# subspace
sub_space_ind = np.ix_([3, 5, 6, 9, 10, 12], [3, 5, 6, 9, 10, 12])
comp_sub_sub_space_ind = [1, 2, 3, 4]

# Data loaded by Tobias
data_path = r"C:\Users\Inst2C\Documents\python-filter_functions\examples\data"
struct = loadmat(os.path.join(data_path, 'CNOT.mat'))
eps = np.asarray(struct['eps'], order='C')
dt = np.asarray(struct['t'].ravel(), order='C')
cnot_infid_fast = struct['infid_fast'].ravel()
B = np.asarray(struct['B'].ravel(), order='C')
B_avg = struct['BAvg'].ravel()

# Frequently used auxiliary classes
id_tf = OversamplingTF(num_ctrls=3)
id_tf.set_times(dt)

exp_amp_func = UnaryAnalyticAmpFunc(
    value_function=np.exp, derivative_function=np.exp)


class CompareToPreviousResults(unittest.TestCase):
    """
    Calculate infidelities in multiple ways and compare with the results of
    Dr. Cerfontaine.

    The entanglement fidelity is compared with the sum of the previous
    systematic deviation and the leakage, because in the previous optimization,
    the evolution was mapped to the closest unitary.

    """
    def test_filter_functions(self):

        # Basis for qubit subspace
        qubit_subspace_basis = ff.Basis(
            [np.pad(b, 1, 'constant') for b in ff.Basis.pauli(2)],
            skip_check=True,
            btype='Pauli'
        )

        c_opers = ff_testutil.subspace_opers
        n_opers = c_opers
        c_coeffs, n_coeffs = ff_testutil.c_coeffs, ff_testutil.n_coeffs
        dt = ff_testutil.dt
        infid_MC = ff_testutil.cnot_infid_fast
        A = ff_testutil.A

        identifiers = ['eps_12', 'eps_23', 'eps_34', 'b_12', 'b_23', 'b_34']
        H_c = list(zip(c_opers[:3] + [c_opers[3] + 7 * c_opers[4] - c_opers[5]],
                       c_coeffs[:3] + [c_coeffs[3]], identifiers[:4]))
        H_n = list(zip(n_opers[:3], n_coeffs[:3], identifiers[:3]))
        cnot = ff.PulseSequence(H_c, H_n, dt, basis=qubit_subspace_basis)

        T = dt.sum()
        omega = np.logspace(np.log10(1/T), 2, 125)
        S_t, omega_t = ff.util.symmetrize_spectrum(A[0]/omega**0.0, omega)
        infid, xi = ff.infidelity(cnot, S_t, omega_t, identifiers[:3],
                                  return_smallness=True)
        # infid scaled with d = 6, but we actually have d = 4
        infid *= 1.5
        self.assertLessEqual(np.abs(1 - (infid.sum()/infid_MC[0])), .4)
        self.assertLessEqual(infid.sum(), xi**2/4)

        time_slot_comp_closed = SchroedingerSolver(
            h_drift=[OPERATORS['h_drift']] * len(dt),
            h_ctrl=OPERATORS['h_ctrl'],
            initial_state=OPERATORS['initial_state'],
            tau=list(dt),
            calculate_propagator_derivatives=True,
            exponential_method='spectral',
            is_skew_hermitian=True,
            transfer_function=id_tf,
            amplitude_function=exp_amp_func,
            filter_function_h_n=H_n,
            filter_function_basis=qubit_subspace_basis
        )
        time_slot_comp_closed.set_optimization_parameters(eps.T)

        ff_infid = OperatorFilterFunctionInfidelity(
            solver=time_slot_comp_closed,
            noise_power_spec_density=S_t,
            omega=omega_t
        )
        print(ff_infid.grad())
        np.testing.assert_array_almost_equal(infid, ff_infid.costs()*1.5)

    def test_fast_noise_master_equation(self):
        """
        Changing the oversampling does not change the result.

        """
        infids = []
        for noise_damping in .1 * np.arange(10):
            diss_super_op_fkt = create_diss_super_op_fkt(4e-5 * noise_damping)
            diss_super_op_deriv_fkt = GaAs.diss_super_op_deriv_fkt

            time_slot_lindblad = LindbladSolver(
                h_drift=[OPERATORS['h_drift']] * len(dt),
                h_ctrl=OPERATORS['h_ctrl'],
                initial_state=OPERATORS['initial_state_sup_op'], tau=list(dt),
                is_skew_hermitian=False, super_operator_function=diss_super_op_fkt,
                super_operator_derivative_function=diss_super_op_deriv_fkt,
                exponential_method='Frechet',
                transfer_function=id_tf,
                amplitude_function=exp_amp_func)
            time_slot_lindblad.set_optimization_parameters(eps.T)
            sub_space_ind_super_op = [el1 + el2 * 6
                                      for el2 in comp_sub_sub_space_ind
                                      for el1 in comp_sub_sub_space_ind]

            control_me_infid = OperationInfidelity(
                solver=time_slot_lindblad, target=OPERATORS['CNOT_4'],
                fidelity_measure='entanglement', super_operator_formalism=True,
                index=['white_noise_lindblad'],
                computational_states=sub_space_ind_super_op
            )
            infids.append(control_me_infid.costs())
        self.assertLess(
            np.abs(1 - 4 / 5 * control_me_infid.costs() / INFIDELITIES[
                'fast_white']), REQUIRED_ACCURACY['fast_white_me'])
