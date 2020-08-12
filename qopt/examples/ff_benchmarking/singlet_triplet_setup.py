import numpy as np

from qopt.matrix import DenseOperator
from qopt.solver_algorithms import SchroedingerSolver, LindbladSolver
from qopt.amplitude_functions import CustomAmpFunc
from qopt.cost_functions import OperatorFilterFunctionInfidelity, \
    OperationInfidelity
from qopt.simulator import Simulator


pauli_x = DenseOperator(np.asarray([[0, 1], [1, 0]]))
pauli_z = DenseOperator(np.diag([1, -1]))

x_pi_half = (.5 * pauli_x).exp(.5j * np.pi)

J_0 = 1.
EPS_0 = 1.
S_0 = 1.
S_0_WHITE = 1.


def exchange_interaction(eps, j_0=J_0, eps_0=EPS_0):
    return j_0 * np.exp(eps / eps_0)


def deriv_exchange_interaction(eps, j_0=J_0, eps_0=EPS_0):
    return j_0 / eps_0 * np.exp(eps / eps_0)


def deriv_2_exchange_interaction(eps, j_0=J_0, eps_0=EPS_0):
    return j_0 / eps_0 ** 2 * np.exp(eps / eps_0)


def one_over_f_noise(f, s_0=S_0):
    return s_0 / f


def white_psd(s_0=S_0_WHITE):
    def psd(f):
        return s_0 * np.ones_like(f)
    return psd


def create_simulators(
        dbz=1.,
        n_time_steps=10,
        time_step=1,
        j_0=J_0,
        eps_0=EPS_0,
        psd=white_psd(),
        omega=None,
        s_0_white=S_0_WHITE,
        target=x_pi_half
):
    h_ctrl = [.5 * pauli_x, ]
    h_drift = [dbz * .5 * pauli_z, ]

    def noise_hamiltonian(eps):
        h_n = [.5 * pauli_x,
               deriv_exchange_interaction(eps=eps, j_0=j_0, eps_0=eps_0)]
        return h_n

    def s_derivs(eps):
        s_d = deriv_2_exchange_interaction(eps=eps, j_0=j_0, eps_0=eps_0)
        s_d = np.expand_dims(s_d, 0)
        return s_d

    amp_func = CustomAmpFunc(
        value_function=exchange_interaction,
        derivative_function=deriv_exchange_interaction
    )

    solver = SchroedingerSolver(
        h_drift=h_drift,
        h_ctrl=h_ctrl,
        tau=time_step * np.ones(n_time_steps),
        filter_function_h_n=noise_hamiltonian,
        filter_function_s_derivs=s_derivs,
        amplitude_function=amp_func
    )

    syst_infid = OperationInfidelity(
        solver=solver,
        target=target
    )

    ff_infid = OperatorFilterFunctionInfidelity(
        solver=solver,
        noise_power_spec_density=psd,
        omega=omega
    )

    def prefactor_function(j, _):
        return s_0_white * (j / eps_0) ** 2

    def prefactor_function_derivative(j, _):
        return s_0_white * 2 * (j / eps_0) ** 2 / eps_0

    solver_lindblad = LindbladSolver(
        h_drift=h_drift,
        h_ctrl=h_ctrl,
        tau=time_step * np.ones(n_time_steps),
        lindblad_operators=h_ctrl,
        prefactor_function=prefactor_function,
        prefactor_derivative_function=prefactor_function_derivative
    )

    lindblad_infid = OperationInfidelity(
        solver=solver_lindblad,
        target=target,
        super_operator_formalism=True,
    )

    simulator_ff = Simulator(
        solvers=[solver, ],
        cost_fktns=[syst_infid, ff_infid]
    )

    simulator_l = Simulator(
        solvers=[solver_lindblad, ],
        cost_fktns=[lindblad_infid, ]
    )

    return simulator_ff, simulator_l
