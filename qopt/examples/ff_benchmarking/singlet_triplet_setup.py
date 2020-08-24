import numpy as np

from qopt.matrix import DenseOperator
from qopt.solver_algorithms import SchroedingerSolver, LindbladSolver
from qopt.amplitude_functions import CustomAmpFunc
from qopt.cost_functions import OperatorFilterFunctionInfidelity, \
    OperationInfidelity
from qopt.simulator import Simulator
from qopt.optimize import LeastSquaresOptimizer


pauli_x = DenseOperator(np.asarray([[0, 1], [1, 0]]))
pauli_z = DenseOperator(np.diag([1, -1]))

x_pi_half = (.5 * pauli_x).exp(.5j * np.pi)

N_TIME_STEPS = 10
TIME_STEP = 1
DBZ = 1.
J_0 = 1.  # in ns^-1
EPS_0 = 1.  # by norming to eps 0
EPS_MIN = -5.4  # in eps_0
EPS_MAX = 2.4  # in eps_0
EPSILON_0 = .272  # mV
SIGMA_EPS = 8e-3 / EPSILON_0  # mV
normation_factor = 1 / ((2 * np.pi) ** 2) / (EPSILON_0 ** 2)
S_0_WHITE = 4e-5 * normation_factor  # in eps_0
S_0_PINK = 4e-5 * normation_factor

exponential_method = 'Frechet'


def exchange_interaction(eps, j_0=J_0, eps_0=EPS_0):
    return j_0 * np.exp(eps / eps_0)


def deriv_exchange_interaction(eps, j_0=J_0, eps_0=EPS_0):
    return j_0 / eps_0 * np.exp(eps / eps_0)


def deriv_2_exchange_interaction(eps, j_0=J_0, eps_0=EPS_0):
    return j_0 / eps_0 ** 2 * np.exp(eps / eps_0)


def one_over_f_noise(f, s_0=S_0_PINK):
    return s_0 / f


def white_psd(s_0=S_0_WHITE):
    def psd(f):
        return s_0 * np.ones_like(f)
    return psd


def create_simulators(
        dbz=DBZ,
        n_time_steps=N_TIME_STEPS,
        time_step=TIME_STEP,
        j_0=J_0,
        eps_0=EPS_0,
        psd=white_psd(2 * S_0_WHITE),
        # The factor of two is inserted to achieve the same values as lindblad
        omega=None,
        s_0_white=S_0_WHITE,
        target=x_pi_half,
        cost_fkts_weights=None
):
    h_ctrl = [.5 * pauli_x, ]
    h_drift = [dbz * .5 * pauli_z, ] * n_time_steps

    def noise_hamiltonian(eps):
        h_n = [[.5 * pauli_x.data,
                deriv_exchange_interaction(
                   eps=eps, j_0=j_0, eps_0=eps_0).flatten()],
               ]
        return h_n

    def s_derivs(eps):
        """Derivatives of the noise susceptibilites s by the control amplitudes.

        After the application of the transfer function! """
        # s_d = deriv_2_exchange_interaction(eps=eps, j_0=j_0, eps_0=eps_0)
        s_d = np.ones_like(eps) / eps_0
        s_d = np.expand_dims(s_d, 0)
        return s_d.transpose((0, 2, 1))

    def derivative_function(eps, j_0_local=j_0, eps_0_local=eps_0):
        x = deriv_exchange_interaction(eps, j_0=j_0_local, eps_0=eps_0_local)
        x = np.expand_dims(x, 1)
        return x

    amp_func = CustomAmpFunc(
        value_function=exchange_interaction,
        derivative_function=derivative_function
    )

    solver = SchroedingerSolver(
        h_drift=h_drift,
        h_ctrl=h_ctrl,
        tau=time_step * np.ones(n_time_steps),
        filter_function_h_n=noise_hamiltonian,
        filter_function_s_derivs=s_derivs,
        amplitude_function=amp_func,
        exponential_method=exponential_method
    )

    solver.set_optimization_parameters(np.zeros((n_time_steps, 1)))

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
        return np.expand_dims(s_0_white * 2 * (j / eps_0) ** 2 / eps_0, 1)

    solver_lindblad = LindbladSolver(
        h_drift=h_drift,
        h_ctrl=h_ctrl,
        tau=time_step * np.ones(n_time_steps),
        lindblad_operators=h_ctrl,
        prefactor_function=prefactor_function,
        prefactor_derivative_function=prefactor_function_derivative,
        amplitude_function=amp_func
    )

    lindblad_infid = OperationInfidelity(
        solver=solver_lindblad,
        target=target,
        super_operator_formalism=True,
    )

    simulator_ff = Simulator(
        solvers=[solver, ],
        cost_fktns=[syst_infid, ff_infid],
        cost_fktn_weights=cost_fkts_weights
    )

    simulator_l = Simulator(
        solvers=[solver_lindblad, ],
        cost_fktns=[lindblad_infid, ]
    )

    simulator_syst = Simulator(
        solvers=[solver, ],
        cost_fktns=[syst_infid, ]
    )

    simulator_ff_only = Simulator(
        solvers=[solver, ],
        cost_fktns=[ff_infid, ]
    )

    return simulator_ff, simulator_l, simulator_syst, simulator_ff_only


def create_optimizer(
        simulator, max_iterations=1000, max_wall_time=30,
        save_intermediary_steps=True, bounds=None, use_jac_fctn=True,
        cost_fkts_weights=None
):
    termination_conditions = {
        "min_gradient_norm": 1e-8,
        "min_cost_gain": 1e-6,
        "max_wall_time": max_wall_time,
        "max_cost_func_calls": 1e6,
        "max_iterations": max_iterations,
        "min_amplitude_change": 1e-8
    }
    if bounds is None:
        bounds = [EPS_MIN, EPS_MAX]
    optimizer = LeastSquaresOptimizer(
        system_simulator=simulator, termination_cond=termination_conditions,
        save_intermediary_steps=save_intermediary_steps, bounds=bounds,
        use_jacobian_function=use_jac_fctn, cost_fktn_weights=cost_fkts_weights
    )
    return optimizer
