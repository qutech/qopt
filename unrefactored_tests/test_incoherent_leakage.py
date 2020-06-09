import numpy as np
from qsim.matrix import DenseOperator
from qsim.noise import NTGQuasiStatic
from qsim.solver_algorithms import SchroedingerSMonteCarlo
from qsim.cost_functions import IncoherentLeakageError

sigma_z = DenseOperator(np.diag([1, -1]))
sigma_x = DenseOperator(np.asarray([[0, 1], [1, 0]]))

h_ctrl = [sigma_z.kron(sigma_x.identity_like())]
h_drift = [0 * sigma_x.kron(sigma_x)]

ntg = NTGQuasiStatic(
    standard_deviation=[.3],
    n_traces=5,
    n_samples_per_trace=3
)

ctrl_amplitudes = np.pi * np.ones((3, 1))

solver = SchroedingerSMonteCarlo(
    h_ctrl=h_ctrl,
    h_drift=h_drift * 3,
    h_noise=[sigma_x.kron(sigma_z)],
    noise_trace_generator=ntg,
    initial_state=np.eye(4),
    tau=[1, 1, 1]
)

incoherent_l = IncoherentLeakageError(
    solver=solver,
    computational_states=[0, 1]
)

solver.set_optimization_parameters(ctrl_amplitudes)
incoherent_l.costs()