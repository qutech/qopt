"""
This Test will compute a single qubit with T1 noise.

Step 1: We compare lindlblad master equation vs Monte
Carlo simulation vs filter functions.

Step 2: Compare the gradients to finite differences.

"""


from qopt import *
import numpy as np
import unittest


np.random.seed(0)
total_time = 2
noise_variance = 1e-3
n_time_steps = 3
fid_ctrl_amps = np.expand_dims(np.zeros(n_time_steps), 1)

# we need to change the pulse, otherwise some analytic gradients
# become 0 and we just get numerical errors.
gradient_pulse = fid_ctrl_amps + 1

bz_rotation = np.pi
target = (.5 * DenseOperator.pauli_x()).exp(1j * bz_rotation)

delta_bz = bz_rotation / total_time
delta_t = total_time / n_time_steps

h_drift = [delta_bz * .5 * DenseOperator.pauli_x()]
h_ctrl = [.5 * DenseOperator.pauli_z()]
time_steps = delta_t * np.ones(n_time_steps)


def create_lindblad_simulator():

    def prefactor_function(transferred_parameters, _):
        return noise_variance * np.ones_like(transferred_parameters)

    lindblad_solver = LindbladSolver(
        h_drift=h_drift,
        h_ctrl=h_ctrl,
        tau=time_steps,
        prefactor_function=prefactor_function,
        lindblad_operators=[.5 * DenseOperator.pauli_z()]
    )

    lindblad_cost_fkt = OperationInfidelity(
        solver=lindblad_solver,
        super_operator_formalism=True,
        target=target
    )
    lindblad_simulator = Simulator(
        solvers=[lindblad_solver, ],
        cost_funcs=[lindblad_cost_fkt, ]
    )
    return lindblad_simulator


def create_mc_simulator(low_freq_ex):

    def noise_spectral_density(f):
        return 2 * noise_variance * np.ones_like(f)  # factor of 2 for
        # one-sided spectrum

    noise_trace_generator = NTGColoredNoise(
        n_samples_per_trace=n_time_steps,
        dt=delta_t,
        noise_spectral_density=noise_spectral_density,
        n_traces=3000,
        low_frequency_extension_ratio=low_freq_ex
    )
    # The low frequency extension leads to the sampling of a noise trace
    # for more time steps. The simulation then uses only a section of this
    # longer trace. The extension allows the noise trace generator to include
    # low frequency noise.

    monte_carlo_solver = SchroedingerSMonteCarlo(
        h_drift=h_drift,
        h_ctrl=h_ctrl,
        tau=time_steps,
        h_noise=[.5 * DenseOperator.pauli_z()],
        noise_trace_generator=noise_trace_generator
    )

    mc_cost_fkt = OperationNoiseInfidelity(
        solver=monte_carlo_solver,
        target=target,
        neglect_systematic_errors=False
    )

    monte_carlo_simulator = Simulator(
        solvers=[monte_carlo_solver, ],
        cost_funcs=[mc_cost_fkt, ]
    )

    return monte_carlo_simulator


def create_ff_simulator(low_freq_rel, ff_n_time_steps):

    ff_hamiltonian_noise = [[
        .5 * DenseOperator.pauli_z().data,
        np.ones(ff_n_time_steps),
        'Noise 1'
    ], ]

    ff_solver = SchroedingerSolver(
        h_drift=h_drift,
        h_ctrl=h_ctrl,
        tau=(total_time / ff_n_time_steps) * np.ones(ff_n_time_steps),
        filter_function_h_n=ff_hamiltonian_noise
    )

    def noise_spectral_density(f):
        return 2 * noise_variance * np.ones_like(f)  # factor of 2 for
        # one-sided spectrum

    ff_cost_fkn = OperatorFilterFunctionInfidelity(
        solver=ff_solver,
        noise_power_spec_density=noise_spectral_density,
        omega=(1 / (total_time * low_freq_rel)) *
        (np.arange(ff_n_time_steps * low_freq_rel) + 1)
    )

    ff_simulator = Simulator(
        solvers=[ff_solver, ],
        cost_funcs=[ff_cost_fkn, ]
    )

    return ff_simulator


class TestT1Decay(unittest.TestCase):

    def test_compare_infidelitites_and_gradient_accuracy(self):
        np.random.seed(0)
        ff_n_time_steps = 20
        simulators = [
            create_lindblad_simulator(),
            create_mc_simulator(low_freq_ex=20),
            create_ff_simulator(
                low_freq_rel=100, ff_n_time_steps=ff_n_time_steps)
        ]

        infidelities = []
        gradient_accuracy = []

        ff_fid_ctrl_amps = np.expand_dims(np.zeros(ff_n_time_steps), 1)

        for sim, pulse, delta in zip(
            simulators,
            [fid_ctrl_amps, fid_ctrl_amps, ff_fid_ctrl_amps],
            [1e-8, 1e-1, 1e-8]
        ):
            infidelities.append(sim.wrapped_cost_functions(pulse))
            gradient_accuracy.append(
                sim.compare_numeric_to_analytic_gradient(pulse + 1,
                                                         delta_eps=delta)
            )

        for k in range(3):
            # assert all simulations yield the same result
            rel_infidelity_variation = \
                .5 * (infidelities[0] - infidelities[k]) \
                / (infidelities[0] + infidelities[k])
            self.assertTrue(rel_infidelity_variation < 1e-1)

        # Assert correct Gradients

        # Lindblad is about as accurate as the finite differences
        self.assertTrue(gradient_accuracy[0][0] < 1e-7)

        # Monte Carlo has stability issues to converge to 0:
        self.assertTrue(gradient_accuracy[1][0] < 1e-2)

        # Filter functions very small:
        self.assertTrue(gradient_accuracy[2][0] < 1e-10)

