import numpy as np

from qsim.matrix import OperatorDense
from qsim.energy_spectrum import plot_energy_spectrum

pauli_z = OperatorDense(np.asarray([[1, 0], [0, -1]]))
pauli_x = OperatorDense(np.asarray([[0, 1], [1, 0]]))

tau_z = pauli_z.kron(pauli_z.identity_like())
tau_x = pauli_x.kron(pauli_x.identity_like())
sigma_z = (pauli_z.identity_like()).kron(pauli_z)
sigma_x = (pauli_x.identity_like()).kron(pauli_x)
sigma_x_tau_x = pauli_x.kron(pauli_x)


def example_hamiltonian_flopping(detuning):
    hamiltonian = []
    t_c = 20
    e_z = 24
    g_mu_b_b_x = 15
    g_mu_b_b_z = 4
    for eps in detuning:
        omega = np.sqrt(eps ** 2 + (2 * t_c) ** 2)
        theta = np.arctan(eps / (2 * t_c))
        hamiltonian.append(
            .5 * omega * tau_z + (e_z / 2) * sigma_z
            - (g_mu_b_b_x / 2) * sigma_x * (
                np.cos(theta) * tau_x - np.sin(theta) * tau_z
            )
            - (g_mu_b_b_z / 2) * sigma_z * (
                    np.cos(theta) * tau_x - np.sin(theta) * tau_z)
        )
    return hamiltonian


def example_hamiltonian_in_output_theo(detuning):
    hamiltonian = []
    t_c = 10.2
    b_x = 10
    b_z = 24
    for eps in detuning:
        hamiltonian.append(
            .5 * (
                eps * tau_z
                + 2 * t_c * tau_x
                + b_z * sigma_z
                + b_x * sigma_x * tau_z
            )
        )
    return hamiltonian


def example_simple_hubbard_model(detuning):
    hamiltonian = []
    t_c = 25
    for eps in detuning:
        hamiltonian.append(
            .5 * (
                eps * pauli_z + t_c * pauli_x
            )
        )
    return hamiltonian


n_eps_vals = 200
eps_vals = np.linspace(start=-100, stop=100, num=n_eps_vals)
ex_ham = example_hamiltonian_flopping(eps_vals)

eps_vals_in_out_theo = np.linspace(start=-50, stop=50, num=n_eps_vals)
ex_ham_in_out_theo = example_hamiltonian_in_output_theo(eps_vals_in_out_theo)

eig_vals, eig_vecs = plot_energy_spectrum(ex_ham, eps_vals, 'detuning')

ex_ham_hub = example_simple_hubbard_model(eps_vals)
eig_vals, eig_vecs = plot_energy_spectrum(ex_ham_hub, eps_vals, 'detuning')
