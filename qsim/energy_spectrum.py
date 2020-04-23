"""
This file serves to plot energy spectra of Hamiltonians.


"""

import matplotlib.pyplot as plt
import numpy as np

from typing import List

from qsim.matrix import OperatorMatrix


def vector_color_map(vectors):
    """
    Maps an eigenvector to a color, encoding the contributions.
    """
    assert len(vectors.shape) == 2
    n = vectors.shape[0]
    basis = np.asarray([
        [1, 0, 0, 1, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 1]
    ])
    basis = basis[:, :vectors.shape[1]]
    values = np.einsum('ni,ji -> nj', np.abs(vectors), basis)
    for i in range(n):
        values[i, :] /= np.linalg.norm(values[i, :])
    return values


def plot_energy_spectrum(hamiltonian: List[OperatorMatrix], x_val, x_label):
    """
    Calculates and plots the energy spectra of hamilton operators.

    Parameters
    ----------
    hamiltonian: list of OperatorMatrix
        The Hamiltonians which shall provide the energy spectra. They need to
        be hermitian.

    x_val: array of float, shape (n, )
        The x_vales by which the eigenvalues are plotted.

    x_label: str
        Label of the x-axis.

    Returns
    -------

    """
    d = hamiltonian[0].shape[0]
    eigenvalues = np.empty((len(hamiltonian), d))
    eigenvectors = np.empty((len(hamiltonian), d, d))
    for i, h in enumerate(hamiltonian):
        eig_val, eig_vec = h.spectral_decomposition(hermitian=True)
        eigenvalues[i, :] = eig_val
        eigenvectors[i, :, :] = eig_vec
    # plt.plot(eigenvalues)

    colors = [[tuple(vec) for vec in eigenvectors[:, :, j]] for j in range(d)]
    plt.figure()
    for i in range(d):
        plt.scatter(x=x_val, y=eigenvalues[:, i],
                    c=vector_color_map(eigenvectors[:, :, i]))
    return eigenvalues, eigenvectors
