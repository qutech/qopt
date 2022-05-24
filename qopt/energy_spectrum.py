# -*- coding: utf-8 -*-
# =============================================================================
#     qopt
#     Copyright (C) 2020 Julian Teske, Forschungszentrum Juelich
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program. If not, see <http://www.gnu.org/licenses/>.
#
#     Contact email: j.teske@fz-juelich.de
# =============================================================================
"""This file serves to plot energy spectra of a Hamiltonian.

The convenience functions implemented in this module can be used to plot the
eigenvalues and eigenvectors of the Hamiltonian as function of a parameter.
This is especially useful to analyse the theoretical properties of a system.

Functions
---------
:func:`vector_color_map`
    Maps eigenvectors to a coloring.

:func:`plot_energy_spectrum`
    plot the energy spectrum of an Hamiltonian.

"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from typing import List

from qopt.matrix import OperatorMatrix


def vector_color_map(vectors: np.array):
    """
    Maps eigenvectors to a coloring, encoding the contributions.

    Parameters
    ----------
    vectors: array
        Array of eigenvectors. The eigenvectors are given as columns. There
        may be no more than 7.

    Returns
    -------
    color_values: array
        The coloring is given as array. Each column signifies one tuple of
        RGB color values.

    """
    assert len(vectors.shape) == 2

    points_on_cmap = np.linspace(0, 1, vectors.shape[-1])
    basis_colors = mpl.cm.get_cmap('brg')(points_on_cmap)
    basis_colors = np.array(basis_colors)[:, :-1]
    
    values = np.einsum('ij, ni -> nj', basis_colors, np.abs(vectors))

    # in this basis, the normalization to the rgb range is not preserved, thus it needs to be renormalized. 
    values /= np.max(values, axis=-1)[..., None]

    return values


def plot_energy_spectrum(hamiltonian: List[OperatorMatrix],
                         x_val: np.array,
                         x_label: str,
                         ax=None,
                         **scatter_kwargs):
    """
    Calculates and plots the energy spectra of hamilton operators.

    The colors demonstrate the contribution of individual base vectors.

    Parameters
    ----------
    hamiltonian: list of OperatorMatrix
        The Hamiltonians which shall provide the energy spectra. They need to
        be hermitian.

    x_val: array of float, shape (n, )
        The x_vales by which the eigenvalues are plotted.

    x_label: str
        Label of the x-axis.

    ax: matplotlib pyplot axes
        Instance of axes to plot the data in. Defaults to None.

    """
    d = hamiltonian[0].shape[0]
    eigenvalues = np.empty((len(hamiltonian), d))
    eigenvectors = np.empty((len(hamiltonian), d, d))
    for i, h in enumerate(hamiltonian):
        eig_val, eig_vec = h.spectral_decomposition(hermitian=True)
        eigenvalues[i, :] = eig_val
        eigenvectors[i, :, :] = np.abs(eig_vec)

    if ax is None:
        _, ax = plt.subplots()
    for i in range(d):
        ax.scatter(x=x_val, y=eigenvalues[:, i],
                   c=vector_color_map(eigenvectors[:, :, i]),
                   **scatter_kwargs)
    ax.set_xlabel(x_label)
    return ax
