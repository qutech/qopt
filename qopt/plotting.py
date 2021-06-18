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
""" Plotting functions.

The function `plot_bloch_vector_evolution` can be used to plot the evolution
under a series of propagators on the bloch sphere. It uses QuTiP and is only
available if QuTiP is installed in the environment. (See installation
instructions on https://github.com/qutech/qopt)

Functions
---------
:func:`plot_bloch_vector_evolution`
    Plots the evolution of the forward propagators of the initial state on the
    bloch sphere.

Notes
-----
The implementation was adapted from the filter_functions package.

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from unittest import mock
from warnings import warn
from typing import Sequence

from qopt.matrix import OperatorMatrix

__all__ = []

try:
    import qutip as qt
    __all__.append('plot_bloch_vector_evolution')
except ImportError:
    warn('Qutip not installed. plot_bloch_vector_evolution() is not available')
    qt = mock.Mock()


def plot_bloch_vector_evolution(
        forward_propagators: Sequence[OperatorMatrix],
        initial_state: OperatorMatrix,
        return_bloch: bool = False,
        **bloch_kwargs):
    """
    Plots the evolution of the forward propagators of the initial state on the
    bloch sphere.

    Parameters
    ----------
    forward_propagators: list of DenseOperators
        The forward propagators whose evolution shall be plotted on the Bloch
        sphere.

    initial_state: DenseOperator
        The initial state aka. beginning point of the plotting.

    return_bloch: bool, optional
        If True, the Bloch sphere is returned as object.

    bloch_kwargs: dict, optional
        Plotting parameters for the Bloch sphere.

    Returns
    -------
    bloch_sphere:
        Only returned if return_bloch is set to true.

    """
    try:
        import qutip as qt
    except ImportError as err:
        raise RuntimeError(
            'Requirements not fulfilled. Please install Qutip') from err

    if not forward_propagators[0].shape[0] == 2:
        raise ValueError('Plotting Bloch sphere evolution only implemented '
                         'for one-qubit case!')

    figsize = bloch_kwargs.pop('figsize', [5, 5])
    view = bloch_kwargs.pop('view', [-60, 30])
    fig = plt.figure(figsize=figsize)
    axes = mplot3d.Axes3D(fig, azim=view[0], elev=view[1])
    bloch_kwargs.setdefault('view', [-150, 30])
    b = qt.Bloch(fig=fig, axes=axes, **bloch_kwargs)

    # https://github.com/qutip/qutip/issues/1385
    if hasattr(b.axes, 'set_box_aspect'):
        b.axes.set_box_aspect([1, 1, 1])

    b.xlabel = [r'$|+\rangle$', '']
    b.ylabel = [r'$|+_i\rangle$', '']

    states = [
        qt.Qobj((prop * initial_state).data) for prop in forward_propagators
    ]
    a = np.empty((3, len(states)))
    x, y, z = qt.sigmax(), qt.sigmay(), qt.sigmaz()
    for i, state in enumerate(states):
        a[:, i] = [qt.expect(x, state),
                   qt.expect(y, state),
                   qt.expect(z, state)]
    b.add_points(a.real, meth='l')
    b.make_sphere()

    if return_bloch:
        return b
