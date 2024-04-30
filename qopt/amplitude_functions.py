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
r"""This class is designed to express a functional relationship between the
optimization parameters, which can be directly controlled and the control
amplitudes, which appear as factors in the Hamiltonian.

If the Hamiltonian is given as sum of a drift Hamiltonian and a control
Hamiltonian described by operators multiplied with time dependent control
amplitudes

.. math::

    H = H_{drift} + \sum_k u_k(t) H_k,

then this class describes the control amplitudes as function of optimization
parameters:

.. math::

    u_k(t) = u_k(x(t))

The `AmplitudeFunction` class is used as attribute of the `Solver` class.

Classes
-------
:class:`AmplitudeFunction`
    Abstract base class of the amplitude function.
:class:`IdentityAmpFunc`
    The transferred optimization parameters are the control amplitudes.
:class:`UnaryAnalyticAmpFunc`
    An amplitude function which can be given by a unary function.
:class:`CustomAmpFunc`
    Applies functions handles specified by the user at the initialization.

Notes
-----
The implementation was inspired by the optimal control package of QuTiP [1]_
(Quantum Toolbox in Python)

References
----------
.. [1] J. R. Johansson, P. D. Nation, and F. Nori: "QuTiP 2: A Python framework
    for the dynamics of open quantum systems.", Comp. Phys. Comm. 184, 1234
    (2013) [DOI: 10.1016/j.cpc.2012.11.019].

"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

from typing import Union

class AmplitudeFunction(ABC):
    """Abstract Base class of the amplitude function. """
    @abstractmethod
    def __call__(self, x):
        """ Calculates the control amplitudes u from the optimization
        parameters x.

        Parameters
        ----------
        x : np.array
            Optimization parameters of shape (num_t, num_par), where num_t is
            the number of time slices and num_par the number of different
            optimization parameters.

        Returns
        -------
        u : np.array
            Control amplitudes of shape (num_t, num_ctrl), where num_x is
            the number of time slices and num_ctrl the number of control
            operators in the Hamiltonian.

        """
        return None

    @abstractmethod
    def derivative_by_chain_rule(self, deriv_by_ctrl_amps, x):
        """ Calculates the derivatives of some function f by the optimization
        parameters x, when given the optimization parameters x and the
        derivative by the control amplitudes. The calculation is performed
        using the chain rule: df/dx = df/du * du/dx.

        Parameters
        ----------
        deriv_by_ctrl_amps : np.array, shape (num_t, num_f, num_ctrl)
            The gradients of num_f functions by num_ctrl different pulses at
            num_t different time steps, i.e. the derivatives df/du.

        x : np.array
            Optimization parameters of shape (num_t, num_par), where num_t is
            the number of time slices and num_par the number of different
            optimization parameters.

        Returns
        -------
        deriv_by_opt_par : np.array, shape: (num_t, num_f, num_par)
            The derivatives by the optimization parameters.

        """
        return None


class IdentityAmpFunc(AmplitudeFunction):
    """The control amplitudes are identical with the optimization parameters.

    """
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """See base class. """
        return x

    def derivative_by_chain_rule(self, deriv_by_ctrl_amps: np.ndarray,
                                 x: np.ndarray) -> np.ndarray:
        """See base class. """
        return deriv_by_ctrl_amps


class UnaryAnalyticAmpFunc(AmplitudeFunction):
    """A unary analytic amplitude function which is applied to each amplitude
    value. This class can be used for every application case where all
    transferred parameters are mapped one-to-one to the control amplitudes
    by a single unary function.

    Parameters
    ----------
    value_function : Callable float to float
        This scalar function expresses the functional dependency of the control
        amplitudes on the optimization parameters. The function is vectorized
        internally.

    derivative_function : Callable float to float
        This scalar function describes the derivative of the control
        amplitudes. The function is vectorized internally.

    """
    def __init__(self,
                 value_function: Callable[[float, ], float],
                 derivative_function: Callable[[float, ], float]):
        self.value_function = np.vectorize(value_function)
        self.derivative_function = np.vectorize(derivative_function)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """See base class. """
        return self.value_function(x)

    def derivative_by_chain_rule(self, deriv_by_ctrl_amps: np.ndarray, x):
        """See base class. """
        du_by_dx = self.derivative_function(x)
        # du_by_dx shape: (n_time, n_ctrl)
        # deriv_by_ctrl_amps shape: (n_time, n_func, n_ctrl)
        # deriv_by_opt_par shape: (n_time, n_func, n_ctrl
        # since the function is unary we have n_ctrl = n_amps
        return np.einsum('ij,ikj->ikj', du_by_dx, deriv_by_ctrl_amps)


class CustomAmpFunc(AmplitudeFunction):
    """A general amplitude function which is applied to the amplitude
    values.

    Parameters
    ----------
    value_function : Callable array to array
        This function expresses the functional dependency of the control
        amplitudes on the optimization parameters. The function receives the
        optimization parameters x as array of the shape (num_t, num_par) and
        must return the control amplitudes u as array of the shape
        (num_t, num_ctrl). Where num_t is the number of time slices,
        num_par the number of optimization parameters and num_ctrl the number
        of control operators in the Hamiltonian.

    derivative_function : Callable array to array
        This function describes the derivative of the control amplitudes by the
        optimization parameters.
        The function receives the optimisation parameters x as array
        of shape (num_t, num_par) and must return the derivatives of the
        control amplitudes by the optimization parameters as array of shape
        (num_t, num_par, num_ctrl).

    """
    def __init__(self,
                 value_function: Callable[[np.ndarray, ], np.ndarray],
                 derivative_function: Callable[[np.ndarray, ], np.ndarray]):
        self.value_function = value_function
        self.derivative_function = derivative_function

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """See base class. """
        return self.value_function(x)

    def derivative_by_chain_rule(self, deriv_by_ctrl_amps: np.ndarray,
                                 x: np.ndarray) -> np.ndarray:
        """See base class. """
        du_by_dx = self.derivative_function(x)
        # du_by_dx: shape (time, par, ctrl)
        # deriv_by_ctrl_amps: shape (time, func, ctrl)
        # return: shape (time, func, par)

        return np.einsum('imj,ikj->ikm', du_by_dx, deriv_by_ctrl_amps)


###############################################################################

try:
    import jax.numpy as jnp
    from jax import jit,vmap,jacfwd
    _HAS_JAX = True
except ImportError:
    from unittest import mock
    jit, vmap, jacfwd = mock.Mock(), mock.Mock(), mock.Mock()
    jnp = mock.Mock()
    _HAS_JAX = False


class IdentityAmpFuncJAX(AmplitudeFunction):
    """See docstring of class without JAX.
    Designed to return jax-numpy-arrays.
    """

    def __init__(self):
        if not _HAS_JAX:
            raise ImportError("JAX not available")

    def __call__(self, x: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """See base class. """
        #TODO: is asarray the best way, or rather array with default copying?
        return jnp.asarray(x)

    def derivative_by_chain_rule(
            self,
            deriv_by_ctrl_amps: Union[np.ndarray,jnp.ndarray],
            x: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """See base class. """
        return jnp.asarray(deriv_by_ctrl_amps)


class UnaryAnalyticAmpFuncJAX(AmplitudeFunction):
    """See docstring of class without JAX.
    Designed to return jax-numpy-arrays.
    Functions need to be compatible with jit.
    (Includes that functions need to be pure
    (i.e. output solely depends on input)).
    """
    #TODO: jax autodiff

    def __init__(self,
                 value_function: Callable[[float, ], float],
                 derivative_function: [Callable[[float, ], float]]):
        if not _HAS_JAX:
            raise ImportError("JAX not available")
        self.value_function = jit(jnp.vectorize(value_function))
        self.derivative_function = jit(jnp.vectorize(derivative_function))

    def __call__(self, x: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """See base class. """
        return jnp.asarray(self.value_function(x))

    def derivative_by_chain_rule(
            self,
            deriv_by_ctrl_amps: Union[np.ndarray, jnp.ndarray], x):
        """See base class. """
        du_by_dx = self.derivative_function(x)
        # du_by_dx shape: (n_time, n_ctrl)
        # deriv_by_ctrl_amps shape: (n_time, n_func, n_ctrl)
        # deriv_by_opt_par shape: (n_time, n_func, n_ctrl
        # since the function is unary we have n_ctrl = n_amps
        return jnp.einsum('ij,ikj->ikj', du_by_dx, deriv_by_ctrl_amps)


class CustomAmpFuncJAX(AmplitudeFunction):
    """See docstring of class without JAX.
    Designed to return jax-numpy-arrays.
    Functions need to be compatible with jit.
    (Includes that functions need to be pure
    (i.e. output solely depends on input)).
    If derivative_function=None, autodiff is used.
    t_to_vectorize: if value_function/derivative_function not yet
    vectorized for num_t
    """

    def __init__(
            self,
            value_function: Callable[[Union[np.ndarray, jnp.ndarray],],
                                      Union[np.ndarray, jnp.ndarray]],
            derivative_function: Callable[[Union[np.ndarray, jnp.ndarray],],
                                           Union[np.ndarray, jnp.ndarray]],
            t_to_vectorize: bool = False
            ):
        if not _HAS_JAX:
            raise ImportError("JAX not available")
        if t_to_vectorize == True:
            self.value_function = jit(vmap(value_function),in_axes=(0,))
        else:
            self.value_function = jit(value_function)
        if derivative_function is not None:
            if t_to_vectorize == True:
                self.derivative_function = jit(vmap(derivative_function),in_axes=(0,))
            else:
                self.derivative_function = jit(derivative_function)
        else:
            #TODO: is jacfwd or jacrev better here?
            if t_to_vectorize == True:
                def der_wrapper(x):
                    return jnp.swapaxes(vmap(jacfwd(lambda x: value_function(x)),in_axes=(0,))(x),1,2)
            else:
                def der_wrapper(x):
                    return jnp.swapaxes(vmap(jacfwd(lambda x: value_function(jnp.expand_dims(x,axis=0))[0,:]),in_axes=(0,))(x),1,2)
            self.derivative_function = jit(der_wrapper)

    def __call__(self, x: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        #TODO: potentially cases where jnp array causes errors
        #when passed to custom func only supporting np?
        """See base class. """
        return jnp.asarray(self.value_function(x))

    def derivative_by_chain_rule(
            self,
            deriv_by_ctrl_amps: Union[np.ndarray, jnp.ndarray],
            x: Union[np.ndarray, jnp.ndarray]) -> jnp.ndarray:
        """See base class. """
        du_by_dx = self.derivative_function(x)
        # du_by_dx: shape (time, par, ctrl)
        # deriv_by_ctrl_amps: shape (time, func, ctrl)
        # return: shape (time, func, par)

        return jnp.einsum('imj,ikj->ikm', du_by_dx, deriv_by_ctrl_amps)
