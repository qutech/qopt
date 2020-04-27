"""This class is designed to express a functional relationship between the
optimization parameters, which can be directly controlled and the control
amplitudes, which appear as factors in the Hamiltonian.

Classes
-------
AmplitudeFunction
    Abstract base class of the amplitude function.

IdentityAmpFunc
    The trivial amplitude function acting as identity operation.

UnaryAnalyticAmpFunc
    An amplitude function which can be given by a unary function.

"""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class AmplitudeFunction(ABC):
    """Abstract Base class of the amplitude function. """
    @abstractmethod
    def __call__(self, x):
        """ Calculates the control amplitudes u from the optimization parameters
        x.

        Parameters
        ----------
        x : array
            Optimization parameters of shape (num_x, num_amp), where num_x is
            the number of time slices and num_amp the number of control
            amplitudes.

        Returns
        -------
        u : np.array
            Control amplitudes of shape (num_x, num_ctrl), where num_x is
            the number of time slices and num_ctrl the number of control
            operators.

        """
        return None

    @abstractmethod
    def gradient_u2x(self, deriv_by_ctrl_amps, x):
        """ Calculates the derivatives of some function f by the optimization
        parameters x i.e. df/dx by the chain rule.

        The calculation is df/dx = df/du * du/dx.

        Parameters
        ----------
        deriv_by_ctrl_amps : np.array, shape (num_u, num_f, num_ctrl)
            The gradients of num_f functions by num_ctrl different pulses at
            num_u different time steps, i.e. the derivatives df/du

        x : np.array
            Optimization parameters of shape (num_x, num_amp), where num_x is
            the number of time slices and num_ctrl the number of control
            parameters.

        Returns
        -------
        deriv_by_opt_par : np.array, shape: (num_x, num_f, num_amp)
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

    def gradient_u2x(self, deriv_by_ctrl_amps: np.ndarray,
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
        This scalar function describes the derivative of the control amplitudes.
        The function is vectorized internally.

    """
    def __init__(self,
                 value_function: Callable[[float, ], float],
                 derivative_function: Callable[[float, ], float]):
        self.value_function = np.vectorize(value_function)
        self.derivative_function = np.vectorize(derivative_function)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """See base class. """
        return self.value_function(x)

    def gradient_u2x(self, deriv_by_ctrl_amps: np.ndarray, x):
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
        This scalar function expresses the functional dependency of the control
        amplitudes on the optimization parameters. The function receives an
        array of the shape (num_x, num_amp) and must return an array of the
        shape (num_x, num_ctrl). Where num_x is the number of time slices,
        num_amp the number of amplitudes and num_ctrl the number of control
        terms in the Hamiltonian.

    derivative_function : Callable array to array
        This scalar function describes the derivative of the control amplitudes.
        The function receives the transferred optimisation parameters as array
        of shape (num_x, num_ctrl) and returns the derivatives of the control
        amplitudes by the transferred optimization parameters as array of shape
        (num_x, num_amp, num_ctrl).

    """
    def __init__(self,
                 value_function: Callable[[np.ndarray, ], np.ndarray],
                 derivative_function: Callable[[np.ndarray, ], np.ndarray]):
        self.value_function = value_function
        self.derivative_function = derivative_function

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """See base class. """
        return self.value_function(x)

    def gradient_u2x(self, deriv_by_ctrl_amps: np.ndarray,
                     x: np.ndarray) -> np.ndarray:
        """See base class. """
        du_by_dx = self.derivative_function(x)
        # du_by_dx: shape (time, amp, ctrl)
        # deriv_by_ctrl_amps: shape (time, func, ctrl)
        # return: shape (time, func, amp)

        return np.einsum('imj,ikj->ikm', du_by_dx, deriv_by_ctrl_amps)
