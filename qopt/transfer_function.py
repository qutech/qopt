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
"""Models the response function of control electronics and pulse smoothing.

Due to the imperfection of the control electronics, the generated control pulse
which is received by the physical qubit is not exactly the pulse which has been
implemented at the control level.

If for example a voltage is changed from one
value to another at a single point in time at the control level, then the
control electronics might need some time to physically reach the new voltage.

Another example would be an amplifier, which has a non-linearity in the
amplification of a control pulse.

There are two kinds of transfer functions. Those who are based on transfer
matrices and those who are not. In principle every transfer function can be
expressed as a matrix multiplication because it is linear by definition,
but the transfer function will have n_time ** 2 entries which might be huge
number if you consider a control pulse with a large number n_time of time
steps. Therefore it is advantageous to use the matrix based transfer functions
for small pulses with a lot of correlations.

The matrix based transfer function have their own classes for concatenation
and parallel application.

See Also
--------

Optimal control methods for rapidly time-varying Hamiltonians, 2011
Motzoi, F. and Gambetta, J. M. and Merkel, S. T. and Wilhelm, F. K.
PhysRevA.84.022307, https://link.aps.org/doi/10.1103/PhysRevA.84.022307

Classes
-------
:class:`TransferFunction`
    Abstract base class.

:class:`IdentityTF`
    Optimization variables are the amplitudes of the control fields.

:class:`OversamplingTF`
    Oversamples the pulse and adds boundary conditions.

:class:`GaussianConvolution`
    Applies a Gaussian convolution.

:class:`ConcatenateTF`
    Concatenation of two transfer functions.

:class:`ParallelTF`
    Using to transfer functions for two sets of parameters in paralell.

:class:`MatrixTF`
    Abstract base class for transfer functions as matrices.

:class:`OversamplingMTF`
    Matrix version of OversamplingTF.

:class:`CustomMTF`
    Transfer function which receives an explicitly constructed constant
    transfer matrix.

:class:`ConcatenateMTF`
    Matrix version of ConcatenateTF.

:class:`ParallelMTF`
    Matrix version of ParallelTF.

:class:`ExponentialMTF`
    The amplitudes are smoothed by exponential saturation functions.

Functions
---------
:func:`exp_saturation`
    Exponential saturation function.

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

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.ndimage
import copy
from typing import Tuple, Optional, Union, Sequence
from abc import ABC, abstractmethod

from qopt.util import deprecated, needs_refactoring


class TransferFunction(ABC):
    """
    A class for representing transfer functions, between optimization
    variables of the optimization algorithm and the amplitudes of the control
    fields.

    The intended workflow is to initialise the transfer function object first
    and subsequently set the x_times, which is the time scale of the
    optimization variables. Then the transfer function is called to calculate
    control amplitudes and gradients.

    Examples
    --------
    Example work flow with the abstract base class:

    >>> x_times = np.ones(5)
    >>> optimization_variables = np.random.rand(5)
    >>> gradient_fidelity_by_control_amplitudes = np.random.rand(shape=(30,5))
    >>> transfer_function = TransferFunction(oversampling=5,
    >>>                                      bound_type=('x', 1))
    >>> transfer_function.set_times(x_times)
    >>> control_amplitudes = transfer_function(optimization_variables)
    >>> gradient_fidelity_by_optimization_variables = \
    >>>     transfer_function.gradient_chain_rule(
    >>>         gradient_fidelity_by_control_amplitudes)

    Parameters
    ----------
    num_ctrls: int
        Number of controlled amplitudes.

    oversampling: int
        Each time step of the optimization variables is sliced into a number
        of time steps of the control amplitudes.

    bound_type: (str, int)
        The pulse can be padded with zeros (before adding the offset) to avoid
        bleedthrough i.e. that the pulses overlap slightly and thereby
        influence each other.

        The string states, whether you want to pad before or after the
        oversampling or just to the end of the pulse.

        The integer specifies the amount of padding elements (also depending on
        the code of course).

        If not all time steps have the same length:
        Let dt denote the first (or respectively last) time duration when you
        are padding to the beginning (end) of the sequence.

        string options:
            "n": n extra slice of dt/overSampleRate
            "x": n extra slice of dt (default with n=1)
            "right_n": n extra slice of dt/overSampleRage on the right side

    offset: float
        Constant offset which is added to the optimization parameters.


    Attributes
    ----------
    num_ctrls: int
        Number of controlled amplitudes.

    oversampling: int
        Each time step of the optimization variables is sliced into a number
        of time steps of the control amplitudes.

    bound_type: (str, int) or None
        The pulse can be padded with zeros (before adding the offset) to avoid
        bleedthrough i.e. that the pulses overlap slightly and thereby
        influence each other.

        The string states, whether you want to pad before or after the
        oversampling or just to the end of the pulse.

        The integer specifies the amount of padding elements (also depending on
        the code of course).

        If not all time steps have the same length:
        Let dt denote the first (or respectively last) time duration when you
        are padding to the beginning (end) of the sequence.

        string options:
            "n": n extra slice of dt/overSampleRate
            "x": n extra slice of dt (default with n=1)
            "right_n": n extra slice of dt/overSampleRage on the right side

    offset: float
        Constant offset which is added to the optimization parameters.

    num_x: int
        Number of time slices of the transferred optimization variables.

    x_times: array, shape (num_u)
        Time values for the transferred optimization parameters. These
        describe the length of the time slices.

    _num_y: int
        Number of time slices of the raw optimization variables.

    _y_times: array, shape (num_x)
        Time values for the raw control variables. These  describe the length
        of the time slices.

    _absolute_y_times : array_like, shape (num_x + 1)
        Absolute times of the raw optimization variables. The values describe
        the point in time where a time slice ends and the next one begins.

    Methods
    -------
    __call__(y):
        Application of the transfer function.

    transfer_matrix: property, returns array, shape (num_x, num_y, num_par)
        Returns the transfer matrix.

    num_padding_elements: property, returns list
        Two elements list with the number of elements padded to the beginning
        and end, as specified by the bound type.

    set_times(times):
        Set the times of the optimization variables and calculates the times
        of the optimization variables.

    set_absolute_times(absolute_y_times):
        Set the absolute times (time points of beginning and ending a time
        step) of the optimization variables.

    plot_pulse(y):
        For the raw optimisation variables (y), plot the resulting pulse.

    `Todo`
        * bound type seems to be buggy. test with exp_transfer
        * parse bound_type to raise exception only in one function.
        * add exception to the docstring

    """

    def __init__(self,
                 num_ctrls: int = 1,
                 bound_type: Optional[Tuple[str, int]] = None,
                 oversampling: int = 1,
                 offset: Optional[float] = None
                 ):

        self.num_ctrls = num_ctrls
        self.bound_type = bound_type
        self.oversampling = oversampling
        self.offset = offset

        self._transfer_matrix = None

        # num_y is set, by setting the time
        self._num_y = 0
        self._y_times = None
        self._absolute_y_times = None
        # num_x is calculated when setting the time
        self.num_x = 0
        self.x_times = None

    @abstractmethod
    def __call__(self, y: np.array) -> np.array:
        """Calculate the transferred optimization parameters (x).

        Evaluates the transfer function at the raw optimization parameters (y)
        to calculate the transferred optimization parameters (x).

        Parameters
        ----------
        y: np.array, shape (num_y, num_par)
            Raw optimization variables; num_y is the number of time slices of
            the raw optimization parameters and num_par is the number of
            distinct raw optimization parameters.

        Returns
        -------
        u: np.array, shape (num_x, num_par)
            Control parameters; num_u is the number of times slices for the
            transferred optimization parameters.

        """
        pass

    @property
    def num_padding_elements(self) -> (int, int):
        """
        Convenience function. Returns the number of elements padded to the
        beginning and the end of the control amplitude times.

        Returns
        -------
        num_padding_elements: (int, int)
            (elements padded to the beginning, elements padded to the end)

        """
        if self.bound_type is None:
            return 0, 0
        elif self.bound_type[0] == 'n':
            return self.bound_type[1], self.bound_type[1]
        elif self.bound_type[0] == 'x':
            return self.bound_type[1] * self.oversampling, \
                   self.bound_type[1] * self.oversampling
        elif self.bound_type[0] == 'right_n':
            return 0, self.bound_type[1]
        else:
            raise ValueError('Unknown bound type ' + str(self.bound_type[0]))

    @abstractmethod
    def gradient_chain_rule(
            self, deriv_by_transferred_par: np.array) -> np.array:
        """
        Obtain the derivatives of a quantity a i.e. da/dy by the optimization
        variables from the derivatives by the amplitude of the control fields.

        The chain rule applies: df/dy = df/dx * dx/dy.

        Parameters
        ----------
        deriv_by_transferred_par: np.array, shape (num_x, num_f, num_par)
            The gradients of num_f functions by num_par optimization parameters
            at num_x different time steps.

        Returns
        -------
        deriv_by_opt_par: np.array, shape: (num_y, num_f, num_par)
            The derivatives by the optimization parameters at num_y time steps.

        """
        pass

    def set_times(self, y_times: np.array) -> None:
        """
        Generate the time_slot duration array 'transferred_time'
        (here: x_times).

        The time slices depend on the oversampling of the control variables
        and the boundary conditions. The times are for the intended use cases
        only set once.

        Parameters
        ----------
        y_times: Union[np.ndarray, list], shape (num_y)
            The time steps / durations of constant optimization variables.
            num_y is the number of time steps for the raw optimization
            variables.

        """
        if isinstance(y_times, list):
            y_times = np.array(y_times)
        if not isinstance(y_times, np.ndarray):
            raise Exception("times must be a list or np.array")

        y_times = np.atleast_1d(np.squeeze(y_times))

        if len(y_times.shape) > 1:
            raise ValueError('The x_times should not have more than one '
                             'dimension!')

        self._num_y = y_times.size
        self._y_times = y_times

        if self.bound_type is None:
            self.num_x = self.oversampling * self._num_y
            self.x_times = np.repeat(
                self._y_times, self.oversampling) / self.oversampling

        elif self.bound_type[0] == 'n':
            self.num_x = self.oversampling * self._num_y + 2 \
                         * self.bound_type[1]
            self.x_times = np.concatenate((
                self._y_times[0] / self.oversampling
                * np.ones(self.bound_type[1]),
                np.repeat(
                    self._y_times / self.oversampling, self.oversampling),
                self._y_times[-1] / self.oversampling
                * np.ones(self.bound_type[1])))

        elif self.bound_type[0] == 'x':
            self.num_x = self.oversampling * (self._num_y
                                              + 2 * self.bound_type[1])
            self.x_times = np.concatenate((
                self._y_times[0] / self.oversampling
                * np.ones(self.bound_type[1] * self.oversampling),
                np.repeat(self._y_times / self.oversampling,
                          self.oversampling),
                self._y_times[-1] / self.oversampling
                * np.ones(self.bound_type[1] * self.oversampling)))

        elif self.bound_type[0] == 'right_n':
            self.num_x = self.oversampling * self._num_y + self.bound_type[1]
            self.x_times = np.concatenate((
                np.repeat(self._y_times / self.oversampling,
                          self.oversampling),
                self._y_times[-1] / self.oversampling
                * np.ones(self.bound_type[1])))

        else:
            raise ValueError('The boundary type ' + str(self.bound_type[0])
                             + ' is not implemented!')

    def set_absolute_times(self, absolute_y_times: np.ndarray) -> None:
        """
        Generate the time_slot duration array 'transferred_time'
        (here: x_times)

        This time slices depend on the oversampling of the control variables
        and the boundary conditions. The differences of the absolute times
        give the time steps x_times.

        Parameters
        ----------
        absolute_y_times: Union[np.ndarray, list]
            Absolute times of the start / end of each time segment for the raw
            optimization parameters.

        """
        if isinstance(absolute_y_times, list):
            absolute_y_times = np.array(absolute_y_times)
        if not isinstance(absolute_y_times, np.ndarray):
            raise Exception("times must be a list or np.array")
        if not np.all(np.diff(absolute_y_times) >= 0):
            raise Exception("times must be sorted")

        self._absolute_y_times = absolute_y_times
        self.set_times(np.diff(absolute_y_times))

    def plot_pulse(self, y: np.array) -> None:
        """
        Plot the control amplitudes corresponding to the given optimisation
        variables.

        Parameters
        ----------
        y: array, shape (num_y, num_par)
            Raw optimization parameters.

        """

        x = self(y)
        n_padding_start, n_padding_end = self.num_padding_elements
        for y_per_control, x_per_control in zip(y.T, x.T):
            plt.figure()
            plt.bar(np.cumsum(self.x_times) - .5 * self.x_times[0],
                    x_per_control, self.x_times[0])
            plt.bar(np.cumsum(self._y_times) - .5 * self._y_times[0]
                    + np.cumsum(self.x_times)[n_padding_start]
                    - self.x_times[n_padding_start],
                    y_per_control, self._y_times[0],
                    fill=False)
        plt.show()

    def _check_dimensions_datatype(self, y: np.array) -> None:
        """
        This function verifies that the transfer function can be applied to
        the pulse y. For this purpose, the shape and the data type of y must
        fit the transfer function.

        Parameters
        ----------
        y: np.array, shape (num_y, num_par)
            Raw optimization variables; num_y is the number of time slices of
            the raw optimization parameters and num_par is the number of
            distinct raw optimization parameters.

        Raises
        ------
        IndexError: is raised when the pulse has the wrong dimensions.

        TypeError: is raised when the pulse has the wrong data type.

        RuntimeError: is raised when the transfer function is not yet properly
        initialised.

        """

        # check that the transfer function is correctly initialised.
        if self._y_times is None:
            raise RuntimeError(
                'The times are not set to this instance of transfer functions.'
                'Please use the function TransferFunction.set_times before '
                'you continue.'
            )

        # check for the correct data type:
        if not y.dtype in [np.float64, np.float32]:
            raise TypeError(
                'The transfer function assumes an input pulse as numpy array'
                'of the dtype np.float32 or np.float64 but the given input'
                'has type: ' + str(y.dtype)
            )

        # check for the correct data dimensions
        shape = y.shape

        if not len(shape) == 2:
            raise IndexError(
                'The input data pulse must be a numpy array with 2 dimensions,'
                'but your pulse has ' + str(len(shape)) + ' dimensions.'
            )

        if not shape[0] == self._num_y:
            raise IndexError(
                'The pulse data must have exactly as many entries as there are'
                'time steps. You have ' + str(self._num_y) +
                ' time steps but there are ' + str(shape[0]) +
                'entries in your pulse. Please make sure that you have set the'
                'time steps correctly.'
            )

        if not shape[1] == self.num_ctrls:
            raise IndexError(
                'The pulse data must have exactly as many pulses as there are'
                'control signals. You initialised this transfer function for'
                + str(self.num_ctrls) +
                'control signals but your pulse data has ' + str(shape[1]) +
                'control pulses. Please make sure you set the parameter'
                'num_ctrls correctly when initializing the transfer function.'
            )


class IdentityTF(TransferFunction):
    """Numerically efficient identity transfer function which does not change
    pulse nor time steps.

    Base class functions __call__ and gradient_chane_rule are reimplemented in
    order to avoid setting a transfer matrix.

    """
    def __init__(self, num_ctrls=1):
        super().__init__(
            bound_type=None,
            oversampling=1,
            num_ctrls=num_ctrls,
            offset=0.
        )
        self.name = 'Identity'

    def __call__(self, y: np.array) -> np.array:
        """See base class. """
        return y

    def gradient_chain_rule(
            self, deriv_by_transferred_par: np.array) -> np.array:
        """See base class. """
        return deriv_by_transferred_par


class OversamplingTF(TransferFunction):
    """ Handles oversampling and boundaries without transfer matrix.

    This function is destined to be used for the oversampling and the boundary
    functions for all transfer functions which do not compute a transfer
    matrix.

    See Also
    --------

    Base Class

    """
    def __init__(self,
                 num_ctrls: int = 1,
                 bound_type: Optional[Tuple[str, int]] = None,
                 oversampling: int = 1
                 ):
        super().__init__(
            num_ctrls=num_ctrls,
            bound_type=bound_type,
            oversampling=oversampling
        )

    def _calculate_transfer_matrix(self):
        """Overrides the base class method. """
        raise NotImplementedError

    def __call__(self, y: np.array) -> np.array:
        """Calculate the transferred optimization parameters (x).

        Only the oversampling and boundaries are taken into account.

        Parameters
        ----------
        y: np.array, shape (num_y, num_par)
            Raw optimization variables; num_y is the number of time slices of
            the raw optimization parameters and num_par is the number of
            distinct raw optimization parameters.

        Returns
        -------
        u: np.array, shape (num_x, num_par)
            Control parameters; num_u is the number of times slices for the
            transferred optimization parameters.

        """
        self._check_dimensions_datatype(y)

        # oversample pulse by repetition
        u = np.repeat(y, self.oversampling, axis=0)

        # add the padding elements
        padding_start, padding_end = self.num_padding_elements

        u = np.concatenate(
            (np.zeros((padding_start, self.num_ctrls)),
             u,
             np.zeros((padding_end, self.num_ctrls))), axis=0)

        return u

    def gradient_chain_rule(
            self, deriv_by_transferred_par: np.array) -> np.array:
        """
        See base class.

        Processing without transfer matrix.

        Parameters
        ----------
        deriv_by_transferred_par: np.array, shape (num_x, num_f, num_par)
            The gradients of num_f functions by num_par optimization parameters
            at num_x different time steps.

        Returns
        -------
        deriv_by_opt_par: np.array, shape: (num_y, num_f, num_par)
            The derivatives by the optimization parameters at num_y time steps.

        """

        shape = deriv_by_transferred_par.shape
        assert len(shape) == 3
        assert shape[0] == self.num_x
        assert shape[2] == self.num_ctrls

        # delete the padding elements
        padding_start, padding_end = self.num_padding_elements

        # deriv_by_ctrl_amps: shape (num_x, num_f, num_par)
        if padding_end > 0:
            cropped_derivs = deriv_by_transferred_par[
                             padding_start:-padding_end, :, :]
        else:
            cropped_derivs = deriv_by_transferred_par[
                             padding_start:, :, :]

        cropped_derivs = np.expand_dims(cropped_derivs, axis=1)
        cropped_derivs = np.reshape(
            cropped_derivs, (
                self._num_y,
                self.oversampling,
                cropped_derivs.shape[2],
                cropped_derivs.shape[3]
            )
        )
        deriv_by_opt_par = np.sum(cropped_derivs, axis=1)
        return deriv_by_opt_par


class ConvolutionTF(TransferFunction):
    """ A convolution as filter function.

    This implementation uses the function scipy.ndimage.convolve. For
    oversampling and boundaries use this TransferFunction in combination with
    EfficientOversamplingTF and ConcatenateTF.

    Parameters
    ----------

    """

    def __init__(self,
                 kernel: np.ndarray,
                 mode: str = 'nearest',
                 num_ctrls: int = 1,
                 cval: float = 0.0):
        super().__init__(num_ctrls=num_ctrls)
        if len(kernel.shape) == 1:
            kernel = np.expand_dims(kernel, axis=1)
        assert len(kernel.shape) == 2
        self.kernel = kernel
        self.mode = mode
        self.cval = cval

    def __call__(self, y: np.array) -> np.array:
        """ See base class.

        Evaluates the transfer function at the raw optimization parameters (y)
        to calculate the transferred optimization parameters (x).

        Parameters
        ----------
        y: np.array, shape (num_y, num_par)
            Raw optimization variables; num_y is the number of time slices of
            the raw optimization parameters and num_par is the number of
            distinct raw optimization parameters.

        Returns
        -------
        u: np.array, shape (num_x, num_par)
            Control parameters; num_u is the number of times slices for the
            transferred optimization parameters.
        """
        self._check_dimensions_datatype(y)

        return scipy.ndimage.convolve(
            y, weights=self.kernel, mode=self.mode, cval=self.cval
        )

    def gradient_chain_rule(
            self, deriv_by_transferred_par: np.array) -> np.array:
        """ See base class.

        The application of the chain rule is another convolution.

        Parameters
        ----------
        deriv_by_transferred_par: np.array, shape (num_x, num_f, num_par)
            The gradients of num_f functions by num_par optimization parameters
            at num_x different time steps.

        Returns
        -------
        deriv_by_opt_par: np.array, shape: (num_y, num_f, num_par)
            The derivatives by the optimization parameters at num_y time steps.
        """

        shape = deriv_by_transferred_par.shape
        assert len(shape) == 3
        assert shape[0] == self.num_x
        assert shape[2] == self.num_ctrls
        assert deriv_by_transferred_par.dtype in [np.float64, np.float32]

        return scipy.ndimage.convolve(
            deriv_by_transferred_par,
            weights=np.expand_dims(self.kernel, axis=1),
            cval=self.cval
        )

    @property
    def transfer_matrix(self) -> np.array:
        """Overrides the base class method. """
        raise NotImplementedError

    def _calculate_transfer_matrix(self):
        """Overrides the base class method. """
        raise NotImplementedError


class GaussianConvolution(TransferFunction):
    """ A gaussian convolution is applied as filter function.

    For oversampling and boundaries use this TransferFunction in combination
    with EfficientOversamplingTF and ConcatenateTF.
    The implementation makes use of the gaussian filter function in the
    scipy.ndimage package.

    Parameters
    ----------
    sigma: float or sequence of float
        standard deviation for Gaussian kernel

    order: int, optional
        An order of 0 corresponds to convolution with a Gaussian kernel. A
        positive order corresponds to convolution with that derivative of a
        Gaussian.

    mode: {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional
        The mode parameter determines how the input array is extended beyond its
        boundaries. Default is ‘reflect’. Behavior for each valid value is as
        follows:

        ‘reflect’ (d c b a | a b c d | d c b a)
        The input is extended by reflecting about the edge of the last pixel.

        ‘constant’ (k k k k | a b c d | k k k k)
        The input is extended by filling all values beyond the edge with the
        same constant value, defined by the cval parameter.

        ‘nearest’ (a a a a | a b c d | d d d d)
        The input is extended by replicating the last pixel.

        ‘mirror’ (d c b | a b c d | c b a)
        The input is extended by reflecting about the center of the last pixel.

        ‘wrap’ (a b c d | a b c d | a b c d)
        The input is extended by wrapping around to the opposite edge

        Default is 'nearest'.

    truncate: float, optinal
        Truncate the filter at this many standard deviations. Default is 4.0.

    """
    def __init__(self,
                 sigma: Union[float, Sequence[float]],
                 num_ctrls=1,
                 order: int = 0,
                 mode: str = 'nearest',
                 truncate: float = 4.):
        super().__init__(num_ctrls=num_ctrls)
        self.sigma = sigma
        self.order = order
        self.mode = mode
        self.truncate = truncate

    def __call__(self, y: np.array) -> np.array:
        """Calculate the transferred optimization parameters (x).

        Evaluates the transfer function at the raw optimization parameters (y)
        to calculate the transferred optimization parameters (x).

        Parameters
        ----------
        y: np.array, shape (num_y, num_par)
            Raw optimization variables; num_y is the number of time slices of
            the raw optimization parameters and num_par is the number of
            distinct raw optimization parameters.

        Returns
        -------
        u: np.array, shape (num_x, num_par)
            Control parameters; num_u is the number of times slices for the
            transferred optimization parameters.

        Raises
        ------
        ValueError: Indicating that the data shape or type of the argument is
        incompatible with the transfer function configuration.

        """
        self._check_dimensions_datatype(y)

        return scipy.ndimage.gaussian_filter1d(
            y, self.sigma, axis=0, order=self.order, mode=self.mode,
            truncate=self.truncate
        )

    def gradient_chain_rule(
            self, deriv_by_transferred_par: np.array) -> np.array:
        """ See base class.

        The application of the chain rule is another Gaussian filter.

        Parameters
        ----------
        deriv_by_transferred_par: np.array, shape (num_x, num_f, num_par)
            The gradients of num_f functions by num_par optimization parameters
            at num_x different time steps.

        Returns
        -------
        deriv_by_opt_par: np.array, shape: (num_y, num_f, num_par)
            The derivatives by the optimization parameters at num_y time steps.
        """

        shape = deriv_by_transferred_par.shape
        assert len(shape) == 3
        assert shape[0] == self.num_x
        assert shape[2] == self.num_ctrls
        assert deriv_by_transferred_par.dtype in [np.float64, np.float32]

        return scipy.ndimage.gaussian_filter1d(
            deriv_by_transferred_par, self.sigma, axis=0, order=self.order,
            mode=self.mode, truncate=self.truncate
        )

    @property
    def transfer_matrix(self) -> np.array:
        """Overrides the base class method. """
        raise NotImplementedError

    def _calculate_transfer_matrix(self):
        """Overrides the base class method. """
        raise NotImplementedError


class ConcatenateTF(TransferFunction):
    """
    Concatenates transfer functions.

    This class can be used if there are two transfer functions which are to be
    applied one after another. For example if first the pulse generation and
    subsequently a pulse amplification shall be modeled.

    Parameters
    ----------
    tf1: TransferFunction
        First matrix transfer function. This function operates directly on the
        optimization variables.

    tf2: TransferFunction
        Second matrix transfer function. This function operates on the
        output of the first transfer function.

    """
    def __init__(self, tf1: TransferFunction, tf2: TransferFunction):
        offset = 0
        for of in [tf1.offset, tf2.offset]:
            if of is not None:
                offset += of
        super().__init__(
            num_ctrls=tf1.num_ctrls,
            offset=offset,
            oversampling=tf1.oversampling * tf2.oversampling
        )
        self.tf1 = tf1
        self.tf2 = tf2
        self._num_y = tf1._num_y
        self._y_times = tf1._y_times
        self.x_times = tf2.x_times
        self.num_x = tf2.num_x

    def __call__(self, y: np.ndarray, *args, **kwargs):
        """Calls the concatenated transfer functions in sequence."""
        self._check_dimensions_datatype(y)
        return self.tf2(self.tf1(y))

    def gradient_chain_rule(
            self, deriv_by_transferred_par: np.ndarray) -> np.ndarray:
        """Applies the concatenation formula for both transfer functions."""
        intermediate_gradient = self.tf2.gradient_chain_rule(
            deriv_by_transferred_par=deriv_by_transferred_par)
        return self.tf1.gradient_chain_rule(
            deriv_by_transferred_par=intermediate_gradient)

    def plot_pulse(self, y: np.ndarray) -> None:
        """Calls the plot_pulse routine of the second transfer function."""
        self.tf2.plot_pulse(self.tf1(y))

    def set_times(self, y_times: np.ndarray) -> None:
        """
        Sets x_times on the first transfer function and sets the resulting
        x_times on the second transfer function.

        Parameters
        ----------
            y_times: Optional[np.array, list]
                Time durations of the constant control steps of the raw
                optimization parameters.

        """
        self.tf1.set_times(y_times)
        self.tf2.set_times(y_times=self.tf1.x_times)
        self._num_y = self.tf1._num_y
        self._y_times = self.tf1._y_times
        self.x_times = self.tf2.x_times
        self.num_x = self.tf2.num_x
        return


class ParallelTF(TransferFunction):
    """
    This transfer function will parallelize two transfer functions, such that
    they are applied to different control terms. Thus adding in the third
    dimension of the transfer matrix.

    Parameters
    ----------
    tf1: TransferFunction
        First transfer function. This function operates on the first
        tf1.num_ctrls control pulses.

    tf2: TransferFunction
        Second transfer function. This function operates on the
        next tf2._num_ctrls number of control pulses.

    """

    def __init__(self, tf1: TransferFunction, tf2: TransferFunction):
        super().__init__(
            num_ctrls=tf1.num_ctrls + tf2.num_ctrls,
            oversampling=tf1.oversampling,
            offset=None
        )
        self.bound_type = tf1.bound_type

        self.tf1 = tf1
        self.tf2 = tf2

        if not tf1._num_y == tf2._num_y:
            raise ValueError("The parallelized transfer functions must operate"
                             "on the same time frame. The transfer functions"
                             "expect a different number of input time steps.")

        if not tf1.num_x == tf2.num_x:
            raise ValueError("The parallelized transfer functions must operate"
                             "on the same time frame. The transfer functions"
                             "generate signals with a different number of time"
                             " steps.")

        # tf1 and tf2 should have identical times
        self._y_times = tf1._y_times
        self.x_times = tf1.x_times
        self.num_x = self.tf1.num_x
        self._num_y = tf1._num_y

    def __call__(self, y: np.array) -> np.array:
        """See base class.

        The transfer functions are evaluated separatly and the results are
        concatenated.

        """
        self._check_dimensions_datatype(y)
        return np.concatenate(
            (self.tf1(y[:, :self.tf1.num_ctrls]),
             self.tf2(y[:, self.tf1.num_ctrls:])),
            axis=1)

    def gradient_chain_rule(
            self, deriv_by_transferred_par: np.array) -> np.array:
        """ See base class.

        The gradients are calculated separatly and then concatenated.

        """
        grad_1 = self.tf1.gradient_chain_rule(
            deriv_by_transferred_par[:, :, :self.tf1.num_ctrls])
        grad_2 = self.tf2.gradient_chain_rule(
            deriv_by_transferred_par[:, :, self.tf1.num_ctrls:])
        return np.concatenate((grad_1, grad_2), axis=2)

    def set_times(self, y_times: np.ndarray):
        """See base class. """
        self.tf1.set_times(y_times)
        self.tf2.set_times(y_times)
        self._num_y = self.tf1._num_y
        self._y_times = self.tf1._y_times
        self.num_x = self.tf1.num_x
        self.x_times = self.tf1.x_times


class MatrixTF(TransferFunction):

    def __call__(self, y: np.array) -> np.array:
        """Calculate the transferred optimization parameters (x).

        Evaluates the transfer function at the raw optimization parameters (y)
        to calculate the transferred optimization parameters (x).

        Parameters
        ----------
        y: np.array, shape (num_y, num_par)
            Raw optimization variables; num_y is the number of time slices of
            the raw optimization parameters and num_par is the number of
            distinct raw optimization parameters.

        Returns
        -------
        u: np.array, shape (num_x, num_par)
            Control parameters; num_u is the number of times slices for the
            transferred optimization parameters.

        """
        self._check_dimensions_datatype(y)

        if self._transfer_matrix is None:
            self._calculate_transfer_matrix()
        x = np.einsum('ijk,jk->ik', self._transfer_matrix, y)
        if self.offset is not None:
            x += self.offset
        return x

    @property
    def transfer_matrix(self) -> np.array:
        """
        If necessary, calculates the transfer matrix. Then returns it.

        Returns
        -------
        T: ndarray, shape (num_u, num_x, num_ctrl)
            Transfer matrix (the linearization of the control amplitudes).

        """
        if self._transfer_matrix is None:
            self._calculate_transfer_matrix()
        return copy.deepcopy(self._transfer_matrix)

    @abstractmethod
    def _calculate_transfer_matrix(self):
        """Create the transfer matrix. """
        pass

    def gradient_chain_rule(
            self, deriv_by_transferred_par: np.array) -> np.array:
        """ See base class.

        """
        shape = deriv_by_transferred_par.shape
        assert len(shape) == 3
        assert shape[0] == self.num_x
        assert shape[2] == self.num_ctrls

        if self._transfer_matrix is None:
            self._calculate_transfer_matrix()

        # T: shape (num_x, num_y, num_par)
        # deriv_by_ctrl_amps: shape (num_x, num_f, num_par)
        return np.einsum('ijk,ifk->jfk',
                         self._transfer_matrix,
                         deriv_by_transferred_par)


class OversamplingMTF(MatrixTF):
    """Oversamples and applies boundary conditions.

    """
    def __init__(
            self,
            oversampling: int = 1,
            bound_type: Tuple = None,
            num_ctrls: int = 1,
            offset: float = 0):
        super().__init__(
            bound_type=bound_type,
            oversampling=oversampling,
            num_ctrls=num_ctrls,
            offset=offset
        )
        self.name = "Oversampling"

    def _calculate_transfer_matrix(self) -> None:
        """See base class. """
        # identity for each oversampling segment
        transfer_matrix = np.eye(self._num_y)
        transfer_matrix = np.repeat(transfer_matrix, self.oversampling, axis=0)

        # add the padding elements
        padding_start, padding_end = self.num_padding_elements
        transfer_matrix = np.concatenate(
            (np.zeros((padding_start, self._num_y)),
             transfer_matrix,
             np.zeros((padding_end, self._num_y))), axis=0)

        # add control axis
        transfer_matrix = np.expand_dims(transfer_matrix, axis=2)
        transfer_matrix = np.repeat(transfer_matrix, self.num_ctrls, axis=2)

        self._transfer_matrix = transfer_matrix


class ConcatenateMTF(MatrixTF):
    """
    Concatenates two matrix transfer functions.

    Matrix version of ConcatenateTF.

    Parameters
    ----------
    tf1: MatrixTF
        First matrix transfer function. This function operates directly on the
        optimization variables.

    tf2: MatrixTF
        Second matrix transfer function. This function operates on the
        output of the first transfer function.

    """
    def __init__(self, tf1: MatrixTF, tf2: MatrixTF):
        offset = 0
        for of in [tf1.offset, tf2.offset]:
            if of is not None:
                offset += of
        super().__init__(
            num_ctrls=tf1.num_ctrls,
            offset=offset,
            oversampling=tf1.oversampling * tf2.oversampling
        )
        self.tf1 = tf1
        self.tf2 = tf2
        self._num_y = tf1._num_y
        self._y_times = tf1._y_times
        self.x_times = tf2.x_times
        self.num_x = tf2.num_x

    @property
    def transfer_matrix(self):
        """The total transfer matrix is the product of the individual ones."""
        return np.einsum('ijk,jlk->ilk',
                         self.tf2.transfer_matrix,
                         self.tf1.transfer_matrix)

    def set_times(self, y_times: np.ndarray) -> None:
        """
        Sets x_times on the first transfer function and sets the resulting
        x_times on the second transfer function.

        Parameters
        ----------
            y_times: Optional[np.array, list]
                Time durations of the constant control steps of the raw
                optimization parameters.

        """
        self.tf1.set_times(y_times)
        self.tf2.set_times(y_times=self.tf1.x_times)
        self._num_y = self.tf1._num_y
        self._y_times = self.tf1._y_times
        self.x_times = self.tf2.x_times
        self.num_x = self.tf2.num_x
        return

    def plot_pulse(self, y: np.ndarray) -> None:
        """Calls the plot_pulse routine of the second transfer function."""
        self.tf2.plot_pulse(self.tf1(y))

    def _calculate_transfer_matrix(self):
        """See base class. """
        self.tf1._calculate_transfer_matrix()
        self.tf2._calculate_transfer_matrix()


class ParallelMTF(MatrixTF):
    """
    This transfer function will parallelize two transfer functions, such that
    they are applied to different control terms. Thus adding in the third
    dimension of the transfer matrix.

    Parameters
    ----------
    tf1: MatrixTF
        First transfer function. This function operates on the first
        tf1.num_ctrls control pulses.

    tf2: MatrixTF
        Second transfer function. This function operates on the
        next tf2._num_ctrls number of control pulses.

    """

    def __init__(self, tf1: MatrixTF, tf2: MatrixTF):
        super().__init__(
            num_ctrls=tf1.num_ctrls + tf2.num_ctrls,
            oversampling=tf1.oversampling,
            offset=None
        )
        self.bound_type = tf1.bound_type

        self.tf1 = tf1
        self.tf2 = tf2

        assert tf1._num_y == tf2._num_y
        self._num_y = tf1._num_y

        # tf1 and tf2 should have identical times
        self._y_times = tf1._y_times
        self.x_times = tf1.x_times
        self.num_x = self.tf1.num_x

        if not tf1.bound_type == tf2.bound_type:
            raise ValueError("The parallized transfer functions must have the "
                             "same bound_types.")

        if not tf1.oversampling == tf2.oversampling:
            raise ValueError("The parallized transfer functions must have the "
                             "same oversampling.")

    def _calculate_transfer_matrix(self):
        """See base class. """
        self._transfer_matrix = np.concatenate(
            (self.tf1.transfer_matrix, self.tf2.transfer_matrix), axis=2)

    def set_times(self, y_times: np.ndarray):
        """See base class. """
        self.tf1.set_times(y_times)
        self.tf2.set_times(y_times)
        self._num_y = self.tf1._num_y
        self._y_times = self.tf1._y_times
        self.num_x = self.tf1.num_x
        self.x_times = self.tf1.x_times


class CustomMTF(MatrixTF):
    """
    This class implements a linear transfer function.

    The action is fully described by the transfer function and a constant
    offset, given at the initialization of the instance.

    Parameters
    ----------
        transfer_matrix: np.array, shape (num_x, num_y, num_par)
            Constant transfer function.

        offset: float
            Constant offset.

        x_times: np.array, shape (num_x)
            Time slices of the control amplitudes. If they are not explicitly
            given, they are constructed from oversampling and bound_type.

        bound_type: see base class
            If no bound_type is specified. The program assumes that there is no
            padding.

        oversampling: int
            If the oversampling is not explicitly given, it is constructed from
            the bound_type and the transfer matrix.


    `Todo`
        * does it make sense so set the utimes explicitly? breakes the usual
            * workflow

    """
    def __init__(self,
                 transfer_matrix: np.array,
                 x_times: Optional[np.array] = None,
                 bound_type: Optional[Tuple] = None,
                 oversampling: int = 1,
                 offset: Optional[float] = None,
                 num_ctrls: int = 1):
        super().__init__(
            oversampling=oversampling,
            bound_type=bound_type,
            offset=offset,
            num_ctrls=num_ctrls
        )
        self._transfer_matrix = transfer_matrix
        self._num_y = transfer_matrix.shape[1]
        self.num_x = transfer_matrix.shape[0]
        self.x_times = x_times
        self.bound_type = bound_type
        self.oversampling = oversampling

    @property
    def transfer_matrix(self) -> np.ndarray:
        """See base class."""
        return self._transfer_matrix

    def set_times(self, y_times: np.ndarray) -> None:
        """See base class."""
        if self.x_times is None:
            # construct the oversampling
            if self.bound_type is None:
                # assume no padding
                self.oversampling = self.num_x // self._num_y
                if self.num_x % self._num_y:
                    raise ValueError('Dimensions of transfer matrix '
                                     'impossible if no padding is used.'
                                     'State the boundary_type!')
            elif self.bound_type[0] == 'n':
                self.oversampling = (self.num_x - 2
                                     * self.bound_type[1]) / self._num_y
            elif self.bound_type[0] == 'x':
                self.oversampling = self.num_x / (2 * self.bound_type[1]
                                                   + self._num_y)
            elif self.bound_type[0] == 'right_n':
                self.oversampling = (self.num_x - self.bound_type[
                    1]) / self._num_y
            else:
                raise ValueError('Unknown boundary type:'
                                 + str(self.bound_type[0]))
            self.oversampling = int(self.oversampling)
            super().set_times(y_times)
        else:
            y_times = np.squeeze(y_times)
            self.x_times = y_times
            if len(y_times) != self._num_y:
                raise ValueError('Trying to set x_times, which do not fit the'
                                 'dimension of the transfer function.')

    def _calculate_transfer_matrix(self):
        """See base class. """
        if self._transfer_matrix is None:
            raise ValueError("The custom transfer function cannot create its"
                             "transfer matrix. It must be constructed "
                             "externally and set in the init method!")


def exp_saturation(t: float, t_rise: float, val_1: float, val_2: float) -> int:
    """Exponential saturation function."""
    return val_1 + (val_2 - val_1) * (1 - np.exp(-(t / t_rise)))


class ExponentialMTF(MatrixTF):
    """
    This transfer function model smooths the control amplitudes by exponential
    saturation.

    The functionality is meant to model the finite rise time of voltage
    sources.

    `Todo`
        * add initial and final level. Currently fixed at 0 (or the offset)

    """

    def __init__(self, awg_rise_time: float, oversampling: int = 1,
                 bound_type: Tuple = ('x', 0), offset: Optional[float] = None,
                 num_ctrls: int = 1):
        super().__init__(
            oversampling=oversampling,
            bound_type=bound_type,
            num_ctrls=num_ctrls
        )
        self.awg_rise_time = awg_rise_time
        self.offset = offset

    @property
    def transfer_matrix(self) -> np.ndarray:
        """See base class."""
        if self._transfer_matrix is None:
            self._calculate_transfer_matrix()
        return self._transfer_matrix

    @deprecated
    def old_call(self, x: np.ndarray):
        """ `TODO` only alive for testing"""
        start_value = 0
        stop_value = 0

        x_tau = self.xtimes[1] - self.xtimes[0]
        if self.bound_type is None:
            y = np.zeros((self._num_y * self.oversampling))
        elif self.bound_type[0] == 'n':
            y = np.zeros((self._num_y * self.oversampling + self.bound_type[1],
                          self.num_ctrls))
        elif self.bound_type[0] == 'x':
            y = np.zeros(((self._num_y + self.bound_type[1]) * self.oversampling,
                          self.num_ctrls))
        else:
            raise ValueError('The boundary type ' + str(self.bound_type[0])
                             + ' is not implemented!')
        for k in range(self.num_ctrls):
            for j in range(self.oversampling):
                y[j, k] = exp_saturation((j + 1) / self.oversampling * x_tau,
                                         self.awg_rise_time,
                                         start_value[k], x[0, k])
        for k in range(self.num_ctrls):
            for i in range(1, self._num_y):
                for j in range(self.oversampling):
                    y[i * self.oversampling + j, k] = \
                        exp_saturation((j + 1) / self.oversampling * x_tau,
                                       self.awg_rise_time,
                                       x[i - 1, k], x[i, k])
            if self.bound_type[0] == 'n':
                for i in range(self.bound_type[1]):
                    y[self._num_y * self.oversampling + i] = \
                        exp_saturation((i + 1) / self.oversampling * x_tau,
                                       self.awg_rise_time, x[-1, k],
                                       self.stop_value[k])
            elif self.bound_type[0] == 'x':
                for i in range(self.bound_type[1]):
                    for j in range(self.oversampling):
                        y[self._num_y * self.oversampling
                          + i * self.oversampling + j] = \
                            exp_saturation(((j + 1) / self.oversampling + i)
                                           * x_tau, self.awg_rise_time,
                                           x[-1, k], stop_value[k])

        return y

    @deprecated
    def plot_pulse_old(self, x: np.ndarray) -> None:
        """Plot the control amplitudes corresponding to the given optimisation
        variables. """
        u = self(x)
        n_padding_start, n_padding_end = self.num_padding_elements
        for x_per_control, u_per_control in zip(x.T, u.T):
            plt.figure()
            plt.bar(np.cumsum(self._y_times) - .5 * self._y_times[0],
                    u_per_control, self._y_times[0])
            plt.bar(np.cumsum(self._y_times) - .5 * self._y_times[0]
                    + np.cumsum(self._y_times)[n_padding_start]
                    - self._y_times[n_padding_start],
                    x_per_control, self._y_times[0],
                    fill=False)
        plt.show()

    def _calculate_transfer_matrix(self) -> None:
        """Calculate the transfer matrix as function of the oversampling, the
        boundary conditions, the set x_times and the awg rise time.

        Currently only equal time spacing is supported!"""

        num_padding_start, num_padding_end = self.num_padding_elements
        dudx = np.zeros(shape=(self.num_x - num_padding_start, self._num_y))

        x_tau = self._y_times[0]

        # calculate blocks
        exp = np.zeros((self.oversampling,))
        for j in range(self.oversampling):
            t = (j + 1) * x_tau / self.oversampling
            exp[j] = np.exp(-(t / self.awg_rise_time))
        one_minus_exp = np.ones((self.oversampling,)) - exp

        # build 2d gradient matrix

        # for the padding at the beginning
        dudx[0:self.oversampling, 0] = one_minus_exp
        if self.num_x > self.oversampling:
            dudx[self.oversampling:2 * self.oversampling, 0] = exp

        # main part
        for i in range(1, self._num_y - 1):
            dudx[i * self.oversampling:(i + 1) *
                 self.oversampling, i] = one_minus_exp

            dudx[(i + 1) * self.oversampling:(i + 2) *
                 self.oversampling, i] = exp

        # at the end
        dudx[(self._num_y - 1) * self.oversampling:
             self._num_y * self.oversampling, self._num_y - 1] = one_minus_exp

        for i in range(num_padding_end):
            t = (i + 1) / self.oversampling * x_tau
            dudx[self._num_y * self.oversampling + i, -1] = np.exp(
                -(t / self.awg_rise_time))

        # zeros for the first elements
        dudx = np.concatenate((np.zeros(shape=(num_padding_start,
                                               self._num_y)),
                              dudx), axis=0)

        dudx = np.repeat(
            np.expand_dims(dudx, axis=2), repeats=self.num_ctrls, axis=2)
        self._transfer_matrix = dudx

    def gradient_chain_rule(
            self, deriv_by_transferred_par: np.ndarray) -> np.ndarray:
        """See base class. """
        if self._transfer_matrix is None:
            self._calculate_transfer_matrix()

        # T: shape (num_u, num_x, num_ctrl)
        # deriv_by_ctrl_amps: shape (num_u, num_f, num_ctrl)
        return np.einsum('ijk,ifk->jfk', self._transfer_matrix,
                         deriv_by_transferred_par)

    @needs_refactoring
    def reverse_state(self, amplitudes=None, times=None, targetfunc=None):
        """
        I assume only to be applied to Pulses generated by self.__call__(x)
        If times is None:
        We either need to know num_x or the oversampling. For now I assume that
        self.num_x is valid for the input data.
        :param amplitudes:
        :param times
        :param targetfunc:
        :return:
        """

        num_ctrls = amplitudes.shape[1]
        xtau = (self.xtimes[1] - self.xtimes[0])
        if times is not None:
            if times.size < 2:
                # TODO: log warning
                return amplitudes
            tau = times[1] - times[0]
            oversampling = int(round(xtau / tau))
            num_x = times.size // oversampling
        elif amplitudes is not None:
            oversampling = amplitudes.size // num_ctrls // self._num_y
            num_x = self._num_y
        elif targetfunc is not None:
            raise NotImplementedError
        else:
            raise ValueError(
                "please specify the amplitues or the target function! (not yet "
                "implemented for target functions)")

        if amplitudes is not None:
            x = np.zeros((num_x, num_ctrls))
            t = 1 / oversampling * xtau
            exp = np.exp(-(t / self.awg_rise_time))
            for k in range(num_ctrls):
                x[0, k] = (amplitudes[0, k] - self.start_value) / (
                            1 - exp) + self.start_value
                for i in range(1, num_x):
                    x[i, k] = (amplitudes[i * oversampling, k] - x[
                        i - 1, k]) / (1 - exp) + x[i - 1, k]
        elif targetfunc is not None:
            raise NotImplementedError
        else:
            raise ValueError(
                "please specify the amplitues or the target function! (not yet "
                "implemented for target functions)")
        return x


class GaussianMTF(MatrixTF):
    """
    Deprecated! - Might be reimplemented upon reasonable request.

    Represent square function filtered through a gaussian filter.

    Can not be used in conjunction with the concatenate tf.

    Parameters:
    -----------
    omega: (float, list) bandwitdh of the

    over_sample_rate: number of timeslice of the amplitudes for each x block.

    start, end: amplitudes at the boundaries of the time range.

    bound_type = (code, number): control the number of time slice of padding
                                before and after the original time range.

        code:
            "n": n extra slice of dt/overSampleRate
            "x": n extra slice of dt
            "w": go until a dampening of erf(n) (default, n=2)

    The amplitudes time range is wider than the given times.

    `Todo`
        * reworked the comments but the code has not been refactored

    """

    @deprecated
    def __init__(self, omega=1, over_sample_rate=5, start=0., end=0.,
                 bound_type=("w", 2)):
        super().__init__()
        self.N = over_sample_rate
        self.dt = 1 / over_sample_rate
        self.boundary = [start, end]
        self.omega = omega
        self.bound_type = bound_type
        self.name = "Gaussian"

        self._transfer_matrix = None
        self.cte = None

    def _calculate_transfer_matrix(self):
        """Calculate the transfer matrix. """
        Dxt = (self.xtimes[1] - self.xtimes[0]) * 0.25
        self._transfer_matrix = np.zeros((len(self._y_times) - 1, self._num_y, self.num_ctrls))
        self.cte = np.zeros((len(self._y_times) - 1, self.num_ctrls))
        time = (self._y_times[:-1] + self._y_times[1:]) * 0.5
        xtime = (self.xtimes[:-1] + self.xtimes[1:]) * 0.5
        for j, t in enumerate(time):
            self.cte[j] = (0.5 - 0.5 * scipy.special.erf(self.omega * 0.5 * t)) \
                          * self.boundary[0]
            self.cte[j] += (0.5 + 0.5 * scipy.special.erf(
                self.omega * 0.5 * (t - self.xtimes[-1]))) * self.boundary[1]
            for k, xt in enumerate(xtime):
                T = (t - xt) * 0.5
                self._transfer_matrix[j, k] = (scipy.special.erf(self.omega * (T + Dxt))
                                 - scipy.special.erf(self.omega * (T - Dxt))) * 0.5

    def __call__(self, y):
        if self._transfer_matrix is None:
            self._calculate_transfer_matrix()
        try:
            return np.einsum('ijk,jk->ik', self._transfer_matrix, y) + self.cte
        except ValueError:
            print('error')

    @property
    def transfer_matrix(self):
        """See base class. """
        if self._transfer_matrix is None:
            self._calculate_transfer_matrix()
        return self._transfer_matrix

    def gradient_chain_rule(self, deriv_by_transferred_par):
        """See base class. """
        # index i over the u_values
        # index j over the x_values
        # index k over the num_crtls
        # an index for the cost functions is missing J. Teske
        # index l inserted for the cost functions
        try:
            # return np.einsum('ijk,ik->jk', self._transfer_matrix, gradient)
            return np.einsum('ijk,...i->...j', self._transfer_matrix,
                             deriv_by_transferred_par)
        except ValueError:
            print('error')

    def set_times(self, times):
        """
        See base class.

        Times/transferred_time correspond to the timeslot before the interpolation.
        """
        if not np.allclose(np.diff(times), times[1] - times[0]):
            raise Exception("Times must be equaly distributed")

        super().set_times(times)
        # TODO: properly implement 'w'
