"""
Contains a transfer_functions class that allows the optimization variables
of the grape algorithm to not be directly the amplitudes of the control
fields.

In an experimental context, the transfer function is of an ambigous nature.
It can be used to implement the physical relation between the optimization
variables (e.g. voltages applied) to the amplitudes in the modeled control
Hamiltonian (e.g. an exchange interaction energy) but also to smooth pulses
and implement realistic constrains on the control electronics (e.g. finite
rise times of arbitrary waveform generators).

Optimal control methods for rapidly time-varying Hamiltonians, 2011
Motzoi, F. and Gambetta, J. M. and Merkel, S. T. and Wilhelm, F. K.
PhysRevA.84.022307, https://link.aps.org/doi/10.1103/PhysRevA.84.022307

Classes
-------
TransferFunction:
    Abstract base class which defines the interface and implements standard
    functions.

IdentityTF:
    Optimization variables are the amplitudes of the control fields

ConcatenateTF:
    Concatenation of two arbitrary transfer functions.

CustomTF:
    Linear transfer function which receives an explicitly constructed
    constant transfer matrix.

ExponentialTF:
    The amplitudes are smoothed by exponential saturation functions.

FourierTF:
    The amplitudes of the control fields is obtained by the fourier
    series of the optimization variables.
    u[t] = x[i] * sin(t*(i+1)*pi/times[-1])
    The number of frequency used is set during initiation.

SplineTF:
    The amplitudes of the control fields is the spline interpolation
    of the optimization variables.
    The number of sampling per time slices and the start and end
    value are set at initiation.

Gaussian:
    Represent a Gaussian filtering in the frequency domain.
    At initiation, the reference frequency, sampling rate, start
    and end values can be set.

PulseGenCrab:
    Currently not supported

PulseGenCrabFourier:
    Currently not supported

Functions
---------
exp_saturation:
    Exponential saturation function.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import interpolate
from scipy.special import erf
from typing import Tuple, Callable, Optional
from abc import ABC, abstractmethod
from qsim.util import deprecated, needs_refactoring
import copy


class TransferFunction(ABC):
    """
    A class for representing transfer functions, between optimization
    variables of the optimization algorithm and the amplitudes of the control
    fields.

    The intended workflow is to initialise the transfer function object first
    and subsequently set the x_times, which is the time scale of the
    optimization variables. Then the transfer function is called to calculate
    control amplitudes and gradients.

    Minimal code example with abstract base class:
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

    Attributes
    ----------
    _num_x: int
        Number of time slices of the optimization variables.

    _num_u: int
        Number of time slices of the control amplitudes.

    num_ctrls: int
        Number of controlled amplitudes.

    _u_times: ndarray, shape (num_u)
        Time values for the actual pulses (controlled amplitudes). These
        describe the lenght of the time slices.

    _x_times: ndarray, shape (num_x)
        Time values for the control variables. These  describe the lenght of the
        time slices.

    _absolute_x_times : array_like, shape (num_x + 1)
        Absolute times of the optimization variables. The values describe the
        point in time where a time slice ends or begins.

    _x_max: float
        Maximal value of the optimization variables.

    _x_min: float
        Minimal value of the optimization variables.

    oversampling: int
        Each time step of the optimization variables is sliced into a number
        of time steps of the control amplitudes. This number is oversampling.

    bound_type: (code, number)
        Control the number of time slice of padding before and after the
        original time range. Let dt denote the first or respectively last time
        duration of the optimization variables.

        code:
            "n": n extra slice of dt/overSampleRate
            "x": n extra slice of dt (default with n=1)
            "right_n": n extra slice of dt/overSampleRage on the right side

    Methods
    -------
    __call__(x):


    set_times(times):
        Set the times of the optimization variables and calculates the times
        of the optimization variables.

    set_absolute_times(absolute_x_times):
        Set the absolute times (time points of beginning and ending a time step)
        of the optimization variables.

    plot_pulse(x):
        For the optimisation variables (x), plot the resulting pulse.

    T: ndarray, shape (num_u, num_x, num_ctrl)
        Returns the transfer matrix, which is the linearization of the
        functional dependency between the field amplitudes (u) and the
        control variables.

    num_padding_elements: list
        Two elements list with the number of elements padded to the beginning
        and end, as specified by the bound type.

    TODO:
        * parse bound_type to raise exception only in one function.
        * add exception to the docstring
        * _x_max, _x_min only useful for deprecated functions
        * combinator transfer function that allows multiple controls with
            * distinct transfer functions
        * use the name attribute?
        * refactor the use of the offset. maybe make it an array of the
            * shape of u

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

        self._T = None

        # num_x is set, by setting the time
        self._num_x = 0
        self._x_times = None
        self._absolute_x_times = None
        # num_u is calculated when setting the time from the
        self._num_u = 0
        self._u_times = None

        # deprecated
        self._x_max = None
        self._x_min = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return the amplitudes (u) from the optimisation variables (x).

        Parameters
        ----------
        x: np.ndarray, shape (num_x, num_ctrl)
            Optimization variables; num_x is the number of time slices for the
            optimization and num_ctrl is the number of quantities under control.

        Returns
        -------
        u: np.ndarray, shape (num_u, num_ctrl)
            Control parameters; num_u is the number of times slices for the
            control amplitudes.
        """
        shape = x.shape
        assert len(shape) == 2
        assert shape[0] == self._num_x
        assert shape[1] == self.num_ctrls

        if self._T is None:
            self._make_T()
        u = np.einsum('ijk,jk->ik', self._T, x)
        if self.offset is not None:
            u += self.offset
        return u

    @property
    def T(self) -> np.ndarray:
        """
        If necessary, calculates the transfer matrix. Then returns it.

        Returns
        -------
        T: ndarray, shape (num_u, num_x, num_ctrl)
            Transfer matrix (the linearization of the control amplitudes).

        """
        if self._T is None:
            self._make_T()
        return copy.deepcopy(self._T)

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

    def gradient_chain_rule(self, deriv_by_ctrl_amps: np.ndarray) -> np.ndarray:
        """
        Obtain the derivatives of a quantity a i.e. da/dx by the optimization
        variables from the derivatives by the amplitude of the control fields.

        The chain rule applies: df/dx = df/du * du/dx.

        Parameters
        ----------
        deriv_by_ctrl_amps: np.array, shape (num_u, num_f, num_ctrl)
            The gradients of num_f fidelities by num_ctrl different pulses at
            num_u different time steps.

        Returns
        -------
        deriv_by_opt_par: np.array, shape: (num_x, num_f, num_ctrl)
            The derivatives by the optimization parameters.

        """

        shape = deriv_by_ctrl_amps.shape
        assert len(shape) == 3
        assert shape[0] == self._num_u
        assert shape[2] == self.num_ctrls

        if self._T is None:
            self._make_T()

        # T: shape (num_u, num_x, num_ctrl)
        # deriv_by_ctrl_amps: shape (num_u, num_f, num_ctrl)
        return np.einsum('ijk,ifk->jfk', self._T, deriv_by_ctrl_amps)

    def set_times(self, x_times: np.ndarray) -> None:
        """
        Generate the time_slot duration array 'tau' (here: u_times)

        This time slices depend on the oversampling of the control variables
        and the boundary conditions. The times are for the intended use cases
        only set once.

        Parameters
        ----------
        x_times: Union[np.ndarray, list]
            The time steps / durations of constant optimization variables.

        """
        if isinstance(x_times, list):
            x_times = np.array(x_times)
        if not isinstance(x_times, np.ndarray):
            raise Exception("times must be a list or np.array")

        x_times = np.squeeze(x_times)

        if len(x_times.shape) > 1:
            raise ValueError('The x_times should not have more than one '
                             'dimension!')

        self._num_x = x_times.size
        self._x_times = x_times

        if self.bound_type is None:
            self._num_u = self.oversampling * self._num_x
            self._u_times = np.repeat(self._x_times, self.oversampling) \
                            / self.oversampling

        elif self.bound_type[0] == 'n':
            self._num_u = self.oversampling * self._num_x + 2 * self.bound_type[1]
            self._u_times = np.concatenate((
                self._x_times[0] / self.oversampling
                * np.ones(self.bound_type[1]),
                np.repeat(self._x_times / self.oversampling, self.oversampling),
                self._x_times[-1] / self.oversampling
                * np.ones(self.bound_type[1])))

        elif self.bound_type[0] == 'x':
            self._num_u = self.oversampling * (self._num_x
                                               + 2 * self.bound_type[1])
            self._u_times = np.concatenate((
                self._x_times[0] / self.oversampling
                * np.ones(self.bound_type[1] * self.oversampling),
                np.repeat(self._x_times / self.oversampling, self.oversampling),
                self._x_times[-1] / self.oversampling
                * np.ones(self.bound_type[1] * self.oversampling)))

        elif self.bound_type[0] == 'right_n':
            self._num_u = self.oversampling * self._num_x + self.bound_type[1]
            self._u_times = np.concatenate((
                np.repeat(self._x_times / self.oversampling, self.oversampling),
                self._x_times[-1] / self.oversampling
                * np.ones(self.bound_type[1])))

        else:
            raise ValueError('The boundary type ' + str(self.bound_type[0])
                             + ' is not implemented!')

    def set_absolute_times(self, absolute_x_times: np.ndarray) -> None:
        """
        Generate the time_slot duration array 'tau' (here: u_times)

        This time slices depend on the oversampling of the control variables
        and the boundary conditions. The differences of the absolute times
        give the time steps x_times.

        Parameters
        ----------
        absolute_x_times: Union[np.ndarray, list]
            Absolute times of the start / end of each time segment.

        """
        if isinstance(absolute_x_times, list):
            absolute_x_times = np.array(absolute_x_times)
        if not isinstance(absolute_x_times, np.ndarray):
            raise Exception("times must be a list or np.array")
        if not np.all(np.diff(absolute_x_times) >= 0):
            raise Exception("times must be sorted")

        self._absolute_x_times = absolute_x_times
        self.set_times(np.diff(absolute_x_times))

    def plot_pulse(self, x: np.ndarray) -> None:
        """
        Plot the control amplitudes corresponding to the given optimisation
        variables.
        """

        u = self(x)
        n_padding_start, n_padding_end = self.num_padding_elements
        for x_per_control, u_per_control in zip(x.T, u.T):
            plt.figure()
            plt.bar(np.cumsum(self._u_times) - .5 * self._u_times[0],
                    u_per_control, self._u_times[0])
            plt.bar(np.cumsum(self._x_times) - .5 * self._x_times[0]
                    + np.cumsum(self._u_times)[n_padding_start]
                    - self._u_times[n_padding_start],
                    x_per_control, self._x_times[0],
                    fill=False)
        plt.show()
        """
        u = self(x)
        t = np.asarray(self.u_times[:-1] + self.u_times[1:]) * 0.5
        dt = np.diff(self.u_times)
        xt = np.asarray(self.x_times[:-1] + self.x_times[1:]) * 0.5
        dxt = np.diff(self.x_times)
        for i in range(self.num_ctrls):
            plt.bar(t, u[:, i], dt)
            plt.bar(xt, x[:, i], dxt, fill=False)
            plt.show()
        """

    @abstractmethod
    def _make_T(self):
        """Create the transfer matrix. """
        pass


class IdentityTF(TransferFunction):
    """Identity as transfer function."""
    def __init__(
            self,
            oversampling: int = 1,
            bound_type: Tuple = None,
            num_ctrls: int = 1,
            offset: float = 0):
        super().__init__(
            bound_type=bound_type,
            oversampling=oversampling,
            num_ctrls=num_ctrls
        )
        self.name = "Identiy"
        self.bound_type = bound_type
        self.oversampling = oversampling
        self.offset = offset

    def _make_T(self) -> None:
        # identity for each oversampling segment
        transfer_matrix = np.eye(self._num_x)
        transfer_matrix = np.repeat(transfer_matrix, self.oversampling, axis=0)

        # add the padding elements
        padding_start, padding_end = self.num_padding_elements
        transfer_matrix = np.concatenate(
            (np.zeros((padding_start, self._num_x)),
             transfer_matrix,
             np.zeros((padding_end, self._num_x))), axis=0)

        # add control axis
        transfer_matrix = np.expand_dims(transfer_matrix, axis=2)
        transfer_matrix = np.repeat(transfer_matrix, self.num_ctrls, axis=2)

        self._T = transfer_matrix


class LinearTF(IdentityTF):
    """
    A linear transfer function.

    Parameters
    -----------
    linear_factor: float
        The factor by which the optimization parameters are multiplied to
        calculate the control amplitudes.

    """
    def __init__(
            self,
            oversampling: int = 1,
            bound_type: Tuple = None,
            num_ctrls: int = 1,
            offset: Optional[float] = None,
            linear_factor: float = 1
    ):
        super().__init__(
            bound_type=bound_type,
            oversampling=oversampling,
            num_ctrls=num_ctrls,
            offset=offset
        )
        self.linear_factor = linear_factor

    def _make_T(self):
        """See base class. """
        # The parent class creates the identity.
        super()._make_T()
        self._T *= self.linear_factor


class ConcatenateTF(TransferFunction):
    """
    Concatenates two transfer functions.

    Parameters
    ----------
    tf1: TransferFunction
        First transfer function. This function operates directly on the
        optimization variables.

    tf2: TransferFunction
        Second transfer function. This function operates on the
        output of the first transfer function.
    """
    def __init__(self, tf1: TransferFunction, tf2: TransferFunction):
        super().__init__()
        self.tf1 = tf1
        self.tf2 = tf2
        self.num_x = tf1._num_x
        self.num_ctrls = tf1.num_ctrls
        self.u_times = tf2._u_times
        self.xtimes = tf1._x_times
        self.x_max = self.tf1._x_max
        self.x_min = self.tf2._x_min

    def __call__(self, x: np.ndarray, *args, **kwargs):
        """Calls the concatenated transfer functions in sequence."""
        return self.tf2(self.tf1(x))

    @property
    def T(self):
        """The total transfer matrix is the product of the individual ones."""
        return np.einsum('ijk,jlk->ilk', self.tf2.T, self.tf1.T)

    def reverse_state(self, amplitudes: np.ndarray = None,
                      times: np.ndarray = None,
                      targetfunc: Callable = None) -> np.ndarray:
        """Calls the reverse_state functions of its members."""
        intermediate_state = self.tf2.reverse_state(amplitudes=amplitudes,
                                                    times=times,
                                                    targetfunc=targetfunc)
        return self.tf1.reverse_state(amplitudes=intermediate_state,
                                      times=times, targetfunc=targetfunc)

    def gradient_chain_rule(self, deriv_by_ctrl_amps: np.ndarray) -> np.ndarray:
        """Applies the concatenation formula for both transfer functions."""
        intermediate_gradient = self.tf2.gradient_chain_rule(
            deriv_by_ctrl_amps=deriv_by_ctrl_amps)
        return self.tf1.gradient_chain_rule(deriv_by_ctrl_amps=intermediate_gradient)

    def set_times(self, x_times: np.ndarray) -> None:
        """
        Sets x_times on the first transfer function and sets the resulting
        u_times on the second transfer function.

        Parameters
        ----------
            x_times: Optional[np.array, list]
                Time durations of the constant control steps of the optimization
                variables.
        """
        self.tf1.set_times(x_times)
        return self.tf2.set_times(x_times=self.tf1._u_times)

    def plot_pulse(self, x: np.ndarray) -> None:
        """Calls the plot_pulse routine of the second transfer function."""
        self.tf2.plot_pulse(self.tf1(x))

    def _make_T(self):
        """See base class. """
        self.tf1._make_T()
        self.tf2._make_T()

    @deprecated
    def get_xlimit(self):
        return self.tf1.get_xlimit()


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
            bound_type=tf1.bound_type,
            oversampling=tf1.oversampling,
            offset=None
        )
        self.tf1 = tf1
        self.tf2 = tf2

        assert tf1._num_x == tf2._num_x
        self._num_x = tf1._num_x

        # tf1 and tf2 should have identical times
        self.u_times = tf1._u_times
        self.xtimes = tf1._x_times

        if not tf1.bound_type == tf2.bound_type:
            raise ValueError("The parallized transfer functions must have the "
                             "same bound_types.")

        if not tf1.oversampling == tf2.oversampling:
            raise ValueError("The parallized transfer functions must have the "
                             "same oversampling.")

    def _make_T(self):
        self._T = np.concatenate((self.tf1.T, self.tf2.T), axis=2)

    def set_times(self, x_times: np.ndarray):
        """See base class. """
        self.tf1.set_times(x_times)
        self.tf2.set_times(x_times)
        self._num_x = self.tf1._num_x
        self._x_times = self.tf1._x_times
        self._num_u = self.tf1._num_u
        self._u_times = self.tf1._u_times


class CustomTF(TransferFunction):
    """
    This class implements a linear transfer function.

    The action is fully described by the transfer function and a constant
    offset.

    Parameters
    ----------
        T: np.ndarray, shape (num_u, num_x, num_ctrl)
            Constant transfer function.

        offset: float
            Constant offset.

        u_times: np.ndarray, shape (num_u)
            Time slices of the control amplitudes. If they are not explicitly
            given, they are constructed from oversampling and bound_type.

        bound_type: see base class
            If no bound_type is specified. The program assumes that there is no
            padding.

        oversampling: int
            If the oversampling is not explicitly given, it is constructed from
            the bound_type and the transfer matrix.


    TODO:
        * does it make sense so set the utimes explicitly? breakes the usual
            * workflow

    """
    def __init__(self,
                 T: np.ndarray,
                 u_times: Optional[np.ndarray] = None,
                 bound_type: Optional[Tuple] = None,
                 oversampling: Optional[int] = None,
                 offset: Optional[float] = None,
                 num_ctrls: int = 1):
        super().__init__(
            oversampling=oversampling,
            bound_type=bound_type,
            offset=offset,
            num_ctrls=num_ctrls
        )
        self._T = T
        self._num_x = T.shape[1]
        self._num_u = T.shape[0]
        self.u_times = u_times
        self.bound_type = bound_type
        self.oversampling = oversampling

    @property
    def T(self) -> np.ndarray:
        """See base class."""
        return self._T

    def set_times(self, x_times: np.ndarray) -> None:
        """See base class."""
        if self.u_times is None:
            if self.oversampling is None:
                # construct the oversampling
                if self.bound_type is None:
                    # assume no padding
                    self.oversampling = self._num_u // self._num_x
                    if self._num_u % self._num_x:
                        raise ValueError('Dimensions of transfer matrix '
                                         'impossible if no padding is used.'
                                         'State the boundary_type!')
                elif self.bound_type[0] == 'n':
                    self.oversampling = (self._num_u - 2
                                         * self.bound_type[1]) / self._num_x
                elif self.bound_type[0] == 'x':
                    self.oversampling = self._num_u / (2 * self.bound_type[1]
                                                       + self._num_x)
                elif self.bound_type[0] == 'right_n':
                    self.oversampling = (self._num_u - self.bound_type[
                        1]) / self._num_x
                else:
                    raise ValueError('Unknown boundary type:'
                                     + str(self.bound_type[0]))

            super().set_times(x_times)
        else:
            x_times = np.squeeze(x_times)
            self.x_times = x_times
            if len(x_times) != self._num_x:
                raise ValueError('Trying to set x_times, which do not fit the'
                                 'dimension of the transfer function.')

    def _make_T(self):
        """See base class. """
        if self._T is None:
            raise ValueError("The custom transfer function cannot create its"
                             "transfer matrix. It must be constructed "
                             "externally and set in the init method!")


def exp_saturation(t: float, t_rise: float, val_1: float, val_2: float) -> int:
    """Exponential saturation function."""
    return val_1 + (val_2 - val_1) * (1 - np.exp(-(t / t_rise)))


class ExponentialTF(TransferFunction):
    """
    This transfer function models smooths the control amplitudes by exponential
    saturation.

    The functionality is meant to model the finite rise time of voltage sources.

    TODO:
        * add initial and final level. Currently fixed at 0 (or the offset)

    See also base class.

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
    def T(self) -> np.ndarray:
        """See base class."""
        if self._T is None:
            self._make_T()
        return self._T

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """See base class."""
        if self._T is None:
            self._make_T()
        u = np.einsum('ijk,jk->ik', self._T, x)
        if self.offset is not None:
            u += self.offset
        return u

    @deprecated
    def old_call(self, x: np.ndarray):
        """TODO: only alive for testing"""
        start_value = 0
        stop_value = 0

        x_tau = self.xtimes[1] - self.xtimes[0]
        if self.bound_type is None:
            y = np.zeros((self._num_x * self.oversampling))
        elif self.bound_type[0] == 'n':
            y = np.zeros((self._num_x * self.oversampling + self.bound_type[1],
                          self.num_ctrls))
        elif self.bound_type[0] == 'x':
            y = np.zeros(((self._num_x + self.bound_type[1]) * self.oversampling,
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
            for i in range(1, self._num_x):
                for j in range(self.oversampling):
                    y[i * self.oversampling + j, k] = \
                        exp_saturation((j + 1) / self.oversampling * x_tau,
                                       self.awg_rise_time,
                                       x[i - 1, k], x[i, k])
            if self.bound_type[0] == 'n':
                for i in range(self.bound_type[1]):
                    y[self._num_x * self.oversampling + i] = \
                        exp_saturation((i + 1) / self.oversampling * x_tau,
                                       self.awg_rise_time, x[-1, k],
                                       self.stop_value[k])
            elif self.bound_type[0] == 'x':
                for i in range(self.bound_type[1]):
                    for j in range(self.oversampling):
                        y[self._num_x * self.oversampling
                          + i * self.oversampling + j] = \
                            exp_saturation(((j + 1) / self.oversampling + i)
                                           * x_tau, self.awg_rise_time,
                                           x[-1, k], stop_value[k])

        return y

    def plot_pulse_old(self, x: np.ndarray) -> None:
        """Plot the control amplitudes corresponding to the given optimisation
        variables. """
        u = self(x)
        n_padding_start, n_padding_end = self.num_padding_elements
        for x_per_control, u_per_control in zip(x.T, u.T):
            plt.figure()
            plt.bar(np.cumsum(self._u_times) - .5 * self._u_times[0],
                    u_per_control, self._u_times[0])
            plt.bar(np.cumsum(self._x_times) - .5 * self._x_times[0]
                    + np.cumsum(self._u_times)[n_padding_start]
                    - self._u_times[n_padding_start],
                    x_per_control, self._x_times[0],
                    fill=False)
        plt.show()

    def _make_T(self) -> None:
        """Calculate the transfer matrix as function of the oversampling, the
        boundary conditions, the set x_times and the awg rise time.

        Currently only equal time spacing is supported!"""

        num_padding_start, num_padding_end = self.num_padding_elements
        dudx = np.zeros(shape=(self._num_u - num_padding_start, self._num_x))

        x_tau = self._x_times[0]

        # calculate blocks
        exp = np.zeros((self.oversampling,))
        for j in range(self.oversampling):
            t = (j + 1) * x_tau / self.oversampling
            exp[j] = np.exp(-(t / self.awg_rise_time))
        one_minus_exp = np.ones((self.oversampling,)) - exp

        # build 2d gradient matrix

        # for the padding at the beginning
        dudx[0:self.oversampling, 0] = one_minus_exp
        dudx[self.oversampling:2 * self.oversampling, 0] = exp

        # main part
        for i in range(1, self._num_x - 1):
            dudx[i * self.oversampling:(i + 1) *
                 self.oversampling, i] = one_minus_exp

            dudx[(i + 1) * self.oversampling:(i + 2) *
                 self.oversampling, i] = exp

        # at the end
        dudx[(self._num_x - 1) * self.oversampling:self._num_x *
                                                   self.oversampling, self._num_x - 1] = one_minus_exp

        for i in range(num_padding_end):
            t = (i + 1) / self.oversampling * x_tau
            dudx[self._num_x * self.oversampling + i, -1] = np.exp(
                -(t / self.awg_rise_time))

        dudx = np.concatenate((np.zeros(shape=(num_padding_start, self._num_x)),
                              dudx), axis=0)

        dudx = np.repeat(
            np.expand_dims(dudx, axis=2), repeats=self.num_ctrls, axis=2)
        self._T = dudx

    def gradient_chain_rule(self, deriv_by_ctrl_amps: np.ndarray) -> np.ndarray:
        """See base class. """
        if self._T is None:
            self._make_T()

        # T: shape (num_u, num_x, num_ctrl)
        # deriv_by_ctrl_amps: shape (num_u, num_f, num_ctrl)
        return np.einsum('ijk,ifk->jfk', self._T, deriv_by_ctrl_amps)

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
            oversampling = amplitudes.size // num_ctrls // self._num_x
            num_x = self._num_x
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


class Gaussian(TransferFunction):
    """
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

    Todo:
        * reworked the comments but the code has not been refactored

    """

    def __init__(self, omega=1, over_sample_rate=5, start=0., end=0.,
                 bound_type=("w", 2)):
        super().__init__()
        self.N = over_sample_rate
        self.dt = 1 / over_sample_rate
        self.boundary = [start, end]
        self.omega = omega
        self.bound_type = bound_type
        self.name = "Gaussian"

        self._T = None
        self.cte = None

    def make_T(self):
        """Calculate the transfer matrix. """
        Dxt = (self.xtimes[1] - self.xtimes[0]) * 0.25
        self._T = np.zeros((len(self._u_times) - 1, self._num_x, self.num_ctrls))
        self.cte = np.zeros((len(self._u_times) - 1, self.num_ctrls))
        time = (self._u_times[:-1] + self._u_times[1:]) * 0.5
        xtime = (self.xtimes[:-1] + self.xtimes[1:]) * 0.5
        for j, t in enumerate(time):
            self.cte[j] = (0.5 - 0.5 * erf(self.omega * 0.5 * t)) \
                          * self.boundary[0]
            self.cte[j] += (0.5 + 0.5 * erf(
                self.omega * 0.5 * (t - self.xtimes[-1]))) * self.boundary[1]
            for k, xt in enumerate(xtime):
                T = (t - xt) * 0.5
                self._T[j, k] = (erf(self.omega * (T + Dxt))
                                 - erf(self.omega * (T - Dxt))) * 0.5

    def __call__(self, x):
        if self._T is None:
            self.make_T()
        try:
            return np.einsum('ijk,jk->ik', self._T, x) + self.cte
        except ValueError:
            print('error')

    @property
    def T(self):
        """See base class. """
        if self._T is None:
            self.make_T()
        return self._T

    def gradient_chain_rule(self, deriv_by_ctrl_amps):
        """See base class. """
        # index i over the u_values
        # index j over the x_values
        # index k over the num_crtls
        # an index for the cost functions is missing J. Teske
        # index l inserted for the cost functions
        try:
            # return np.einsum('ijk,ik->jk', self._T, gradient)
            return np.einsum('ijk,...i->...j', self._T, deriv_by_ctrl_amps)
        except ValueError:
            print('error')

    def set_times(self, times):
        """
        See base class.

        Times/tau correspond to the timeslot before the interpolation.
        """
        if not np.allclose(np.diff(times), times[1] - times[0]):
            raise Exception("Times must be equaly distributed")

        super().set_times(times)
        # TODO: properly implement 'w'
