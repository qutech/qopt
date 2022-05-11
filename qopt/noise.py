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
"""
This file contains classes and helper functions for the sampling of noise
distributions.

The `NoiseTraceGenerator` class and its children are used by the `Solver` class
to generate noise samples. They use various helper functions for the respective
sampling methods. In the current version, the sampling of a quasi static
distribution and the spectral noise density of fast noise is supported.

Classes
-------
:class:`NoiseTraceGenerator`
    Abstract base class defining the interface of the noise trace generators.

:class:`NTGQuasiStatic`
    Generates noise traces for quasi static noise.

:class:`NTGColoredNoise`
    Generates noise traces of arbitrary colored spectra.

Functions
---------
:func:`bell_curve_1dim`
    One dimensional bell curve.

:func:`sample_1dim_gaussian_distribution`
    Draw samples from the one dimensional bell curve.

:func:`bell_curve_2dim`
    Two dimensional bell curve.

:func:`sample_2dim_gaussian_distribution`
    Draw samples from the two dimensional bell curve.

:func:`fast_colored_noise`
    Samples an arbitrary colored noise spectrum.

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

from typing import Callable, Tuple, Optional, List, Union
from abc import ABC, abstractmethod
from scipy import signal


def bell_curve_1dim(x: Union[np.ndarray, float],
                    stdx: float) -> Union[np.ndarray, float]:
    """
    One dimensional Bell curve.

    Parameters
    ----------
    x: np.ndarray or float
        Point at which the bell curve is evaluated.

    stdx: float
        Standard deviation of the bell curve.

    Returns
    -------
    out: np.ndarray or scalar
        Function values.

    """
    normalization_factor = 1 / 2 / np.pi / stdx
    exponential = np.exp(-.5 * ((x / stdx) ** 2))
    return normalization_factor * exponential


def sample_1dim_gaussian_distribution(std1: float,
                                      n_samples: int) -> List:
    """
    Returns 'n_samples' samples from the one dimensional bell curve.

    The samples are chosen such, that the integral over the bell curve between
    two adjacent samples is always the same.

    Parameters
    ----------
    std1: float
        Standard deviation of the bell curve.

    n_samples: int
        Number of samples returned.

    Returns
    -------
    selected_x: list of float
        Noise samples.

    """
    x = np.mgrid[-5 * std1:5.0001 * std1:0.001 * std1]
    normal_distribution = bell_curve_1dim(x, std1)
    normal_distribution /= np.sum(normal_distribution)
    selected_x = []
    it_sum = 0
    iterator = 1
    for i in range(10000):
        it_sum += normal_distribution[i]
        if it_sum > iterator / (n_samples + 1):
            iterator += 1
            selected_x.append(x[i])
    return selected_x


def bell_curve_2dim(x: Union[np.ndarray, float], stdx: float,
                    y: Union[np.ndarray, float], stdy: float) \
        -> Union[np.ndarray, float]:
    """
    Two dimensional Bell curve.

    Parameters
    ----------
    x: np.ndarray or float
        First dimension value at which the bell curve is evaluated.

    stdx: float
        Standard deviation of the bell curve in the x dimension.

    y: np.ndarray or float
        Second dimension value at which the bell curve is evaluated.

    stdy: float
        Standard deviation of the bell curve in the y dimension.

    Returns
    -------
    out: np.ndarray or scalar
        Function values.

    """
    normalization_factor = 1 / 2 / np.pi / stdx / stdy
    exponential = np.exp(-.5 * ((x / stdx) ** 2 + (y / stdy) ** 2))
    return normalization_factor * exponential


def sample_2dim_gaussian_distribution(
        std1: float, std2: float, n_samples: int) \
        -> (List, List):
    """
    Returns 'n_samples' samples from the two dimensional bell curve.

    The samples are chosen such, that the integral over the bell curve between
    two adjacent samples is always the same.

    Parameters
    ----------
    std1: float
        Standard deviation of the bell curve in the first dimension.

    std2: float
        Standard deviation of the bell curve in the second dimension.

    n_samples: int
        Number of samples returned.

    Returns
    -------
    selected_x: np.ndarray,
        X values of the noise samples.

    selected_y: np.ndarray,
        Y values of the noise samples.

    """
    x, y = np.mgrid[-5 * std1:5.0001 * std1:0.001 * std1,
           -5 * std2:5.0001 * std2:0.001 * std2]
    normal_distribution = bell_curve_2dim(x, std1, y, std2)
    normal_distribution /= np.sum(normal_distribution)
    selected_x = []
    selected_y = []
    cumulative = 0
    iterator = 1
    for i in range(10000):
        for j in range(10000):
            cumulative += normal_distribution[i, j]
            if cumulative > iterator / (n_samples + 1):
                iterator += 1
                selected_x.append(x[i, j])
                selected_y.append(y[i, j])
    return selected_x, selected_y


def fast_colored_noise(spectral_density: Callable, dt: float, n_samples: int,
                       output_shape: Tuple, r_power_of_two=False
                       ) -> np.ndarray:
    """
    Generates noise traces of arbitrary colored noise.

    Use this code for validation:
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> traces = fast_colored_noise(spectral_density, dt, n_samples,
    >>>                             output_shape)
    >>> f_max = 1 / dt
    >>> f, S = signal.welch(traces, f_max, axis=-1)
    >>> plt.loglog(f, spectral_density(f))
    >>> plt.loglog(f, S.mean(axis=0))

    Parameters
    ----------
    spectral_density: Callable
        The one sided spectral density as function of frequency.

    dt: float
        Time distance between two samples.

    n_samples: int
        Number of samples.

    output_shape: tuple of int
        Shape of the noise traces to be returned.

    r_power_of_two: bool
        If true, then n_samples is rounded downwards to the next power of 2 for
        an efficient fast fourier transform.

    Returns
    -------
    delta_colored: np.ndarray, shape(output_shape, actual_n_samples)
        Where actual_n_samples is n_samples or the largest power of 2 smaller
        than n_samples if r_power_of_two is true.

    """
    f_max = 1 / dt
    f_nyquist = f_max / 2
    s0 = 1 / f_nyquist
    if r_power_of_two:
        actual_n_samples = int(2 ** np.ceil(np.log2(n_samples)))
    else:
        actual_n_samples = int(n_samples)

    delta_white = np.random.randn(*output_shape, actual_n_samples)
    delta_white_ft = np.fft.rfft(delta_white, axis=-1)
    # Only positive frequencies since FFT is real and therefore symmetric
    f = np.linspace(0, f_nyquist, actual_n_samples // 2 + 1)
    f[1:] = spectral_density(f[1:])
    f[0] = 0
    delta_colored = np.fft.irfft(delta_white_ft * np.sqrt(f / s0),
                                 n=actual_n_samples, axis=-1)
    # the ifft takes r//2 + 1 inputs to generate r outputs

    return delta_colored


class NoiseTraceGenerator(ABC):
    """
    Abstract base class defining the interface of the noise trace generators.

    Parameters
    ----------
    n_samples_per_trace: int
        Number of noise samples per trace.

    n_traces: int, optional
        Number of noise traces. Default is 1.

    n_noise_operators: int, optional
        Number of noise operators. Default is 1.

    noise_samples: None or np.ndarray, optional,
        shape (n_noise_operators, n_traces, n_samples_per_trace)
        Precalculated noise samples. Defaults to None.

    always_redraw_samples: bool
        If true. The samples are always redrawn upon request. The stored samples
        are not returned.

    Attributes
    ----------
    always_redraw_samples: bool
        If true. The samples are always redrawn upon request. The stored samples
        are not returned.

    """

    def __init__(self, n_samples_per_trace: int, always_redraw_samples: bool,
                 n_traces: int = 1,
                 n_noise_operators: int = 1,
                 noise_samples: Optional[np.ndarray] = None):
        self._noise_samples = noise_samples
        self._n_samples_per_trace = n_samples_per_trace
        self._n_traces = n_traces
        self._n_noise_operators = n_noise_operators
        self.always_redraw_samples = always_redraw_samples

    @property
    def noise_samples(self) -> np.ndarray:
        if self._noise_samples is None or self.always_redraw_samples:
            self._sample_noise()
        return self._noise_samples

    @property
    def n_samples_per_trace(self) -> int:
        """Number of samples per trace. """
        if self._n_samples_per_trace:
            return self._n_samples_per_trace
        else:
            return self.noise_samples.shape[2]

    @property
    def n_traces(self) -> int:
        """Number of traces. """
        if self._n_traces:
            return self._n_traces
        else:
            return self.noise_samples.shape[1]

    @property
    def n_noise_operators(self) -> int:
        """Number of noise operators. """
        if self._n_noise_operators:
            return self._n_noise_operators
        else:
            return self.noise_samples.shape[0]

    @abstractmethod
    def _sample_noise(self) -> None:
        """Draws the noise samples. """
        pass


class NTGQuasiStatic(NoiseTraceGenerator):
    """
    This class draws noise traces of quasistatic noise.

    The Noise distribution is assumed normal. It would not make sense to use
    the attribute always_redraw_samples if the samples are deterministic,
    and therefore always the same. If multiple noise operators are given, then
    the noise is sampled for each one separately.

    Parameters
    ----------
    standard_deviation: List[float], len (n_noise_operators)
        Standard deviations of the noise assumed on the noise operators.

    n_samples_per_trace: int
        Number of noise samples per trace.

    n_traces: int, optional
        Number of noise traces. Default is 1.

    noise_samples: None or np.ndarray, optional
        shape (n_noise_operators, n_traces, n_samples_per_trace)
        Precalculated noise samples. Defaults to None.

    sampling_mode: {'uncorrelated_deterministic', 'monte_carlo'}, optional
        The method by which the quasi static noise samples are drawn. The
        following are implemented:
        'uncorrelated_deterministic': No correlations are assumed. Each noise
        operator is sampled n_traces times deterministically.
        'monte_carlo': The noise is assumed to be correlated. Samples are drawn
        by pseudo-randomly. Defaults to 'uncorrelated_deterministic'.

    Attributes
    ----------
    standard_deviation: List[float], len (n_noise_operators)
        Standard deviations of the noise assumed on the noise operators.

    See Also
    --------
    noise.NoiseTraceGenerator: Abstract Base Class

    """

    def __init__(self, standard_deviation: List[float],
                 n_samples_per_trace: int,
                 n_traces: int = 1,
                 noise_samples: Optional[np.ndarray] = None,
                 always_redraw_samples: bool = True,
                 sampling_mode: str = 'uncorrelated_deterministic'):
        n_noise_operators = len(standard_deviation)
        super().__init__(noise_samples=noise_samples,
                         n_samples_per_trace=n_samples_per_trace,
                         n_traces=n_traces,
                         n_noise_operators=n_noise_operators,
                         always_redraw_samples=always_redraw_samples)
        self.standard_deviation = standard_deviation
        self.sampling_mode = sampling_mode

    @property
    def n_traces(self) -> int:
        """Number of traces.

        The number of requested traces must be multiplied with the number of
        standard deviations because if standard deviation is sampled
        separately.

        """
        if self._n_traces:
            if self.sampling_mode == 'uncorrelated_deterministic':
                return self._n_traces * len(self.standard_deviation)
            elif self.sampling_mode == 'monte_carlo':
                return self._n_traces
            else:
                raise ValueError('Unsupported sampling mode!')
        else:
            return self.noise_samples.shape[1]

    def _sample_noise(self) -> None:
        """
        Draws quasi static noise samples from a normal distribution.

        Each noise contribution (corresponding to one noise operator) is
        sampled separately. For each standard deviation n_traces traces are
        calculated.

        """
        if self.sampling_mode == 'uncorrelated_deterministic':
            self._noise_samples = np.zeros(
                (len(self.standard_deviation),
                 self._n_traces * len(self.standard_deviation),
                 self.n_samples_per_trace))

            for i, std in enumerate(self.standard_deviation):
                samples = sample_1dim_gaussian_distribution(std, self._n_traces)
                for j in range(self._n_traces):
                    self._noise_samples[i, i * self._n_traces + j, :] \
                        = samples[j] * np.ones(self.n_samples_per_trace)

        elif self.sampling_mode == 'monte_carlo':
            self._noise_samples = np.einsum(
                'i,ijk->ijk',
                np.asarray(self.standard_deviation),
                np.random.randn(len(self.standard_deviation), self.n_traces, 1)
            )
            self._noise_samples = np.repeat(
                self._noise_samples, self.n_samples_per_trace, axis=2)

        else:
            raise ValueError('Unsupported sampling mode!')


class NTGColoredNoise(NoiseTraceGenerator):
    """
    This class draws noise samples from noises of arbitrary colored spectra.

    Parameters
    ----------
    n_samples_per_trace: int
        Number of noise samples per trace.

    n_traces: int, optional
        Number of noise traces. Default is 1.

    n_noise_operators: int, optional
        Number of noise operators. Default is 1.

    always_redraw_samples: bool
        If true. The samples are always redrawn upon request. The stored
        samples are not returned.

    noise_spectral_density: function
        The one-sided noise spectral density as function of frequency.

    dt: float
        Time distance between two adjacent samples.

    low_frequency_extension_ratio: int, optional
        When creating the time samples, the total time is multiplied with this
        factor. This allows taking frequencies into account which are lower
        than 1 / total time. Defaults to 1.

    Attributes
    ----------
    noise_spectral_density: function
        The noise spectral density as function of frequency.

    dt: float
        Time distance between two adjacent samples.

    Methods
    -------
    _sample_noise: None
        Samples noise from an arbitrary colored spectrum.

    See Also
    --------
    noise.NoiseTraceGenerator: Abstract Base Class

    """

    def __init__(self,
                 n_samples_per_trace: int,
                 noise_spectral_density: Callable,
                 dt: float,
                 n_traces: int = 1,
                 n_noise_operators: int = 1,
                 always_redraw_samples: bool = True,
                 low_frequency_extension_ratio: int = 1):
        super().__init__(n_traces=n_traces,
                         n_samples_per_trace=n_samples_per_trace,
                         noise_samples=None,
                         n_noise_operators=n_noise_operators,
                         always_redraw_samples=always_redraw_samples)
        self.noise_spectral_density = noise_spectral_density
        self.dt = dt
        if low_frequency_extension_ratio < 1:
            raise ValueError("The low frequency extension ratio must be "
                             "greater or equal to 1.")
        self.low_frequency_extension_ratio = low_frequency_extension_ratio
        if hasattr(dt, "__len__"):
            raise ValueError('dt is supposed to be a scalar value!')

    def _sample_noise(self, **kwargs) -> None:
        """Samples noise from an arbitrary colored spectrum. """
        if self._n_noise_operators is None:
            raise ValueError('Please specify the number of noise operators!')
        if self._n_traces is None:
            raise ValueError('Please specify the number of noise traces!')
        if self._n_samples_per_trace is None:
            raise ValueError('Please specify the number of noise samples per'
                             'trace!')
        noise_samples = fast_colored_noise(
            spectral_density=self.noise_spectral_density,
            n_samples=
            self.n_samples_per_trace * self.low_frequency_extension_ratio,
            output_shape=(self.n_noise_operators, self.n_traces),
            r_power_of_two=False,
            dt=self.dt)
        self._noise_samples = noise_samples[:, :, :self.n_samples_per_trace]

    def plot_periodogram(self, n_average: int, scaling: str = 'density',
                         log_plot: Optional[str] = None, draw_plot=True):
        """Creates noise samples and plots the corresponding periodogram.

        Parameters
        ----------
        n_average: int
            Number of Periodograms which are averaged.

        scaling: {'density', 'spectrum'}, optional
            If 'density' then the power spectral density in units of V**2/Hz is
            plotted.
            If 'spectral' then the power spectrum in units of V**2 is plotted.
            Defaults to 'density'.

        log_plot: {None, 'semilogy', 'semilogx', 'loglog'}, optional
            If None, then the plot is not plotted logarithmically. If
            'semilogy' only the y-axis is plotted logarithmically, if
            'semilogx' only the x-axis is plotted logarithmically, if 'loglog'
            both axis are plotted logarithmically. Defaults to None.

        draw_plot: bool, optional
            If true, then the periodogram is plotted. Defaults to True.

        Returns
        -------
        deviation_norm: float
            The vector norm of the deviation between the actual power spectral
            density and the power spectral densitry found in the periodogram.

        """

        noise_samples = fast_colored_noise(
            spectral_density=self.noise_spectral_density,
            n_samples=self.n_samples_per_trace,
            output_shape=(n_average,),
            r_power_of_two=False,
            dt=self.dt
        )

        sample_frequencies, spectral_density_or_spectrum = signal.periodogram(
            x=noise_samples,
            fs=1 / self.dt,
            return_onesided=True,
            scaling=scaling,
            axis=-1
        )

        if scaling == 'density':
            y_label = 'Power Spectral Density (V**2/Hz)'
        elif scaling == 'spectrum':
            y_label = 'Power Spectrum (V**2)'
        else:
            raise ValueError('Unexpected scaling argument.')

        if draw_plot:
            plt.figure()

            if log_plot is None:
                plot_function = plt.plot
            elif log_plot == 'semilogy':
                plot_function = plt.semilogy
            elif log_plot == 'semilogx':
                plot_function = plt.semilogx
            elif log_plot == 'loglog':
                plot_function = plt.loglog
            else:
                raise ValueError('Unexpected plotting mode')

            plot_function(sample_frequencies,
                          np.mean(spectral_density_or_spectrum, axis=0),
                          label='Periodogram')
            plot_function(sample_frequencies,
                          self.noise_spectral_density(sample_frequencies),
                          label='Spectral Noise Density')

            plt.ylabel(y_label)
            plt.xlabel('Frequency (Hz)')
            plt.legend(['Periodogram', 'Spectral Noise Density'])
            plt.show()

        deviation_norm = np.linalg.norm(
            np.mean(spectral_density_or_spectrum, axis=0)[1:-1] -
            self.noise_spectral_density(sample_frequencies)[1:-1])
        return deviation_norm
