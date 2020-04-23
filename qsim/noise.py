"""
This file contains classes and helper functions for the generation of noise
traces.

Classes
-------
NoiseTraceGenerator
    Abstract base class defining the interface of the noise trace generators.

NTGQuasiStatic
    Generates noise traces for quasi static noise.

NTGColoredNoise
    Generates noise traces of arbitrary colored spectra.

Functions
---------
bell_curve_1dim:
    One dimensional bell curve.

sample_1dim_gaussian_distribution:
    Draw samples from the one dimensional bell curve.

bell_curve_2dim:
    Two dimensional bell curve.

sample_2dim_gaussian_distribution:
    Draw samples from the two dimensional bell curve.

fast_colored_noise:
    Samples an arbitrary colored noise spectrum.

"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional, List, Union
from abc import ABC, abstractmethod
from scipy import signal

from qsim.util import deprecated


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
    selected_x: np.ndarray
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


def sample_2dim_gaussian_distribution(std1: float, std2: float, n_samples: int) \
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


@deprecated
def fast_white_noise(spectral_density: Callable, f_max: float, f_min: float,
                     shape: Tuple[int], n_samples=None, r_power_of_two=False):
    """
    Generate fast noise with frequencies between f_min and f_max with the noise
    spectrum like spectral_density.

    Validate like so:
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> traces = fast_white_noise(spectral_density, f_max, f_min)
    >>> f, S = signal.welch(traces, f_max, axis=-1)
    >>> plt.loglog(f, spectral_density(f))
    >>> plt.loglog(f, S.mean(axis=0))
    """
    dt = 1 / f_max
    f_nyquist = f_max / 2
    s0 = 1 / f_nyquist
    if r_power_of_two or n_samples is None:
        actual_n_samples = int(2 ** np.ceil(-np.log2(f_min * dt)))
    else:
        actual_n_samples = n_samples

    delta_white = np.random.randn(*shape, actual_n_samples)
    delta_white_ft = np.fft.rfft(delta_white, axis=-1)
    # Only positive frequencies since FFT is real and therefore symmetric
    f = np.linspace(f_min, f_max / 2, actual_n_samples // 2 + 1)
    f = np.asarray(list(map(spectral_density, f)))
    delta_colored = np.fft.irfft(
        delta_white_ft * np.sqrt(f / s0),
        axis=-1)

    return delta_colored


def fast_colored_noise(spectral_density: Callable, dt: float, n_samples: int,
                       output_shape: Tuple, r_power_of_two=False) -> np.ndarray:
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
        The spectral density as function of frequency.

    dt: float
        Time distance between two samples.

    n_samples: int
        Number of samples.

    output_shape: Tuple[int]
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
    delta_colored = np.fft.irfft(delta_white_ft * np.sqrt(f / s0), axis=-1)
    # the ifft takes r//2 + 1 inputs to generate r outputs

    return delta_colored


@deprecated
def generate_noise_trace(n_lines, spectral_density, f_min, f_max,
                         r_power_of_two=True, n_samples=None):
    delta_t = 1 / f_max
    i = 1 / (f_min * delta_t)
    if r_power_of_two or n_samples is None:
        r = 2
        while r < i:
            r *= 2
    else:
        r = n_samples

    eps = np.random.normal(0, 1, (n_lines, r))
    eps_omega = np.fft.rfft(eps, axis=-1)

    g_f = np.array([np.sqrt(spectral_density(f_min * (j + 1)) / 2 * f_max)
                    for j in range(r // 2 + 1)])

    delta_eps = np.fft.irfft(g_f * eps_omega, axis=-1)
    return np.squeeze(delta_eps)


class NoiseTraceGenerator(ABC):
    """
    Abstract base class defining the interface of the noise trace generators.

    Parameters
    ----------
    n_samples_per_trace: int
        Number of noise samples per trace.

    n_traces: int
        Number of noise traces. Default is 1.

    n_noise_operators: int
        Number of noise operators. Default is 1.

    noise_samples: np.ndarray,
                   shape: (n_noise_operators, n_traces, n_samples_per_trace)
        Precalculated noise samples.

    always_redraw_samples: bool
        If true. The samples are always redrawn upon request. The stored samples
        are not returned.

    Attributes
    ----------
    noise_samples: np.ndarray,
                   shape: (n_noise_operators, n_traces, n_samples_per_trace)
        The noise samples stored or generated by this class.

    n_samples_per_trace: int
        Number of noise samples per trace.

    n_traces: int
        Number of noise traces.

    n_noise_operators: int
        Number of noise operators.

    always_redraw_samples: bool
        If true. The samples are always redrawn upon request. The stored samples
        are not returned.

    Methods
    -------
    _sample_noise: None
        Draw noise samples.

    """

    def __init__(self, n_samples_per_trace: int, always_redraw_samples: int,
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
    standard_deviation: List[float], len: (n_noise_operators)
        Standard deviations of the noise assumed on the noise operators.

    sampling_mode: string
        The method by which the quasi static noise samples are drawn. The
        following are implemented:
        'uncorrelated_deterministic': No correlations are assumed. Each noise
        operator is sampled n_traces times deterministically.
        'monte_carlo': The noise is assumed to be correlated. Samples are drawn
        by pseudo-randomly.

    Attributes
    ----------
    standard_deviation: List[float], len: (n_noise_operators)
        Standard deviations of the noise assumed on the noise operators.

    Methods
    -------
    _sample_noise: None
        Samples quasi static noise from a normal distribution.

    TODO:
        * Draw samples for more than one noise operator
        * Draw noise samples for more than one dimension (at least two)

    """

    def __init__(self, standard_deviation: List[float],
                 n_samples_per_trace: int, n_traces: int,
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
        separately. """
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

        If an amplitude function is available, it is applied to the noise
        samples.

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
    noise_spectral_density: function
        The noise spectral density as function of frequency.

    dt: float
        Time distance between two adjacent samples.

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

    """

    def __init__(self, noise_spectral_density: Callable, dt: float,
                 n_samples_per_trace: int, n_traces: int,
                 n_noise_operators: int, always_redraw_samples: bool = True):
        super().__init__(n_traces=n_traces,
                         n_samples_per_trace=n_samples_per_trace,
                         noise_samples=None,
                         n_noise_operators=n_noise_operators,
                         always_redraw_samples=always_redraw_samples)
        self.noise_spectral_density = noise_spectral_density
        self.dt = dt
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
            n_samples=self.n_samples_per_trace,
            output_shape=(self.n_noise_operators, self.n_traces),
            r_power_of_two=False,
            dt=self.dt)
        self._noise_samples = noise_samples

    def plot_periodogram(self, n_average: int, scaling: str = 'density',
                         log_plot: Optional[str] = None, draw_plot=True):

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
