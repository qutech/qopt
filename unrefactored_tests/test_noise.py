from qutip.control_2.noise import fast_colored_noise
from qutip.control_2.noise import generate_noise_trace
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor

import numpy as np
from numpy import fft, ndarray
from numpy.random import randn

def Yoneda_one_over_f_spectral_noise_density(f):
    S = 3e6 / f
    return S


def white_noise_density(f):
    return np.ones_like(f) * 1e-5


def dial_et_al_spectral_noise_density(f):
    if 50e3 <= f < 1e6:
        return 8e-16 * f ** -.7
    elif 1e6 <= f <= 3e9:
        return 8e-16 * 1e6 ** -.7
    else:
        return 0


def dial_vectorized(f):
    if type(f) == float:
        return dial_et_al_spectral_noise_density(f)
    else:
        return np.asarray(list(map(dial_et_al_spectral_noise_density, f)))


def fast_noise_correcto(exponent: float, A: float, f_max: float, f_min: float,
                        f_A: float, T: float, dt: float, n_dt: float,
                        n_MC: float,
                        n_ops: float) -> ndarray:
    """f_A: frequency at which S(f) = A"""

    fast_multiple = ceil(dt*f_max)
    slow_multiple = ceil(1/(f_min*T)) if exponent != 0 else 1
    # fast and slow multiples are are the relation between the requested
    # frequency range and the maximally possible. In the test cases, they are
    # one.
    # fast_multiple > 1: The requested f_max is larger then the possible
    # resolution 1 / dt
    # slow_multiple > 1: the requested f_min is smaller then the aliasing
    # 1 / T.
    # slow_multiple and fast_multiple should always be smaller or equal to 1 for
    # correctly chosen frequency ranges. The ceil then assures they are always 1
    dt_fast = dt/fast_multiple
    S0 = 2*dt_fast
    add_one = True if (n_dt*fast_multiple*slow_multiple) % 2 else False
    R = n_dt*fast_multiple*slow_multiple + add_one
    # this ensures that R is a multiple of 2

    delta_white = randn(n_MC, n_ops, R)*np.sqrt(A/S0)
    # The standardization np.sqrt(A/S0) is the square root of the reference
    # value divided by 2 times the number of time steps

    if exponent != 0:
        f = np.concatenate((1/dt_fast*np.r_[0:(R/2)+1]/R,
                            1/dt_fast*np.r_[(R/2)-1:0:-1]/R))
        # np.r_ is here equivalent to np.arange
        # f is given by ascending and descending values
        psd = np.sqrt(f**exponent / f_A**exponent)
        # the power spectral density is normalized by the reference frequency
        # and also the const. factor of the psd
        psd[0] = 1
    else:
        psd = 1

    r = int(R / 2 - 1)
    # r is always R//1 - 1 because R is always even
    delta_colored = fft.ifft(
        # in contrast to the own methods, here a complex fft is used.
        fft.fft(delta_white, axis=-1) *
        # the FFT is a linear function. Hence we can pull out the factor
        # sqrt(A/S0)
        np.concatenate((np.zeros((n_MC, n_ops, 1)),
                        np.ones((n_MC, n_ops, r)),
                        # J. Teske replaced np.zeros((n_MC, n_ops, R//2-r))
                        # by the following line for better last value in the
                        # periodogram
                        np.ones((n_MC, n_ops, R//2-r)),
                        #                      \_____/
                        #                         1
                        np.zeros((n_MC, n_ops, R//2-r-1)),
                        #                      \_______/
                        #                           0
                        np.ones((n_MC, n_ops, r))), axis=-1) *
        # The concatenated Object hast dimensions (n_MC, n_ops, R) in total
        psd,
        # The (total) normalization factor is given by
        # sqrt(A/ f_A**exponent / S0) = sqrt(A/ f_A**exponent / 2 / dt)
        #                                    \______________/   \____/
        #                                       psd factor    *   f_N
        # f_N is the Nyquist frequency of the time spacing
        # the psd factor cancels out because we did not take the real psd
        # function; only the exponential.
        # Actually the PSD is only normalized by f_N
        axis=-1
    )

    if add_one:
        delta_colored = delta_colored[..., :-1]

    if exponent != 0:
        delta_colored = delta_colored[
            ..., :R // slow_multiple + floor(R // 2)
            - floor(R // slow_multiple // 2)]
        # = delta_colored[..., :R // 2]
        # here we neglect the negative frequencies

    return delta_colored


dt = 100e-9 / 8
n_time_steps = 106
spectral_density = Yoneda_one_over_f_spectral_noise_density
# spectral_density = white_noise_density
f_max = 1 / dt
f_min = 1 / dt / n_time_steps
n_lines = 50000
traces = fast_colored_noise(spectral_density, dt=dt, n_samples=n_time_steps,
                            output_shape=(n_lines,), r_power_of_two=False)

correct_traces = fast_noise_correcto(exponent=-1, A=spectral_density(f_max),
                                     f_max=f_max, f_min=f_min, f_A=f_max,
                                     T=dt * n_time_steps, dt=dt,
                                     n_dt=n_time_steps, n_MC=n_lines, n_ops=1)

f_correct, S_correct = signal.periodogram(np.squeeze(np.real(correct_traces)),
                                          f_max, axis=-1)

# plt.loglog(f_correct[1:], (np.squeeze(S_correct).mean(axis=0))[1:])

f1, S = signal.periodogram(traces, f_max, axis=-1)
fsd = spectral_density(f1)
plt.loglog(f1[1:], fsd[1:])
plt.loglog(f1[1:], (S.mean(axis=0))[1:])
"""
fsd2 = generate_noise_trace(n_lines,
                            spectral_density=spectral_density,
                            f_min=f_min, f_max=f_max, r_power_of_two=False,
                            n_samples=106)

f2, S2 = signal.periodogram(fsd2, f_max, axis=-1)
plt.loglog(f2[1:], (S2.mean(axis=0))[1:])
"""





