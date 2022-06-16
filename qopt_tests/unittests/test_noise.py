"""
Test fast noise generators vs their periodogram.

Test quasi static samples for standard deviation and mean.
"""
import unittest
import numpy as np
from scipy import special, integrate


from qopt.noise import NTGColoredNoise, sample_1dim_gaussian_distribution, \
    NTGQuasiStatic, inverse_cumulative_gaussian_distribution_function

random_scaling_factor = 50


def white_noise_spectrum(frequencies):
    return random_scaling_factor * np.ones_like(frequencies)


def pink_noise_spectrum(frequencies):
    return random_scaling_factor / frequencies


def gaussian(x, std, mean):
    return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(
        -.5 * (x - mean) ** 2 / std ** 2)


class TestNoiseSampling(unittest.TestCase):
    n_average = 10000

    def test_white_noise_sampling(self):

        ntg = NTGColoredNoise(
            noise_spectral_density=white_noise_spectrum,
            n_samples_per_trace=2000, n_traces=2, n_noise_operators=1,
            always_redraw_samples=True, dt=1e-4
        )

        deviation_norm = ntg.plot_periodogram(
            n_average=self.n_average, scaling='density', draw_plot=False)
        self.assertLess(deviation_norm / self.n_average, 1e-2)

        np.random.randn()

    def test_pink_noise_sampling(self):
        ntg = NTGColoredNoise(
            noise_spectral_density=pink_noise_spectrum,
            n_samples_per_trace=2000, n_traces=1, n_noise_operators=1,
            always_redraw_samples=True, dt=1e-4
        )

        deviation_norm = ntg.plot_periodogram(
            n_average=self.n_average, scaling='density', log_plot='loglog',
            draw_plot=False)
        self.assertLess(deviation_norm / self.n_average, 1e-3)

    def test_inverse_cumulative_gaussian_dist(self):
        def gaussian(x, std, mean):
            return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(
                -.5 * (x - mean) ** 2 / std ** 2)

        for std in [1, 2.3, 5]:
            for mean in [-4, .03, 33]:
                for x_lim in [.2, .4, .66]:
                    x_lim = x_lim * std + mean
                    integral, error = integrate.quad(
                        lambda x: gaussian(x, std=std, mean=mean), -np.inf,
                        x_lim)
                    inverse = inverse_cumulative_gaussian_distribution_function(
                        integral, std=std, mean=mean)
                    # print('std: ' + str(std))
                    # print('mean: ' + str(mean))
                    assert (x_lim - inverse) < 1e-5 * error

    def test_gaussian_sampling(self):

        for std in [1, 2.3, 5]:
            for mean in [-4, .03, 33]:
                for n_samples in [5, 10, 200]:
                    samples = sample_1dim_gaussian_distribution(
                        std=std, mean=mean, n_samples=n_samples)

                    # apply the inverse cumulative gaussian dist
                    for i in range(1, n_samples):
                        integral, error = integrate.quad(
                            lambda x: gaussian(x, std=std, mean=mean),
                            samples[i-1],
                            samples[i])
                        assert (integral - 1 / n_samples) < 1e-6 * std

    def test_std_rescaling_quasi_static(self):
        for std in [1, 2.3, 5]:
            for n_samples in [5, 10, 200]:
                ntg = NTGQuasiStatic(standard_deviation=[std, ],
                                     n_samples_per_trace=n_samples,
                                     correct_std_for_discrete_sampling=True)
                samples = ntg.noise_samples
                assert np.std(samples) - std < 1e-10
