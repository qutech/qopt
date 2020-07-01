import unittest
import numpy as np

from qopt.noise import NTGColoredNoise

random_scaling_factor = 50


def white_noise_spectrum(frequencies):
    return random_scaling_factor * np.ones_like(frequencies)


def pink_noise_spectrum(frequencies):
    return random_scaling_factor / frequencies


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
