import numpy as np
import matplotlib.pyplot as plt
import unittest
from typing import Union

from qopt import transfer_function


def exp_saturation(t, t_rise, val_1, val_2):
    """Exponential saturation function."""
    return val_1 + (val_2 - val_1) * (1 - np.exp(-(t / t_rise)))


class ExponentialTFOld(transfer_function.TransferFunction):
    """
    Keept alive for testing.

    bound_type = (code, number): control the number of time slice of padding
    after the original time range.
            code:
                "n": n extra slice of dt/overSampleRate
                "x": n extra slice of dt (default with n=1)
    """

    def __init__(self, x_tau, awg_rise_time, oversampling, num_x, num_ctrls=1,
                 boundary=('x', 0),
                 start_value: Union[int, float, np.ndarray] = 0,
                 stop_value: Union[int, float, np.ndarray] = 0):
        super().__init__()
        self.num_x = num_x
        self.num_ctrls = num_ctrls
        self.xtimes = np.linspace(start=0, stop=x_tau * num_x, num=num_x + 1,
                                  endpoint=True)
        self.u_times = np.linspace(start=0, stop=x_tau * num_x,
                                   num=num_x * oversampling + 1, endpoint=True)
        self.awg_rise_time = awg_rise_time
        self.oversampling = oversampling
        self._T = None
        self.boundary = boundary
        if isinstance(start_value, (float, int)):
            self.start_value = start_value * np.zeros(num_ctrls)
        else:
            self.start_value = start_value
        if isinstance(stop_value, (float, int)):
            self.stop_value = stop_value * np.zeros(num_ctrls)
        else:
            self.stop_value = stop_value

    @property
    def transfer_matrix(self):
        if self._T is None:
            self.make_T()
        return self._T

    def __call__(self, y):
        x_tau = self.xtimes[1] - self.xtimes[0]
        if self.boundary[0] == 'n':
            y = np.zeros((self.num_x * self.oversampling + self.boundary[1],
                          self.num_ctrls))
        elif self.boundary[0] == 'x':
            y = np.zeros(((self.num_x + self.boundary[1]) * self.oversampling,
                          self.num_ctrls))
        else:
            raise ValueError('The boundary type ' + str(self.boundary[0])
                             + ' is not implemented!')
        for k in range(self.num_ctrls):
            for j in range(self.oversampling):
                y[j, k] = exp_saturation((j + 1) / self.oversampling * x_tau,
                                         self.awg_rise_time,
                                         self.start_value[k], y[0, k])
        for k in range(self.num_ctrls):
            for i in range(1, self.num_x):
                for j in range(self.oversampling):
                    y[i * self.oversampling + j, k] = \
                        exp_saturation((j + 1) / self.oversampling * x_tau,
                                       self.awg_rise_time,
                                       y[i - 1, k], y[i, k])
            if self.boundary[0] == 'n':
                for i in range(self.boundary[1]):
                    y[self.num_x * self.oversampling + i] = \
                        exp_saturation((i + 1) / self.oversampling * x_tau,
                                       self.awg_rise_time, y[-1, k],
                                       self.stop_value[k])
            elif self.boundary[0] == 'x':
                for i in range(self.boundary[1]):
                    for j in range(self.oversampling):
                        y[self.num_x * self.oversampling
                          + i * self.oversampling + j] = \
                            exp_saturation(((j + 1) / self.oversampling + i)
                                           * x_tau, self.awg_rise_time,
                                           y[-1, k], self.stop_value[k])

        return y

    def set_x_times(self, x_times, oversampling=None):
        if oversampling is None:
            oversampling = self.oversampling
        self.oversampling = oversampling
        self.xtimes = x_times
        self.num_x = x_times.size - 1
        xtau = x_times[1] - x_times[0]
        times = np.zeros((self.num_x * oversampling + 1,))
        for i in range(self.num_x):
            for j in range(oversampling):
                times[i * oversampling + j] = \
                    x_times[i] + j / oversampling * xtau
        times[-1] = x_times[-1]
        self.u_times = times

    def plot_pulse(self, y):
        """
        Plot the control amplitudes corresponding
        to the given optimisation variables.
        """
        u = self(y)
        dt = self.u_times[1] - self.u_times[0]
        dxt = self.xtimes[1] - self.xtimes[0]

        if self.boundary[0] == 'n':
            times = np.concatenate(
                (self.u_times,
                 dt * np.arange(start=self.u_times.size,
                                stop=self.u_times.size + self.boundary[1])))
        elif self.boundary[0] == 'x':
            times = np.concatenate(
                (self.u_times,
                 dt * np.arange(start=self.u_times.size,
                                stop=self.u_times.size
                                + self.num_x * self.boundary[1])))
        else:
            raise ValueError('Invalid boundary type. Only n and x are '
                             'implemented!')

        for i in range(self.num_ctrls):
            plt.bar(times[:-1] + .5 * dt, u[:, i], dt)
            plt.bar(self.xtimes[:-1] + .5 * dxt, y[:, i], dxt, fill=False)
            plt.show()

    def make_T(self):
        """
            j iterates over the sampling
            i iterates over time slices of the control parameter x
            k iterates over pulses
            in this order
        """
        if self.boundary[0] == 'n':
            dudx = np.zeros((self.num_x * self.oversampling + self.boundary[1],
                             self.num_x, self.num_ctrls))
        elif self.boundary[0] == 'x':
            dudx = np.zeros((
                            (self.num_x + self.boundary[1]) * self.oversampling,
                            self.num_x, self.num_ctrls))
        else:
            raise ValueError('Invalid boundary type. Only n and x are '
                             'implemented!')

        x_tau = self.xtimes[1] - self.xtimes[0]

        # calculate blocks
        exp = np.zeros((self.oversampling,))
        for j in range(self.oversampling):
            t = (j + 1) / self.oversampling * x_tau
            exp[j] = np.exp(-(t / self.awg_rise_time))
        one_minus_exp = np.ones((self.oversampling,)) - exp
        # build 3d gradient matrix
        for k in range(self.num_ctrls):
            dudx[0:self.oversampling:, 0, k] = one_minus_exp
            dudx[self.oversampling:2 * self.oversampling, 0, k] = exp

            for i in range(1, self.num_x - 1):
                dudx[i * self.oversampling:(i + 1) *
                     self.oversampling, i, k] = one_minus_exp

                dudx[(i + 1) * self.oversampling:(i + 2) *
                     self.oversampling, i, k] = exp

            dudx[(self.num_x - 1) * self.oversampling:self.num_x *
                 self.oversampling, self.num_x - 1, k] = one_minus_exp

            if self.boundary[0] == 'n':
                for i in range(self.boundary[1]):
                    t = (i + 1) / self.oversampling * x_tau
                    dudx[self.num_x * self.oversampling + i, -1, k] = np.exp(
                        -(t / self.awg_rise_time))
            elif self.boundary[0] == 'x':
                for i in range(self.boundary[1] * self.oversampling):
                    t = (i + 1) / self.oversampling * x_tau
                    dudx[self.num_x * self.oversampling + i, -1, k] = np.exp(
                        -(t / self.awg_rise_time))

        # cast into 2d form ----- no better keep the 3d version
        # dudx_2d = sc.linalg.block_diag(*[dudx[:, :, k]._T for k in
        # range(self.num_ctrls)])
        self._T = dudx

    def gradient_chain_rule(self, gradient):
        if self._T is None:
            self.make_T()
        return np.einsum('ijk,lik->ljk', self._T, gradient)

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
            oversampling = amplitudes.size // num_ctrls // self.num_x
            num_x = self.num_x
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


class TestTransferFunctions(unittest.TestCase):

    def test_identity(self):
        oversampling = 2
        bound_type = ('n', 1)
        offset = .5
        x_times = np.asarray([1, 2, 3])
        x = np.asarray([[.1], [.2], [.3]])
        df_du = 10 * np.arange(8)
        df_du = np.expand_dims(df_du, axis=1)
        df_du = np.expand_dims(df_du, axis=2)

        identity_tf = transfer_function.OversamplingTF(
            oversampling=oversampling, bound_type=bound_type, offset=offset)

        identity_tf.set_times(x_times)
        u = identity_tf(x)
        df_dx = identity_tf.gradient_chain_rule(df_du)

        u_ideal = np.asarray([0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.5])
        u_ideal = np.expand_dims(u_ideal, axis=1)
        df_dx_ideal = np.asarray([30.,  70., 110.])
        df_dx_ideal = np.expand_dims(df_dx_ideal, axis=1)
        df_dx_ideal = np.expand_dims(df_dx_ideal, axis=2)

        np.testing.assert_array_almost_equal(u, u_ideal)
        np.testing.assert_array_almost_equal(df_dx, df_dx_ideal)

        # second boundary type
        bound_type = ('x', 1)
        df_du = 10 * np.arange(10)
        df_du = np.expand_dims(df_du, axis=1)
        df_du = np.expand_dims(df_du, axis=2)

        identity_tf = transfer_function.OversamplingTF(
            oversampling=oversampling, bound_type=bound_type, offset=offset)

        identity_tf.set_times(x_times)
        u = identity_tf(x)
        df_dx = identity_tf.gradient_chain_rule(df_du)

        u_ideal = np.asarray([0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8, 0.5, 0.5])
        u_ideal = np.expand_dims(u_ideal, axis=1)
        df_dx_ideal = np.asarray([50.,  90., 130.])
        df_dx_ideal = np.expand_dims(df_dx_ideal, axis=1)
        df_dx_ideal = np.expand_dims(df_dx_ideal, axis=2)

        np.testing.assert_array_almost_equal(u, u_ideal)
        np.testing.assert_array_almost_equal(df_dx, df_dx_ideal)

        # no bound type
        bound_type = None

        identity_tf_1 = transfer_function.IdentityTF(num_ctrls=1)
        identity_tf_2 = transfer_function.OversamplingTF(
            oversampling=1, bound_type=bound_type, offset=0)

        identity_tf_1.set_times(x_times)
        identity_tf_2.set_times(x_times)

        u_1 = identity_tf_1(x)
        u_2 = identity_tf_2(x)

        np.testing.assert_array_almost_equal(u_1, u_2)

        df_du = 10 * np.arange(3)
        df_du = np.expand_dims(df_du, axis=1)
        df_du = np.expand_dims(df_du, axis=2)

        grad_1 = identity_tf_1.gradient_chain_rule(df_du)
        grad_2 = identity_tf_2.gradient_chain_rule(df_du)

        np.testing.assert_array_almost_equal(grad_1, grad_2)


    def test_exponential_transfer_function(self):
        num_x = 4
        oversampling = 5
        awg_rise_time = .5
        bound_type = ('x', 1)

        ex = transfer_function.ExponentialTF(
            awg_rise_time=awg_rise_time, oversampling=oversampling,
            bound_type=bound_type, num_ctrls=2)

        x_times = np.ones(num_x)
        ex.set_times(x_times)

        # test application of the transfer function
        x = np.asarray([[1, 3, 2, 4],
                        [1, 3, 2, 4]]).T
        u = ex(x)

        self.assertAlmostEqual(
            u[0 + 5, 0],
            x[0, 0] * (1 - np.exp(-1. / oversampling / awg_rise_time)))
        self.assertAlmostEqual(
            u[1 + 5, 0],
            x[0, 0] * (1 - np.exp(-2. / oversampling / awg_rise_time)))
        self.assertAlmostEqual(
            u[2 + 5, 0],
            x[0, 0] * (1 - np.exp(-3. / oversampling / awg_rise_time)))

        # test reverting the state
        # x_revert = ex.reverse_state(u, ex.times)
        # np.testing.assert_array_almost_equal(x, x_revert)

        # test gradient calculation
        test_dfdu = np.eye(oversampling * (num_x + 2 * bound_type[1]))
        test_dfdu = test_dfdu.reshape(
            oversampling * (num_x + 2 * bound_type[1]),
            oversampling * (num_x + 2 * bound_type[1]), 1)
        grad = ex.gradient_chain_rule(test_dfdu)
        np.testing.assert_array_almost_equal(grad.squeeze(),
                                             ex.transfer_matrix.transpose((1, 0, 2)))

        # for visual test
        # ex.plot_pulse(x[0])
    """
    def test_gaussian_tf(self):
        num_x = 4
        omega = 10
        oversampling = 5

        boundary = ('n', 0)
        gaussian_tf = transfer_function.Gaussian(
            omega=omega, over_sample_rate=oversampling, start=0, end=0,
            bound_type=boundary)

        times = np.arange(0, num_x + 1)
        gaussian_tf.set_times(times)

        # test application of the transfer function
        np.random.seed(1)
        x = np.random.rand(num_x).reshape((num_x, 1))
        u = gaussian_tf(x)

        gaussian_tf._calculate_transfer_matrix()
        gaussian_tf.T.squeeze()

        u_emulated = np.zeros(shape=(num_x * oversampling,))
        for i in range(num_x):
            u_emulated[i * oversampling: (i + 1) * oversampling] = x[i]
        u_emulated = scipy.ndimage.filters.gaussian_filter(
            u_emulated, sigma=.766, mode='constant')
        self.assertLess(np.linalg.norm(u_emulated - u.squeeze()), 1e-3)

        test_dfdu = np.eye(oversampling * num_x)
        test_dfdu = test_dfdu.reshape(
            oversampling * num_x, oversampling * num_x, 1)
        grad = gaussian_tf.gradient_u2x(test_dfdu)
        np.testing.assert_array_almost_equal(grad, gaussian_tf.T)
    """

    def test_concatenation_tf(self):
        num_x = 4
        oversampling = 5
        x_times = np.ones(num_x)
        custom_t = 3 * np.eye(num_x)
        custom_t = np.repeat(custom_t, oversampling, axis=0)

        custom_t = np.repeat(
            np.expand_dims(custom_t, axis=2),
            repeats=1, axis=2)

        custom_tf = transfer_function.CustomTF(custom_t)

        awg_rise_time = .5 / oversampling

        boundary_type = ('n', 0)
        ex = transfer_function.ExponentialTF(
            awg_rise_time=awg_rise_time,
            oversampling=oversampling,
            bound_type=boundary_type)

        concatenated_tf = transfer_function.ConcatenateTF(
            tf1=custom_tf, tf2=ex)

        concatenated_tf.set_times(x_times)

        np.random.seed(1)
        x = np.random.rand(num_x, 1)
        u = concatenated_tf(x)
        self.assertEqual(u.shape, (num_x * oversampling * oversampling, 1))

        test_dfdu = np.ones((oversampling * oversampling * num_x, 1, 1))

        grad = concatenated_tf.gradient_chain_rule(test_dfdu)
        self.assertEqual(grad.shape, (num_x,
                                      1,
                                      1))

    def test_parallize_tf(self):
        num_x = 4
        oversampling = 5
        x_times = np.ones(num_x)

        awg_rise_time = .5 / oversampling

        boundary_type = ('n', 0)
        ex1 = transfer_function.ExponentialTF(
            awg_rise_time=awg_rise_time, oversampling=oversampling,
            bound_type=boundary_type, num_ctrls=1)

        ex2 = transfer_function.ExponentialTF(
            awg_rise_time=awg_rise_time, oversampling=oversampling,
            bound_type=boundary_type, num_ctrls=1)

        ex_2ctrl = transfer_function.ExponentialTF(
            awg_rise_time=awg_rise_time, oversampling=oversampling,
            bound_type=boundary_type, num_ctrls=2)

        parallel_tf = transfer_function.ParallelTF(
            tf1=ex1,
            tf2=ex2
        )

        ex_2ctrl.set_times(x_times)
        parallel_tf.set_times(x_times)

        assert np.all(ex_2ctrl.transfer_matrix == parallel_tf.transfer_matrix)

    def test_efficient_oversampling_tf(self):

        ef_ov_tf = transfer_function.EfficientOversamplingTF(
            oversampling=2,
            bound_type=("n", 2)
        )

        ef_ov_tf.set_times(.5 * np.ones(3))
        np.testing.assert_array_almost_equal(
            ef_ov_tf._x_times,
            .25 * np.ones(10)
        )
        transferred = ef_ov_tf(np.asarray([[1, 2, 3]]).T)
        np.testing.assert_array_almost_equal(
            transferred,
            np.asarray([[0, 0, 1, 1, 2, 2, 3, 3, 0, 0]]).T
        )

    def test_gaussian_convolution(self):
        pass
