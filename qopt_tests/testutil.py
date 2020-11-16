""" Utility functions for the unit tests. """

import numpy as np
from typing import Callable


def calculate_jacobian(
        func: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        delta_x: float):
    x_shape = x0.shape
    num_par = x0.size
    y0 = func(x0)
    num_func = y0.size

    y = np.empty((num_par, num_func, 3), np.float64)

    for i in range(num_par):
        dif_x = np.zeros_like(x0, dtype=np.float64).flatten()
        dif_x[i] = delta_x
        dif_x = dif_x.reshape(x_shape)
        y[i, :, 1] = y0
        y[i, :, 0] = func(x0 - dif_x)
        y[i, :, 2] = func(x0 + dif_x)

    jacobian = np.gradient(y, delta_x, axis=2)
    return jacobian[:, :, 1]
