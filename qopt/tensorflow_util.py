import tensorflow as tf
import numpy as np
import qopt
from typing import Union


DEFAULT_COMPLEX_TYPE = tf.complex128
DEFAULT_FLOAT_TYPE = tf.complex128


# @tf.function  # this cannot be a tf.function because the inputs are not
# Tensors
def convert_to_constant_tensor(
        x: Union[np.array, qopt.DenseOperator, tf.Tensor]):
    """
    Creates a constant Tensor from other formats.

    Parameters
    ----------
    x

    Returns
    -------

    """
    if type(x) == np.array:
        output = tf.constant(x, dtype=DEFAULT_COMPLEX_TYPE)
    elif type(x) == qopt.DenseOperator:
        output = tf.constant(x.data, dtype=DEFAULT_COMPLEX_TYPE)
    else:
        output = tf.constant(x, dtype=DEFAULT_COMPLEX_TYPE)
    return output
