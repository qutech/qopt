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
Utility functions for the optimal control package.

Functions
---------
:func:`deprecated` decorator
    Marks functions and methods which are deprecated.

:func:`needs_refactoring` decorator
    Marks objects which need refactoring.

:func:`timeit` decorator
    Measures the run time of a function evaluation.

:func:`closest_unitary`
    Calculates the closest unitary matrix to a square matrix.

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

import warnings
import functools
import time
import scipy
import numpy as np


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


def needs_refactoring(func):
    """This is a decorator which can be used to mark functions
    which need to be refactored. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to function {} which needs refactoring.".format(
            func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


def timeit(function):
    """Convenience function to measure the run time of a function.

    This function can be applied as decorator to get a function that evaluates
    the input function an measures the run time.

    Parameters
    ----------
    function: Callable
        The function of which the run time is measured.

    Returns
    -------
    timed: Callable
        Timed function.

    """
    def timed(*args, **kw):
        ts = time.time()
        result = function(*args, **kw)
        te = time.time()
        return result, (te - ts)
    return timed


def closest_unitary(A):
    """ Closest unitary to given square matrix.

    Calculate the unitary matrix U that is closest with respect to the
    operator norm distance to the general matrix A.

    Returns
    -------
    U: np.array
        Closest unitary.

    """
    V, __, Wh = scipy.linalg.svd(A)
    U = np.matrix(V.dot(Wh))
    return U
