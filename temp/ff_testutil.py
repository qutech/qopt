# =============================================================================
#     filter_functions
#     Copyright (C) 2019 Quantum Technology Group, RWTH Aachen University
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
#     Contact email: tobias.hangleiter@rwth-aachen.de
# =============================================================================
"""
This module defines some testing utilities.
"""

import unittest
import os
from pathlib import Path

import numpy as np
import qutip as qt
from numpy.random import randn
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io import loadmat
from scipy.linalg import expm

from filter_functions import util


class TestCase(unittest.TestCase):

    def assertCorrectDiagonalization(self, x, rtol=1e-7, atol=0,
                                     equal_nan=True, err_msg='', verbose=True):
        """
        Assert eigenvalues and eigenvectors of PulseSequence fulfill
        characteristic equation.
        """
        try:
            H = np.einsum('ijk,il->ljk', x.c_opers, x.c_coeffs)
        except AttributeError:
            raise ValueError('Can only work with PulseSequences')

        for i, H in enumerate(H):
            self.assertArrayAlmostEqual(x._HV[i].conj().T @ H @ x._HV[i],
                                        np.diag(x._HD[i]), err_msg=err_msg,
                                        atol=atol, rtol=rtol, verbose=verbose,
                                        equal_nan=equal_nan)

    def assertArrayEqual(self, x, y, err_msg='', verbose=True):
        """
        Wraps numpy.testing.assert_array_equal
        """
        assert_array_equal(x, y, err_msg, verbose)

    def assertArrayAlmostEqual(self, actual, desired, rtol=1e-7, atol=0,
                               equal_nan=True, err_msg='', verbose=True):
        """
        Wraps numpy.testing.assert_allclose
        """
        # Catch trying to compare to Nones
        if actual is not None and desired is not None:
            assert_allclose(actual, desired, rtol, atol, equal_nan, err_msg,
                            verbose)
        elif (actual is None and desired is not None or
              actual is not None and desired is None):
            raise AssertionError('One of {} or {} '.format(actual, desired) +
                                 'is None but the other not!')
        else:
            assert_array_equal(actual, desired, err_msg, verbose)


def generate_dd_hamiltonian(n, tau=10, tau_pi=1e-2, dd_type='cpmg',
                            pulse_type='primitive'):
    """
    Generate a Hamiltonian in the correct format as required by PulseSequence
    for a dynamical decoupling sequence of duration *tau* and order *n*.
    *pulse_type* toggles between a primitive NOT-pulse and a dynamically
    corrected gate.
    """
    if dd_type == 'cpmg':
        delta = np.array([0] + [(l - 0.5)/n for l in range(1, n+1)])
    elif dd_type == 'udd':
        delta = np.array(
            [0] + [np.sin(np.pi*l/(2*n + 2))**2 for l in range(1, n+1)]
        )

    if pulse_type == 'primitive':
        tau_p = tau_pi
        s_p = np.pi/tau_pi*np.array([0, 1])
        t_p = tau_pi*np.array([0, 1])
    elif pulse_type == 'dcg':
        tau_p = 4*tau_pi
        s_p = np.pi/tau_pi*np.array([0, 1, 0.5, 1])
        t_p = np.array([0, tau_pi, 2*tau_pi, tau_pi]).cumsum()

    s = np.array([])
    t = np.array([0])
    for i in range(n):
        s = np.append(s, s_p)
        t = np.append(t, t_p + (delta*tau)[i+1] - tau_p/2)
    t = np.append(t, tau)
    s = np.append(s, 0)

    H = [[qt.sigmax()/2, s]]
    return H, np.diff(t)


def rand_herm(d: int, n: int = 1) -> np.ndarray:
    """n random Hermitian matrices of dimension d"""
    A = randn(n, d, d) + 1j*randn(n, d, d)
    return (A + A.conj().transpose([0, 2, 1])).squeeze()


def rand_unit(d: int, n: int = 1) -> np.ndarray:
    """n random unitary matrices of dimension d"""
    H = rand_herm(d, n)
    if n == 1:
        return expm(1j*H)
    else:
        return np.array([expm(1j*h) for h in H])


# Set up Hamiltonian for CNOT gate
data_path = r"C:\Users\Inst2C\Documents\python-filter_functions\examples\data"
struct = loadmat(os.path.join(data_path, 'CNOT.mat'))
eps = np.asarray(struct['eps'], order='C')
dt = np.asarray(struct['t'].ravel(), order='C')
B = np.asarray(struct['B'].ravel(), order='C')
B_avg = struct['BAvg'].ravel()
cnot_infid_fast = struct['infid_fast'].ravel()

J = np.exp(eps)
n_dt = len(dt)

d = 16
H = np.empty((6, d, d), dtype=float)
Id, Px, Py, Pz = util.P_np
# Exchange Hamiltonians
H[0] = 1/4*sum(util.tensor(P, P, Id, Id) for P in (Px, Py, Pz)).real
H[1] = 1/4*sum(util.tensor(Id, P, P, Id) for P in (Px, Py, Pz)).real
H[2] = 1/4*sum(util.tensor(Id, Id, P, P) for P in (Px, Py, Pz)).real
# Zeeman Hamiltonians
H[3] = 1/8*(util.tensor(Pz, Id, Id, Id)*(-3) +
            util.tensor(Id, Pz, Id, Id) +
            util.tensor(Id, Id, Pz, Id) +
            util.tensor(Id, Id, Id, Pz)).real
H[4] = 1/4*(util.tensor(Pz, Id, Id, Id)*(-1) +
            util.tensor(Id, Pz, Id, Id)*(-1) +
            util.tensor(Id, Id, Pz, Id) +
            util.tensor(Id, Id, Id, Pz)).real
H[5] = 1/8*(util.tensor(Pz, Id, Id, Id)*(-1) +
            util.tensor(Id, Pz, Id, Id)*(-1) +
            util.tensor(Id, Id, Pz, Id)*(-1) +
            util.tensor(Id, Id, Id, Pz)*3).real
# Mean Magnetic field
H0 = B_avg/2*sum(util.tensor(*np.roll((Pz, Id, Id, Id), shift=i, axis=0))
                 for i in range(4)).real

opers = [*list(H), H0]
# Reduce to 6x6 subspace
subspace = ([3, 5, 6, 9, 10, 12], [3, 5, 6, 9, 10, 12])
d_subspace = 6
# H0 is zero on the subspace
subspace_opers = [H[np.ix_(*subspace)] for H in H]

# Subtract identity to make Hamiltonian traceless
subspace_opers = [oper - np.trace(oper)/d_subspace*np.eye(d_subspace)
                  for oper in subspace_opers]

c_coeffs = [J[0],
            J[1],
            J[2],
            B[0]*np.ones(n_dt),
            B[1]*np.ones(n_dt),
            B[2]*np.ones(n_dt)]

n_coeffs = [J[0],
            J[1],
            J[2],
            np.ones(n_dt),
            np.ones(n_dt),
            np.ones(n_dt)]

# %% Noise spectrum from Dial et al (1/f^(0, 0.7))

eps0 = 2.7241e-4
# S(f) = A f^{-\alpha}
alpha = np.array([0, 0.7])
# At f = 1 MHz = 1e-3 GHz, S = S_0 = 4e-20 V^2/Hz = 4e-11 1/GHz
# Correspondingly, S(\omega) = A \omega^{-\alpha} such that at
# \omega = 2\pi 10^{-3} GHz, S = S_0 = 4e-11 1/GHz
S0 = 4e-11/eps0**2
A = S0*(2*np.pi*1e-3)**alpha
# Get S(\omega) like so:
# S = [A/omega**alpha for A, alpha in zip(A, alpha)]
