# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:18:02 2019

@author: Tobias Hangleiter, tobias.hangleiter@rwth-aachen.de

changes by Julian Teske, julian.teske@rwth-aachen.de
"""
from math import ceil
from time import perf_counter
from typing import Callable, Tuple

import qutil
import filter_functions as ff
import numpy as np
from numpy import ndarray
from numpy.random import randn
from scipy.linalg import expm

USE_WHITE_NOISE = False


def MC_ent_fidelity(gates, target):
    """Calculate the entanglement fidelity"""
    d = gates.shape[-1]
    return qutil.linalg.abs2(np.einsum('...ll', gates @ target.conj().transfer_matrix) / d)


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


def white_noise(S0: float, f_max: float, shape: Tuple[int]) -> ndarray:
    """Generate white noise with variance S0*f_max/2."""
    var = S0*f_max/2
    return np.sqrt(var)*randn(*shape)


def Yoneda_one_over_f_spectral_noise_density(f):
    S = 3e6 / f
    return S


def monte_carlo_gate(opers: ndarray, coeffs: ndarray,
                     N_MC: int, S0: float, f_max: float, dt: ndarray):
    """
    Return N_MC gates with ceil(dt.min()*f_max) noise steps per time step of
    the gate.
    """
    N_n = ceil(dt.min()*f_max)
    if N_n*dt.size % 2:
        N_n += 1

    dt = np.repeat(dt, N_n)/N_n
    coeffs = np.repeat(coeffs, N_n, axis=1)

    if USE_WHITE_NOISE:
        coeffs_delta = (
            coeffs.reshape(1, *coeffs.shape) +
            white_noise(S0, f_max, (N_MC, dt.size)).reshape(N_MC, 1, dt.size)
        )
    else:
        omega = np.geomspace(1/T, f_max*2*np.pi, n_omega)
        noise = Yoneda_one_over_f_spectral_noise_density(omega)
        coeffs_delta = (
                coeffs.reshape(1, *coeffs.shape) +
                noise.reshape(N_MC, 1, dt.size)
        )

    H = np.einsum('ijk,mil->mljk', opers, coeffs_delta)
    HD, HV = np.linalg.eigh(H)
    P = np.einsum('mlij,mjl,mlkj->mlik', HV,
                  np.exp(-1j*np.asarray(dt)*HD.swapaxes(-1, -2)), HV.conj())
    Q = qutil.linalg.mdot(P[:, ::-1, ...], axis=1)
    return Q


# %%
d_max = 120
dims = np.arange(2, 20+1)
n_alpha = 3
n_dt = 1
n_MC = 100
n_omega = 500

tic_MC = []
toc_MC = []
tic_FF = []
toc_FF = []
Fid_FF = []
Fid_MC = []
for d in dims:
    print(f'd = {d}')
    opers = rand_herm(d, n_alpha)
    coeffs = randn(n_alpha, n_dt)
    dt = np.abs(randn())*np.ones(n_dt)
    T = dt.sum()
    f_max = 1e2/T
    S0 = abs(randn())/1e4

    pulse = ff.PulseSequence(list(zip(opers, coeffs)),
                             list(zip(opers, np.ones_like(coeffs))),
                             dt)
    omega = np.geomspace(1/T, f_max*2*np.pi, n_omega // 2)
    S = S0*np.ones_like(omega)
    S, omega = ff.util.symmetrize_spectrum(S, omega)

    print('FF')
    tic_FF.append(perf_counter())
    FF_fidelity = 1 - ff.infidelity(pulse, S, omega).sum()
    toc_FF.append(perf_counter())
    Fid_FF.append(FF_fidelity)

    print('MC')
    tic_MC.append(perf_counter())
    MC_propagators = monte_carlo_gate(opers, coeffs, n_MC, S0, f_max, dt)
    MC_fidelity = MC_ent_fidelity(MC_propagators, pulse.total_Q).mean()
    toc_MC.append(perf_counter())
    Fid_MC.append(MC_fidelity)

import matplotlib.pyplot as plt
plt.plot(1 - np.asarray(Fid_FF))
plt.plot(1 - np.asarray(Fid_MC))
plt.figure()
plt.plot(np.abs(2 * ((1 - np.asarray(Fid_FF)) - (1 - np.asarray(Fid_MC))) / (
            (1 - np.asarray(Fid_FF)) + (1 - np.asarray(Fid_MC)))))
