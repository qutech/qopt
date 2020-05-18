# -*- coding: utf-8 -*-
# =============================================================================
#     filter_functions
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
This file contains a matrix class which implements the computationally
expensive numeric calculations.

The matrices can be stored and handled either as dense or sparse
matrices for the sake of encapsulation of the numeric methods.
The most frequently used and computationally expensive function is
the matrix exponential and its derivative. These operations are required to
calculate the analytic solution of the Schroedinger and Lindblad master
equation.


Classes
-------
:class:`OperatorMatrix`
    Abstract base class.

:class:`OperatorDense`
    Dense control matrices, which are based on numpy arrays.

:class:`OperatorSparse`
    To be implemented


Functions
---------
:func:`convert_unitary_to_super_operator`
    Converts a unitary propagator into a super operator in the lindblad
    formalism.

:func:`closest_unitary`
    Calculates the closest unitary propagator for a squared matrix.

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

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.linalg as la

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, Sequence

from qutip import Qobj


VALID_SCALARS = [int, float, complex, np.int8, np.int16, np.int32, np.int64,
                 np.float16, np.float32, np.float64, np.complex64,
                 np.complex128]
# These types are recognised as scalars in the scalar multiplication with
# matrices.


class OperatorMatrix(ABC):
    """
    The abstract base class of the operator matrix for the qsim control
    package.

    It offers an identical interface to use sparse and dense matrices and has
    hence the limitations of both representations in terms of usability.

    Attributes
    ----------
    data:
        The stored data. Its type is defined in subclasses.

    Methods
    -------
    dag:
        Adjoint (dagger) of matrix. Also called hermitian conjugate.

    tr:
        Trace of matrix.

    prop(tau):
        The matrix exponential exp(A*tau) of the matrix A.

    dexp(dirr, tau, compute_expm=False)
        The deriative of the exponential in the given dirrection

    clean:
        Delete stored data.

    copy:
        Returns a deep copy of the object.

    shape:
        Returns the shape of the matrix.

    conj(copy_):
        Complex conjugate of the matrix. Returns a copy or operates in place.

    conjugate(copy_):
        Alias for conj.

    transpose(copy_):
        Transpose the matrix.

    flatten:
        Returns the flattened matrix.

    prop(tau):
        Convenience function to calculate the propagator of the Schroedinger
        equation.

    exp(tau, method):
        Returns the matrix exponential.

    dexp(direction, tau, method):
        Returns the matrix exponential and its frechet derivative.

    kron(other):
        Kronecker matrix product.

    identity_like: ControlMatrix
        Returns an identity matrix of the same shape and type. Only for square
        matrices!

    spectral_decomposition(hermitian):
        Eigenvalues and eigenvectors.

    __getitem__(identifier):
        Returns the matrix element corresponding to the identifier.

    __add__(other):
        Addition with other instances of the same type or instances of the
        data storage type.

    __iadd__(other):
        In place addition with other instances of the same type or instances of
        the data storage type.

    __mul__(other):
        Either Matrix multiplication or scalar multiplication.

    __imul__(other):
        Inplace matrix or scalar multiplication.

    __rmul__(other):
        Reflective scalar or matrix multiplication.

TODO:
    * implement element wise division for scalars
    """

    def __init__(self) -> None:
        self.data = None
        self._size = 0

        self._factormatrix = None
        self._prop_eigen = None
        self._eig_vec = None
        self._eig_vec_dag = None
        self._prop = None

    @abstractmethod
    def copy(self):
        """Return a deep copy of the control matrix. """
        pass

    def clean(self):
        """Delete stored data. """
        self._factormatrix = None
        self._prop_eigen = None
        self._eig_vec = None
        self._eig_vec_dag = None
        self._prop = None

    def __add__(self, other: 'OperatorMatrix') -> 'OperatorMatrix':
        """Overloaded addition.

        Add Matrix of the same dimension or scalar value to each element.

        Parameters
        ----------
        other: ControlMatrix or numpy array or scalar
            If other is a scalar value (int, float, complex, np.complex128)
            then the value is added to each matrix element.

        Returns
        -------
        out:
            New instance of the same type containing the result of the
            addition.

        Raises
        ------
        ValueError:
            If the operation is not defined for the input type.

        """
        out = self.copy()
        out += other
        return out

    @abstractmethod
    def __iadd__(self, other: 'OperatorMatrix') -> 'OperatorMatrix':
        """Overloaded in place addition.

        Add Matrix of the same dimension or scalar value to each element.

        Parameters
        ----------
        other: ControlMatrix or numpy array or scalar
            If other is a scalar value (int, float, complex, np.complex128)
            then the value is added to each matrix element.

        Returns
        -------
        self:
            The matrix itself is returned as the operation is executed in
            place.

        Raises
        ------
        ValueError:
            If the operation is not defined for the input type.

        """
        pass

    @abstractmethod
    def __mul__(self, other: Union['OperatorMatrix', complex, float, int,
                                   np.generic]) -> 'OperatorMatrix':
        """Overloaded multiplication.

        Matrix multiplication with another matrix or scalar multiplication with
        a scalar value.

        Parameters
        ----------
        other: ControlMatrix or numpy array or scalar
            If other is a scalar value (int, float, complex, np.complex128)
            then each matrix element is multiplied with the scalar value.
            Otherwise the matrix product is applied.

        Returns
        -------
        self:
            The matrix itself is returned as the operation is executed in
            place.

        Raises
        ------
        ValueError:
            If the operation is not defined for the input type.

        """
        pass

    @abstractmethod
    def __imul__(self, other: Union['OperatorMatrix', complex, float, int,
                                    np.generic]) -> 'OperatorMatrix':
        """Overloaded in place multiplication.

        Matrix multiplication with another matrix or scalar multiplication with
        a scalar value in place.

        Parameters
        ----------
        other: ControlMatrix or numpy array or scalar
            If other is a scalar value (int, float, complex, np.complex128)
            then each matrix element is multiplied with the scalar value.
            Otherwise the matrix product is applied.

        Returns
        -------
        out:
            New instance of the same type containing the result of the
            multiplication.

        Raises
        ------
        ValueError:
            If the operation is not defined for the input type.

        """
        pass

    @abstractmethod
    def __rmul__(self, other: Union['OperatorMatrix', complex, float, int,
                                    np.generic]) -> 'OperatorMatrix':
        """Overloaded reflected multiplication.

        Matrix multiplication with another matrix or scalar multiplication with
        a scalar value.

        Parameters
        ----------
        other: ControlMatrix or numpy array or scalar
            If other is a scalar value (int, float, complex, np.complex128)
            then each matrix element is multiplied with the scalar value.
            Otherwise the matrix product is applied.

        Returns
        -------
        out:
            New instance of the same type containing the result of the
            multiplication.

        Raises
        ------
        ValueError:
            If the operation is not defined for the input type.

        """
        pass

    def __sub__(self, other: 'OperatorMatrix') -> 'OperatorMatrix':
        """Overloaded subtraction.

        Subtract Matrix of the same dimension or scalar value from each
        element.

        Parameters
        ----------
        other: ControlMatrix or numpy array or scalar
            If other is a scalar value (int, float, complex, np.complex128)
            then the value is added to each matrix element.

        Returns
        -------
        out:
            New instance of the same type containing the result of the
            addition.

        Raises
        ------
        ValueError:
            If the operation is not defined for the input type.

        """
        out = self.copy()
        out -= other
        return out

    @abstractmethod
    def __isub__(self, other: 'OperatorMatrix') -> 'OperatorMatrix':
        """Overloaded in place subtraction.

        Subtract Matrix of the same dimension or scalar value from each
        element.

        Parameters
        ----------
        other: ControlMatrix or numpy array or scalar
            If other is a scalar value (int, float, complex, np.complex128)
            then the value is added to each matrix element.

        Returns
        -------
        out:
            New instance of the same type containing the result of the
            addition.

        Raises
        ------
        ValueError:
            If the operation is not defined for the input type.

        """
        pass

    @property
    def shape(self) -> Tuple:
        """Returns the shape of the matrix. """
        return self.data.shape

    @abstractmethod
    def __getitem__(self, index: Tuple) -> complex:
        """Returns the corresponding matrix element.

        Parameters
        ----------
        index: tuple of int, length: 2
            Index describing an entry in the marix.

        Returns
        -------
        value: complex
            Matrix element at the position described by the index.

        """
        pass

    @abstractmethod
    def dag(self, copy_: bool = True) -> Optional['OperatorMatrix']:
        """
        Adjoint (dagger) of the matrix.

        Parameters
        ----------
        copy_: bool, optional
            If false, then the operation is executed inplace. Otherwise returns
            a new instance. Defaults to True.

        Returns
        -------
        out: OperatorMatrix
            If copy_ is true, then a new instance otherwise self.

        """
        return self

    @abstractmethod
    def tr(self) -> complex:
        """Trace of the matrix.

        Returns
        -------
        trace: float
            Trace of the matrix.

        """
        return 0j

    @abstractmethod
    def conj(self, copy_: bool = True) -> Optional['OperatorMatrix']:
        """
        Complex conjugate of the matrix.

        Parameters
        ----------
        copy_: bool, optional
            If false, then the operation is executed inplace. Otherwise returns
            a new instance. Defaults to True.

        Returns
        -------
        out: OperatorMatrix
            If copy_ is true, then a new instance otherwise self.

        """
        pass

    def conjugate(self, copy_: bool = True) -> Optional['OperatorMatrix']:
        """Alias for conj. """
        return self.conj(copy_=copy_)

    @abstractmethod
    def transpose(self, copy_: bool = True) -> Optional['OperatorMatrix']:
        """Transpose of the matrix.

        Parameters
        ----------
        copy_: bool, optional
            If false, then the operation is executed inplace. Otherwise returns
            a new instance. Defaults to True.

        Returns
        -------
        out: OperatorMatrix
            If copy_ is true, then a new instance otherwise self.

        """

    @abstractmethod
    def kron(self, other: 'OperatorMatrix') -> 'OperatorMatrix':
        """
        Computes the kronecker matrix product with another matrix.

        Parameters
        ----------
        other: OperatorMatrix or np.ndarray
            Second factor of the kronecker product.

        Returns
        -------
        out: OperatorMatrix
            Operator matrix of the same type containing the product.

        Raises
        ------
        ValueError:
            If the operation is not defined for the input type.

        """
        pass

    @abstractmethod
    def flatten(self) -> np.ndarray:
        """
        Flattens the matrix.

        Returns
        -------
        out: np.array
            The flattened control matrix as one dimensional numpy array.

        """
        pass

    @abstractmethod
    def spectral_decomposition(self, hermitian: bool = False):
        """
        Calculates the eigenvalues and eigenvectors of a square matrix.

        Parameters
        ----------
        hermitian: bool
            If True, the matrix is assumed to be hermitian.

        Returns
        -------
        eig_vals: array of shape (n, )
            Eigenvalues

        eig_vecs: array of shape (n, n)
            Right Eigenvectors. The normalized eigenvalue eig_vals[i]
            corresponds to the eigenvector eig_vec[:,i].

        """
        pass

    @abstractmethod
    def exp(self,
            tau: complex = 1,
            method: Optional[str] = None,
            is_skew_hermitian: bool = False) -> 'OperatorMatrix':
        """
        The matrix exponential.

        Parameters
        ----------
        tau: complex, optional
            A scalar by which the matrix is multiplied before calculating the
            exponential.

        method: string, optional
            The method by which the matrix exponential is calculated.

        is_skew_hermitian : bool
            If set to true, the matrix is expected to be skew Hermitian, which
            allows to speed up the spectral decomposition.

        Returns
        -------
        exponential: OperatorMatrix
            exponential = exp(A * tau) where A is the stored matrix.

        """
        pass

    @abstractmethod
    def dexp(self,
             direction: 'OperatorMatrix',
             tau: complex = 1,
             compute_expm: bool = False,
             method: Optional[str] = None,
             is_skew_hermitian: bool = False) \
            -> Union['OperatorMatrix', Tuple['OperatorMatrix']]:
        """The Frechet derivative of the exponential in the given direction

        Parameters
        ----------
        direction : OperatorMatrix
            The direction in which the Frechet derivative is to be calculated.

        tau : complex
            A scalar by which the matrix is multiplied before exponentiation.
            This can be i. e. the length of a time segment if a propagator is
            calculated.

        compute_expm : bool
            If set to false only the derivative is calculated and returned.

        method : Optional[string]
            The method by which the exponential is calculated.

        is_skew_hermitian : bool
            If set to true, the matrix is expected to be hermitian, which
            allows to speed up the spectral decomposition.

        Returns
        -------
        prop : OperatorMatrix
            The matrix exponential: exp(self*tau) (Optional, if compute_expm)

        derivative_prop : OperatorMatrix
            The frechet derivative of the matrix exponential:
            (exp((self+direction*dt)*tau)-exp(self*tau)) / dt
        """
        pass

    @abstractmethod
    def identity_like(self) -> 'OperatorMatrix':
        """For square matrices, the identity of same dimension is returned. """

    @abstractmethod
    def truncate_to_subspace(
            self, subspace_indices: Optional[Sequence[int]],
            map_to_closest_unitary: bool = False
    ) -> 'OperatorMatrix':
        """
        Convenience Function to truncate a control matrix to a subspace.

        Parameters
        ----------
        subspace_indices: list of int, optional
            Indices of the subspace to which the control matrix shall be
            truncated. If None, then a reference to the original matrix will be
            returned.

        map_to_closest_unitary: bool
            If True, then the final propagator is mapped to the closest unitary
            before the infidelity is evaluated.

        Returns
        -------
        truncated_matrix: 'OperatorMatrix'
            The truncated operator matrix.

        """
        pass


class DenseOperator(OperatorMatrix):
    """
    Dense control matrix.

    The data is stored as numpy array and uses the implementations of the
    numpy package.

    Parameters
    ----------
    obj: Qobj or numpy array or scipy csr_matrix
        The matrix to be stored and handled as dense matrix.

    Attributes
    ----------
    data: numpy array
        The data stored in a two dimensional numpy array

    """

    def __init__(
            self,
            obj: Union[Qobj, np.ndarray, sp.csr_matrix, 'DenseOperator']) \
            -> None:
        super().__init__()
        self.data = None
        if type(obj) is Qobj:
            self.data = np.array(obj.data.todense())
        elif type(obj) is np.ndarray:
            self.data = obj
        elif type(obj) is sp.csr_matrix:
            self.data = obj.toarray()
        elif type(obj) is DenseOperator:
            self.data = obj.data
        else:
            raise ValueError("Data of this type can not be broadcasted into a "
                             "dense control matrix. Type: " + str(type(obj)))
        self.data = self.data.astype(np.complex128, copy=False)

    def copy(self):
        """See base class. """
        copy_ = DenseOperator(self.data.copy())
        # numpy copy are deep
        return copy_

    def __imul__(self, other: Union['DenseOperator',
                                    complex,
                                    float,
                                    int,
                                    np.generic]) -> 'DenseOperator':
        """See base class. """

        if type(other) == DenseOperator:
            np.matmul(self.data, other.data, out=self.data)
        elif type(other) == np.ndarray:
            np.matmul(self.data, other, out=self.data)
        elif type(other) in VALID_SCALARS:
            self.data *= other
        else:
            raise NotImplementedError(str(type(other)))
        return self

    def __mul__(self, other: Union['DenseOperator', complex, float, int,
                                   np.generic]) -> 'DenseOperator':
        """See base class. """

        if type(other) in VALID_SCALARS:
            out = self.copy()
            out *= other
        elif type(other) == np.ndarray:
            out = self.copy()
            np.matmul(self.data, other, out=out.data)
        elif type(other) == DenseOperator:
            out = self.copy()
            np.matmul(out.data, other.data, out=out.data)
        else:
            raise NotImplementedError(str(type(other)))
        return out

    def __rmul__(self, other: Union['DenseOperator', complex, float, int,
                                    np.generic]) -> 'DenseOperator':
        """See base class. """
        if type(other) == np.ndarray:
            out = self.copy()
            np.matmul(other, self.data, out=out.data)
        elif type(other) in VALID_SCALARS:
            out = self.copy()
            out *= other
        else:
            raise NotImplementedError(str(type(other)))
        return out

    def __iadd__(self, other: 'DenseOperator') -> 'DenseOperator':
        """See base class. """
        if type(other) is DenseOperator:
            self.data += other.data
        elif type(other) == np.ndarray:
            self.data += other
        else:
            raise NotImplementedError(str(type(other)))
        return self

    def __isub__(self, other: 'DenseOperator') -> 'DenseOperator':
        """See base class. """

        if type(other) is DenseOperator:
            self.data -= other.data
        elif type(other) == np.ndarray:
            self.data -= other
        else:
            raise NotImplementedError(str(type(other)))
        return self

    def __getitem__(self, index: Tuple) -> complex:
        """See base class. """
        return self.data[index]

    def dag(self, copy_: bool = True) -> Optional['DenseOperator']:
        """See base class. """
        if copy_:
            cp = self.copy()
            np.conj(cp.data, out=cp.data)
            cp.data = np.copy(cp.data.T)
            return cp
        else:
            np.conj(self.data, out=self.data)
            self.data = self.data.T
            return self

    def conj(self, copy_: bool = True) -> Optional['DenseOperator']:
        """See base class. """
        if copy_:
            copy = self.copy()
            np.conj(copy.data, out=copy.data)
            return copy
        else:
            np.conj(self.data, out=self.data)
            return self

    def transpose(self, copy_: bool = True) -> Optional['DenseOperator']:
        """See base class. """
        if copy_:
            out = self.copy()
        else:
            out = self
        out.data = out.data.transpose()
        return out

    def flatten(self) -> np.ndarray:
        """See base class. """
        return self.data.flatten()

    def tr(self) -> complex:
        """See base class. """
        return self.data.trace()

    def kron(self, other: 'DenseOperator') -> 'DenseOperator':
        """See base class. """
        if type(other) == DenseOperator:
            out = np.kron(self.data, other.data)
        elif type(other) == np.ndarray:
            out = np.kron(self.data, other)
        else:
            raise ValueError('The kronecker product of dense control matrices'
                             'is not defined for: ' + str(type(other)))
        return DenseOperator(out)

    def _exp_diagonalize(self, tau: complex = 1,
                         is_skew_hermitian: bool = False) -> 'DenseOperator':
        """ Calculates the matrix exponential by spectral decomposition.

        Refactored version of _spectral_decomp.

        Parameters
        ----------
        tau : complex
            The matrix is multiplied by tau.

        is_skew_hermitian : bool
            If True, the matrix is expected to be skew hermitian.

        Returns
        -------
        exp: DenseOperator
            Dense operator matrix containing the matrix exponential.

        """
        if is_skew_hermitian:
            eig_val, eig_vec = la.eigh(-1j * self.data)
            eig_val = 1j * eig_val
        else:
            eig_val, eig_vec = la.eig(self.data)

        # apply the exponential function to the eigenvalues and invert the
        # diagonalization transformation
        exp = np.einsum('ij,j,kj->ik', eig_vec, np.exp(tau * eig_val),
                        eig_vec.conj())

        return DenseOperator(exp)

    def _dexp_diagonalization(self,
                              direction: 'DenseOperator', tau: complex = 1,
                              is_skew_hermitian: bool = False,
                              compute_expm: bool = False):
        """ Calculates the matrix exponential by spectral decomposition.

        Refactored version of _spectral_decomp.

        Parameters
        ----------
        direction: DenseOperator
            Direction in which the frechet derivative is calculated. Must be of
            the same shape as self.

        tau : complex
            The matrix is multiplied by tau.

        is_skew_hermitian : bool
            If True, the matrix is expected to be skew hermitian.

        compute_expm : bool
            If True, the matrix exponential is calculated as well.

        Returns
        -------
        exp: DenseOperator
            The matrix exponential. Only returned if compute_expm is set to
            True.

        dexp: DenseOperator
            Frechet derivative of the matrix exponential.

        """
        if is_skew_hermitian:
            eig_val, eig_vec = la.eigh(-1j * self.data)
            eig_val = 1j * eig_val
        else:
            eig_val, eig_vec = la.eig(self.data)

        eig_vec_dag = eig_vec.conj().T

        eig_val_cols = eig_val * np.ones(self.shape)
        eig_val_diffs = eig_val_cols - eig_val_cols.T

        # avoid devision by zero
        eig_val_diffs += np.eye(self.data.shape[0])

        omega = (np.exp(eig_val_diffs * tau) - 1.) / eig_val_diffs

        # override the false diagonal elements.
        np.fill_diagonal(omega, tau)

        direction_transformed = eig_vec @ direction.data @ eig_vec_dag
        dk_dalpha = direction_transformed * omega

        exp = np.einsum('ij,j,jk->ik', eig_vec, np.exp(tau * eig_val),
                        eig_vec_dag)
        # einsum might be less accurate than the @ operator
        dv_dalpha = eig_vec_dag @ dk_dalpha @ eig_vec
        du_dalpha = exp @ dv_dalpha

        if compute_expm:
            return exp, du_dalpha
        else:
            return du_dalpha

    def spectral_decomposition(self, hermitian: bool = False):
        """See base class. """
        if hermitian is False:
            eig_val, eig_vec = scipy.linalg.eig(self.data)
        else:
            eig_val, eig_vec = scipy.linalg.eigh(self.data)

        return eig_val, eig_vec

    def exp(self, tau: complex = 1,
            method: str = "spectral",
            is_skew_hermitian: bool = False) -> 'DenseOperator':
        """
        Matrix exponential.

        Parameters
        ----------
        tau: complex
            The matrix is multiplied by tau before calculating the exponential.

        method: string
            Numerical method used for the calculation of the matrix
            exponential.
            Currently the following are implemented:
            - 'approx', 'Frechet': use the scipy linalg matrix exponential
            - 'first_order': First order taylor approximation
            - 'second_order': Second order taylor approximation
            - 'third_order': Third order taylor approximation
            - 'spectral': Use the self implemented spectral decomposition

        is_skew_hermitian: bool
            Only important for the method 'spectral'. If set to True then the
            matrix is assumed to be skew hermitian in the spectral
            decomposition.

        Returns
        -------
        prop: DenseOperator
            The matrix exponential.

        Raises
        ------
        NotImplementedError:
            If the method given as parameter is not implemented.

        """

        if method == "spectral":
            prop = self._exp_diagonalize(tau=tau,
                                         is_skew_hermitian=is_skew_hermitian)

        elif method in ["approx", "Frechet"]:
            prop = la.expm(self.data * tau)

        elif method == "first_order":
            prop = np.eye(self.data.shape[0]) + self.data * tau

        elif method == "second_order":
            prop = np.eye(self.data.shape[0]) + self.data * tau
            prop += self.data @ self.data * (tau * tau * 0.5)

        elif method == "third_order":
            b = self.data * tau
            prop = np.eye(self.data.shape[0]) + b
            bb = b @ b * 0.5
            prop += bb
            prop += bb @ b * 0.3333333333333333333
        else:
            raise ValueError("Unknown or not specified method for the "
                             "calculation of the matrix exponential:"
                             + str(method))
        return DenseOperator(prop)

    def prop(self, tau: complex = 1) -> 'DenseOperator':
        """See base class. """
        return DenseOperator(self.exp(tau))

    def dexp(self,
             direction: 'DenseOperator',
             tau: complex = 1,
             compute_expm: bool = False,
             method: str = "spectral",
             is_skew_hermitian: bool = False,
             epsilon: float = 1e-10) \
            -> Union['DenseOperator', Tuple['DenseOperator']]:
        """
        Frechet derivative of the matrix exponential.

        Parameters
        ----------
        direction: DenseOperator
            Direction in which the frechet derivative is calculated. Must be of
            the same shape as self.

        tau: complex
            The matrix is multiplied by tau before calculating the exponential.

        compute_expm: bool
            If true, then the matrix exponential is calculated and returned as
            well.

        method: string
            Numerical method used for the calculation of the matrix
            exponential.
            Currently the following are implemented:
            - 'Frechet': Uses the scipy linalg matrix exponential for
            simultaniously calculation of the frechet derivative expm_frechet
            - 'approx': Approximates the Derivative by finite differences.
            - 'first_order': First order taylor approximation
            - 'second_order': Second order taylor approximation
            - 'third_order': Third order taylor approximation
            - 'spectral': Use the self implemented spectral decomposition

        is_skew_hermitian: bool
            Only required, for the method 'spectral'. If set to True, then the
            matrix is assumed to be skew hermitian in the spectral
            decomposition.

        epsilon: float
            Width of the finite difference. Only relevant for the method
            'approx'.

        Returns
        -------
        prop: DenseOperator
            The matrix exponential. Only returned if compute_expm is True!
        prop_grad: DenseOperator
            The frechet derivative d exp(Ax + B)/dx at x=0 where A is the
            direction and B is the matrix stored in self.

        Raises
        ------
        NotImplementedError:
            If the method given as parameter is not implemented.

        """
        prop = None

        if type(direction) != DenseOperator:
            direction = DenseOperator(direction)

        if method == "Frechet":
            a = self.data * tau
            e = direction.data * tau
            if compute_expm:
                prop, prop_grad = la.expm_frechet(a, e, compute_expm=True)
            else:
                prop_grad = la.expm_frechet(a, e, compute_expm=False)

        elif method == "spectral":
            if compute_expm:
                prop, prop_grad = self._dexp_diagonalization(
                    direction=direction, tau=tau,
                    is_skew_hermitian=is_skew_hermitian,
                    compute_expm=compute_expm
                )
            else:
                prop_grad = self._dexp_diagonalization(
                    direction=direction, tau=tau,
                    is_skew_hermitian=is_skew_hermitian,
                    compute_expm=compute_expm
                )

        elif method == "approx":
            d_m = (self.data + epsilon * direction.data) * tau
            dprop = la.expm(d_m)
            prop = self.exp(tau)
            prop_grad = (dprop - prop) * (1 / epsilon)

        elif method == "first_order":
            if compute_expm:
                prop = self.exp(tau)
            prop_grad = direction.data * tau

        elif method == "second_order":
            if compute_expm:
                prop = self.exp(tau)
            prop_grad = direction.data * tau
            prop_grad += (self.data @ direction.data
                          + direction.data @ self.data) * (tau * tau * 0.5)

        elif method == "third_order":
            if compute_expm:
                prop = self.exp(tau)
            prop_grad = direction.data * tau
            prop_grad += (self.data @ direction.data
                          + direction.data @ self.data) * tau * tau * 0.5
            prop_grad += (
                 self.data @ self.data @ direction.data
                 + direction.data @ self.data @ self.data
                 + self.data @ direction.data @ self.data
                         ) * (tau * tau * tau * 0.16666666666666666)
        else:
            raise NotImplementedError(
                'The specified method ' + method + "is not implemented!")
        if compute_expm:
            if type(prop) != DenseOperator:
                prop = DenseOperator(prop)
        if type(prop_grad) != DenseOperator:
            prop_grad = DenseOperator(prop_grad)
        if compute_expm:
            return prop, prop_grad
        else:
            return prop_grad

    def identity_like(self) -> 'DenseOperator':
        """See base class. """
        assert self.shape[0] == self.shape[1]
        return DenseOperator(np.eye(self.shape[0], dtype=complex))

    def truncate_to_subspace(
            self, subspace_indices: Optional[Sequence[int]],
            map_to_closest_unitary: bool = False
    ) -> 'DenseOperator':
        """See base class. """
        if subspace_indices is None:
            return self
        else:
            out = type(self)(
                self.data[np.ix_(subspace_indices, subspace_indices)])
            if map_to_closest_unitary:
                return closest_unitary(out)
            else:
                return out


class SparseOperator(OperatorMatrix):
    pass


def convert_unitary_to_super_operator(
        unitary: Union['OperatorMatrix', np.array]):
    """
    We assume that the unitary U shall be used to propagate a density matrix m
    like

    .. math::

    U m U^dag

    which is equivalent to

    .. math::

        ( U^\ast \otimes U) \vec{m}

    Parameters
    ----------
    unitary: OperatorMatrix or numpy array
        The unitary propagator.

    Returns
    -------
    unitary_super_operator:
        The unitary propagator in the Lindblad formalism.

    Raises
    ------
    ValueError:
        If the operation is not defined for the input type.

    """
    if type(unitary) in (DenseOperator, SparseOperator):
        return unitary.conj(copy_=True).kron(unitary)
    elif isinstance(unitary, np.ndarray):
        return np.kron(np.conj(unitary), unitary)
    else:
        raise ValueError('The target must be given as dense control matrix or '
                         'numpy array!')


def closest_unitary(matrix: OperatorMatrix):
    """
    Calculate the unitary matrix U that is closest with respect to the
    operator norm distance to the general matrix A.

    Parameters
    ----------
    matrix : OperatorMatrix
        The matrix which shall be mapped to the closest unitary.

    Returns
    -------
    unitary : OperatorMatrix
        The closest unitary to the propagator.

    """
    left_singular_vec, __, right_singular_vec_h = scipy.linalg.svd(
        matrix.data)
    return type(matrix)(left_singular_vec.dot(right_singular_vec_h))
