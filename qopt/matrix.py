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
This file contains an operator / matrix class which encapsulates the
numeric operations.

The operators can be stored as dense matrices and a sparse representation
is planed.
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
from unittest import mock
from warnings import warn

try:
    from qutip import Qobj
except ImportError:
    warn('Qutip not installed. plot_bloch_vector_evolution() is not available')
    Qobj = mock.Mock()


VALID_SCALARS = [int, float, complex, np.int8, np.int16, np.int32, np.int64,
                 np.float16, np.float32, np.float64, np.complex64,
                 np.complex128]
# These types are recognised as scalars in the scalar multiplication with
# matrices.


class OperatorMatrix(ABC):
    """
    The abstract base class of the operator matrix for the qopt control
    package.

    It offers an identical interface to use sparse and dense matrices and has
    hence the limitations of both representations in terms of usability.

    Attributes
    ----------
    data
        The stored data. Its type is defined in subclasses.
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

    @abstractmethod
    def __truediv__(self, other: 'OperatorMatrix') -> 'OperatorMatrix':
        pass

    @abstractmethod
    def __itruediv__(self, other: 'OperatorMatrix') -> 'OperatorMatrix':
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
            Index describing an entry in the matrix.

        Returns
        -------
        value: complex
            Matrix element at the position described by the index.

        """
        pass

    @abstractmethod
    def __setitem__(self, key, value) -> None:
        """ Sets the value at the position key.

        Parameters
        ----------
        key: tuple of int, length: 2
            Index specifying an entry in the matrix.

        value: complex
            Value to be set at the position key.

        """
        pass

    @abstractmethod
    def dag(self, do_copy: bool = True) -> Optional['OperatorMatrix']:
        """
        Adjoint (dagger) of the matrix.

        Parameters
        ----------
        do_copy: bool, optional
            If false, then the operation is executed inplace. Otherwise returns
            a new instance. Defaults to True.

        Returns
        -------
        out: OperatorMatrix
            If do_copy is true, then a new instance otherwise self.

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
    def ptrace(self,
               dims: Sequence[int],
               remove: Sequence[int],
               do_copy: bool = True) -> 'OperatorMatrix':
        """
        Partial trace of the matrix.

        If the matrix describes a ket, the corresponding density matrix is
        calculated and used for the partial trace.
        Parameters
        ----------
        dims : list of int
            Dimensions of the subspaces making up the total space on which
            the matrix operates. The product of elements in 'dims' must be
            equal to the matrix' dimension.
        remove : list of int
            The selected subspaces over which the partial trace is formed.
            The given indices correspond to the ordering of subspaces that
            are specified via the 'dim' argument.
        do_copy : bool, optional
            If false, the operation is executed inplace. Otherwise returns
            a new instance. Defaults to True.

        Returns
        -------
        pmat : OperatorMatrix
            The partially traced OperatorMatrix.

        """
        pass

    @abstractmethod
    def conj(self, do_copy: bool = True) -> Optional['OperatorMatrix']:
        r"""
        Complex conjugate of the matrix.

        Parameters
        ----------
        do_copy : bool, optional
            If false, then the operation is executed inplace. Otherwise returns
            a new instance. Defaults to True.

        Returns
        -------
        out: OperatorMatrix
            If do_copy is true, then a new instance otherwise self.

        """
        pass

    def conjugate(self, do_copy: bool = True) -> Optional['OperatorMatrix']:
        """Alias for conj. """
        return self.conj(do_copy=do_copy)

    @abstractmethod
    def transpose(self, do_copy: bool = True) -> Optional['OperatorMatrix']:
        """Transpose of the matrix.

        Parameters
        ----------
        do_copy: bool, optional
            If false, then the operation is executed inplace. Otherwise returns
            a new instance. Defaults to True.

        Returns
        -------
        out: OperatorMatrix
            If do_copy is true, then a new instance otherwise self.

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
    def norm(self, ord: str) -> np.float64:
        """
        Calulates the norm of the matrix.

        Parameters
        ----------
        ord: string
            Defines the norm which is calculated.

        Returns
        -------
        norm: float
            Norm of the Matrix.

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

    @classmethod
    def pauli_0(cls):
        """Pauli 0 i.e. the Identity matrix. """
        return cls(np.eye(2))

    @classmethod
    def pauli_x(cls):
        """Pauli x Matrix. """
        return cls(np.asarray([[0, 1], [1, 0]]))

    @classmethod
    def pauli_y(cls):
        """Pauli y Matrix. """
        return cls(np.asarray([[0, -1j], [1j, 0]]))

    @classmethod
    def pauli_z(cls):
        """Pauli z Matrix. """
        return cls(np.diag([1, -1]))

    @classmethod
    def pauli_m(cls):
        """Pauli minus Matrix i.e. descending operator. """
        return cls(np.asarray([[0, 0], [1, 0]]))

    @classmethod
    def pauli_p(cls):
        """Pauli plus Matrix i.e. ascending operator. """
        return cls(np.asarray([[0, 1], [0, 0]]))


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
            out = DenseOperator(np.matmul(self.data, other))
        elif type(other) == DenseOperator:
            out = DenseOperator(np.matmul(self.data, other.data))
        else:
            raise NotImplementedError(str(type(other)))
        return out

    def __rmul__(self, other: Union['DenseOperator', complex, float, int,
                                    np.generic]) -> 'DenseOperator':
        """See base class. """
        if type(other) == np.ndarray:
            out = DenseOperator(np.matmul(other, self.data))
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
        elif type(other) in VALID_SCALARS:
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
        elif type(other) in VALID_SCALARS:
            self.data -= other
        else:
            raise NotImplementedError(str(type(other)))
        return self

    def __truediv__(self, other: 'DenseOperator') -> 'DenseOperator':
        if isinstance(other, (np.ndarray, *VALID_SCALARS)):
            return DenseOperator(self.data / other)
        raise NotImplementedError(str(type(other)))

    def __itruediv__(self, other: 'DenseOperator') -> 'DenseOperator':
        if isinstance(other, (np.ndarray, *VALID_SCALARS)):
            self.data /= other
            return self
        raise NotImplementedError(str(type(other)))

    def __getitem__(self, index: Tuple) -> np.complex128:
        """See base class. """
        return self.data[index]

    def __setitem__(self, key, value) -> None:
        """See base class. """
        self.data.__setitem__(key, value)

    def __repr__(self):
        """Representation as numpy array. """
        return 'DenseOperator with data: \n' + self.data.__repr__()

    def dag(self, do_copy: bool = True) -> Optional['DenseOperator']:
        """See base class. """
        if do_copy:
            cp = self.copy()
            np.conj(cp.data, out=cp.data)
            cp.data = np.copy(cp.data.T)
            return cp
        else:
            np.conj(self.data, out=self.data)
            self.data = self.data.T
            return self

    def conj(self, do_copy: bool = True) -> Optional['DenseOperator']:
        """See base class. """
        if do_copy:
            copy = self.copy()
            np.conj(copy.data, out=copy.data)
            return copy
        else:
            np.conj(self.data, out=self.data)
            return self

    def transpose(self, do_copy: bool = True) -> Optional['DenseOperator']:
        """See base class. """
        if do_copy:
            out = self.copy()
        else:
            out = self
        out.data = out.data.transpose()
        return out

    def flatten(self) -> np.ndarray:
        """See base class. """
        return self.data.flatten()

    def norm(self, ord: Union[str, None, int] = 'fro') -> np.float64:
        """
        Calulates the norm of the matrix.

        Uses the implementation of numpy.linalg.norm.

        Parameters
        ----------
        ord: string
            Defines the norm which is calculated. Defaults to the Frobenius norm
            'fro'.

        Returns
        -------
        norm: float
            Norm of the Matrix.

        """
        return np.linalg.norm(self.data, ord=ord)

    def tr(self) -> complex:
        """See base class. """
        return self.data.trace()

    def ptrace(self,
               dims: Sequence[int],
               remove: Sequence[int],
               do_copy: bool = True) -> 'DenseOperator':
        """
        Partial trace of the matrix.

        If the matrix describes a ket, the corresponding density matrix is
        calculated and used for the partial trace.

        This implementation closely follows that of QuTip's qobj._ptrace_dense.
        Parameters
        ----------
        dims : list of int
            Dimensions of the subspaces making up the total space on which
            the matrix operates. The product of elements in 'dims' must be
            equal to the matrix' dimension.
        remove : list of int
            The selected subspaces as indices over which the partial trace is
            formed. The given indices correspond to the ordering of
            subspaces specified in the 'dim' argument.
        do_copy : bool, optional
            If false, the operation is executed inplace. Otherwise returns
            a new instance. Defaults to True.

        Returns
        -------
        pmat : OperatorMatrix
            The partially traced OperatorMatrix.

        Raises
        ------
        AssertionError:
            If matrix dimension does not match specified dimensions.

        Examples
        --------
         ghz_ket = DenseOperator(np.array([[1,0,0,0,0,0,0,1]]).T) / np.sqrt(2)
         ghz_rho = ghz_ket * ghz_ket.dag()
         ghz_rho.ptrace(dims=[2,2,2], remove=[0,2])
        DenseOperator with data:
        array([[0.5+0.j, 0. +0.j],
               [0. +0.j, 0.5+0.j]])
        """

        if self.shape[1] == 1:
            mat = (self * self.dag()).data
        else:
            mat = self.data
        if mat.shape[0] != np.prod(dims):
            raise AssertionError("Specified dimensions do not match "
                                 "matrix dimension.")
        n_dim = len(dims)  # number of subspaces
        dims = np.asarray(dims, dtype=int)

        remove = list(np.sort(remove))
        # indices of subspace that are kept
        keep = list(set(np.arange(n_dim)) - set(remove))

        dims_rm = (dims[remove]).tolist()
        dims_keep = (dims[keep]).tolist()
        dims = list(dims)

        # 1. Reshape: Split matrix into subspaces
        # 2. Transpose: Change subspace/index ordering such that the subspaces
        # over which is traced correspond to the first axes
        # 3. Reshape: Merge each, subspaces to be removed (A) and to be kept
        # (B), common spaces/axes.
        # The trace of the merged spaces (A \otimes B) can then be
        # calculated as Tr_A(mat) using np.trace for input with
        # more than two axes effectively resulting in
        # pmat[j,k] = Sum_i mat[i,i,j,k] for all j,k = 0..prod(dims_keep)
        pmat = np.trace(mat.reshape(dims + dims)
                           .transpose(remove + [n_dim + q for q in remove] +
                                      keep + [n_dim + q for q in keep])
                           .reshape([np.prod(dims_rm),
                                    np.prod(dims_rm),
                                    np.prod(dims_keep),
                                    np.prod(dims_keep)])
                        )

        if do_copy:
            return DenseOperator(pmat)
        else:
            self.data = pmat
            return self

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
            eig_val, eig_vec = np.linalg.eigh(-1j * self.data)
            eig_val = 1j * eig_val
        else:
            eig_val, eig_vec = np.linalg.eig(self.data)

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
            eig_val, eig_vec = np.linalg.eigh(-1j * self.data)
            eig_val = 1j * eig_val
        else:
            eig_val, eig_vec = np.linalg.eig(self.data)

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
            prop_grad = (DenseOperator(dprop) - prop) * (1 / epsilon)

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
        elif self.shape[0] == self.shape[1]:
            # square matrix
            out = type(self)(
                self.data[np.ix_(subspace_indices, subspace_indices)])
            if map_to_closest_unitary:
                out = closest_unitary(out)
        elif self.shape[0] == 1:
            # bra-vector
            out = type(self)(self.data[np.ix_([0], subspace_indices)])
            if map_to_closest_unitary:
                out *= 1 / out.norm('fre')
        elif self.shape[0] == 1:
            # ket-vector
            out = type(self)(self.data[np.ix_(subspace_indices, [0])])
            if map_to_closest_unitary:
                out *= 1 / out.norm('fre')
        else:
            out = type(self)(self.data[np.ix_(subspace_indices)])

        return out


class SparseOperator(OperatorMatrix):
    pass


def convert_unitary_to_super_operator(
        unitary: Union['OperatorMatrix', np.array]):
    r"""
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
        The unitary propagator in the Lindblad formalism. Same type as input.

    Raises
    ------
    ValueError:
        If the operation is not defined for the input type.

    """
    if type(unitary) in (DenseOperator, SparseOperator):
        return unitary.conj(do_copy=True).kron(unitary)
    elif isinstance(unitary, np.ndarray):
        return np.kron(np.conj(unitary), unitary)
    else:
        raise ValueError('The target must be given as dense control matrix or '
                         'numpy array!')


def ket_vectorize_density_matrix(
        density_matrix: Union['OperatorMatrix', np.array]):
    r"""
    Vectorizes a density matrix column-wise as ket vector.

    Parameters
    ----------
    density_matrix: OperatorMatrix or numpy array
        The density matrix.

    Returns
    -------
    density_ket_vector:
        The density matrix as ket vector for the Liouville formalism.

    Raises
    ------
    ValueError:
        If the operation is not defined for the input type.

    ValueError:
        If the density matrix is not given in square shape.

    """
    if not density_matrix.shape[0] == density_matrix.shape[1]:
        raise ValueError('The density matrix must be of square shape. ')
    if type(density_matrix) in (DenseOperator, SparseOperator):
        vectorized_matrix = density_matrix.copy()
        vectorized_matrix.data = vectorized_matrix.data.T.reshape(
            [vectorized_matrix.data.size, 1]
        )
        return vectorized_matrix
    elif isinstance(density_matrix, np.ndarray):
        return density_matrix.T.reshape([density_matrix.size, 1])
    else:
        raise ValueError('The target must be given as dense control matrix or '
                         'numpy array!')


def convert_ket_vectorized_density_matrix_to_square(
        vectorized_density_matrix: Union['OperatorMatrix', np.array]):
    r"""
    Bring vectorized density matrix back into square form.

    Parameters
    ----------
    vectorized_density_matrix: OperatorMatrix or numpy array
        The density matrix.

    Returns
    -------
    density_ket_vector:
        The density matrix as ket vector for the Liouville formalism.

    Raises
    ------
    ValueError:
        If the operation is not defined for the input type.

    ValueError:
        If the density matrix is not given as ket vector.

    """
    if not vectorized_density_matrix.shape[1] == 1:
        raise ValueError('The density matrix must be vectorized as ket. ')
    if type(vectorized_density_matrix) in (DenseOperator, SparseOperator):
        d = int(np.sqrt(vectorized_density_matrix.data.size))
        vectorized_matrix = vectorized_density_matrix.copy()
        vectorized_matrix.data = vectorized_matrix.data.reshape(
            [d, d]
        ).T
        return vectorized_matrix
    elif isinstance(vectorized_density_matrix, np.ndarray):
        d = int(np.sqrt(vectorized_density_matrix.size))
        return vectorized_density_matrix.reshape([d, d]).T
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
