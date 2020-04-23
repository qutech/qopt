"""
This file contains a matrix class which implements the computationally expensive
numerics.

The matrices can be stored and handled either as dense or sparse
matrices. The most frequently used and computationally expensive function is the
matrix exponential and its derivative. These operations are required to
calculate the analytic solution of the Schroedinger and Lindblad master
equation.


Classes
-------
OperatorMatrix:
    Abstract base class which defines the interface and implements standard
    functions.

OperatorDense
    The dense control matrices are based on numpy arrays.

OperatorSparse
    The sparse control matrices are based on the QuTiP fast CSR sparse matrices
    which in turn inherit from the scipy CSR sparse matrices.

"""

import numpy as np
import scipy
# TODO: These imports dont look so nice
import scipy.sparse as sp
from scipy.sparse import identity
import scipy.linalg as la
# from qutip.cy.spmatfuncs import spmv
from qutip import Qobj
from qutip.sparse import sp_eigs, sp_expm
# from qutip.cy.spmath import (zcsr_adjoint, zcsr_trace)
from util import needs_refactoring
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, Sequence


# TODO: You must be kidding me! How is this globally defined?
matrix_opt = {
    "fact_mat_round_prec": 1e-10,
    "_mem_eigen_adj": False,
    "_mem_prop": False,
    "epsilon": 1e-6,
    "method": "Frechet",
    "sparse2dense": False,
    "sparse_exp": True}

VALID_SCALARS = [int, float, complex, np.int8, np.int16, np.int32, np.int64,
                 np.float16, np.float32, np.float64, np.complex64,
                 np.complex128]


class OperatorMatrix(ABC):
    """
    The abstract base class of the control matrix for the qutip control package.

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
        """
        Addition with other control matrices or fitting data.

        See __iadd__ of subclasses which is used.

        """
        out = self.copy()
        out += other
        return out

    @abstractmethod
    def __iadd__(self, other: 'OperatorMatrix') -> 'OperatorMatrix':
        """In place addition. """
        return self

    @abstractmethod
    def __mul__(self, other: Union['OperatorMatrix', complex, float, int,
                                   np.generic]) -> 'OperatorMatrix':
        """Matrix or scalar multiplication. """
        return self

    @abstractmethod
    def __imul__(self, other: Union['OperatorMatrix', complex, float, int,
                                    np.generic]) -> 'OperatorMatrix':
        """In place matrix or scalar multiplication. """
        pass

    @abstractmethod
    def __rmul__(self, other: Union['OperatorMatrix', complex, float, int,
                                    np.generic]) -> 'OperatorMatrix':
        """Reflective matrix or scalar multiplication. """
        return self

    @abstractmethod
    def __isub__(self, other: 'OperatorMatrix') -> 'OperatorMatrix':
        """In place subtraction. """
        pass

    def __sub__(self, other: 'OperatorMatrix') -> 'OperatorMatrix':
        """Subtraction of other control matrix or fitting data. """
        out = self.copy()
        out -= other
        return out

    @property
    def shape(self) -> Tuple:
        """Returns the shape of the matrix. """
        return self.data.shape

    @abstractmethod
    def __getitem__(self, item: Tuple) -> complex:
        """Returns the corresponding matrix element. """
        pass

    @abstractmethod
    def dag(self, copy_: bool) -> Optional['OperatorMatrix']:
        """
        Adjoint (dagger) of the matrix.

        Parameters
        ----------
        copy_: bool
            If false, then the operation is executed inplace. Otherwise returns
            a new instance.

        Returns
        -------
        out: OperatorMatrix
            If copy_ is true, then a new instance otherwise self.

        """
        return self

    @abstractmethod
    def tr(self) -> complex:
        """Trace of the matrix. """
        return 0j

    @abstractmethod
    def conj(self, copy_: bool = True) -> Optional['OperatorMatrix']:
        """
        Complex conjugate of the matrix.

        Parameters
        ----------
        copy_: bool
            If false, then the operation is executed inplace. Otherwise returns
            a new instance.

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
        copy_: bool
            If false, then the operation is executed inplace. Otherwise returns
            a new instance.

        Returns
        -------
        out: OperatorMatrix
            If copy_ is true, then a new instance otherwise self.

        """

    @abstractmethod
    def kron(self, other: 'OperatorMatrix') -> 'OperatorMatrix':
        """Kronecker matrix product. """
        pass

    @abstractmethod
    def flatten(self) -> np.ndarray:
        """
        Flattens the matrix.

        Returns
        -------
        out: np.ndarray
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
            coresponds to the eigenvector eig_vec[:,i].

        """
        pass

    @abstractmethod
    def prop(self, tau: complex) -> 'OperatorMatrix':
        """Propagator in the time slice.

        Parameters
        ----------
        tau : double
            Duration of the time slice.

        Returns
        -------
        prop : OperatorMatrix
            Solution to the Schrödinger equation: exp(self*tau)

        Todo:
            * Is this really useful?
        """
        pass

    @abstractmethod
    def exp(self, tau: complex = 1,
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
    def dexp(self, direction: 'OperatorMatrix', tau: complex = 1,
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
            This can be i. e. the lenght of a time segment if a propagator is
            calculated.

        compute_expm : bool
            If set to false only the derivative is calculated and returned.

        method : Optional[string]
            The method by which the exponential is calculated.

        is_skew_hermitian : bool
            If set to true, the matrix is expected to be Hermitian, which allows
            to speed up the spectral decomposition.

        Returns
        -------
        prop : OperatorMatrix
            The matrix exponential: exp(self*tau) (Optional, if compute_expm)

        derr : OperatorMatrix
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
        truncated_matrix: 'ControlMatrix'
            The truncated control matrix.

        """
        pass


class OperatorDense(OperatorMatrix):
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

    Methods
    -------
    _spectral_decomp:
        Applies the spectral decomposition of the stored matrix.

    _eig_vec_adj
        Returns the adjoint eigenvectors.

    """

    def __init__(
            self, obj: Union[Qobj, np.ndarray, sp.csr_matrix, 'OperatorDense']) \
            -> None:
        super().__init__()
        self.full = True
        self.data = None
        if type(obj) is Qobj:
            self.data = np.array(obj.data.todense())
            self._size = self.data.shape[0]
        elif type(obj) is np.ndarray:
            self.data = obj
            self._size = self.data.shape[0]
        elif type(obj) is sp.csr_matrix:
            self.data = obj.toarray()
            self._size = obj.shape[0]
        elif type(obj) is OperatorDense:
            self.data = obj.data
            self._size = obj._size
        else:
            raise ValueError("Data of this type can not be broadcasted into a "
                             "dense control matrix. Type: " + str(type(obj)))
        self.data = self.data.astype(np.complex128, copy=False)

    def copy(self):
        """See base class. """
        copy_ = OperatorDense(self.data.copy())
        # numpy copy are deep
        return copy_

    def __imul__(self, other: Union[
        'OperatorDense', complex, float, int, np.generic]) -> 'OperatorDense':
        """
        In place matrix or scalar multiplication.

        Parameters
        ----------
        other: ControlMatrix or numpy array or scalar
            If other is a scalar value (int, float, complex, np.complex128)
            then the matrix is multiplied by this scalar value.
            If other is another instance of control matrix of a numpy array,
            then a matrix multiplication is applied.

        Returns
        -------
        self:
            Returns itself because the multiplication is executed in place.

        Raises
        ------
        NotImplementedError:
            If the implementation with objects of type other is not implemented.

        """
        if type(other) == OperatorDense:
            np.matmul(self.data, other.data, out=self.data)
        elif type(other) == np.ndarray:
            np.matmul(self.data, other, out=self.data)
        elif type(other) in VALID_SCALARS:
            self.data *= other
        else:
            raise NotImplementedError(str(type(other)))
        return self

    def __mul__(self, other: Union['OperatorDense', complex, float, int,
                                   np.generic]) -> 'OperatorDense':
        """
        Matrix or scalar multiplication.

        Parameters
        ----------
        other: ControlMatrix or numpy array or scalar
            If other is a scalar value (int, float, complex, np.complex128)
            then the matrix is multiplied by this scalar value.
            If other is another instance of control matrix of a numpy array,
            then a matrix multiplication is applied.

        Returns
        -------
        out:
            New instance of ControlDense containing the result of the
            multiplication.

        Raises
        ------
        NotImplementedError:
            If the implementation with objects of type other is not implemented.

        """
        if type(other) in VALID_SCALARS:
            out = self.copy()
            out *= other
        elif type(other) == np.ndarray:
            out = self.copy()
            np.matmul(self.data, other, out=out.data)
        elif type(other) == OperatorDense:
            out = self.copy()
            np.matmul(out.data, other.data, out=out.data)
        else:
            raise NotImplementedError(str(type(other)))
        return out

    def __rmul__(self, other: Union['OperatorDense', complex, float, int,
                                    np.generic]) -> 'OperatorDense':
        """
        Reflective matrix or scalar multiplication.

        Parameters
        ----------
        other: numpy array or scalar
            If other is a scalar value (int, float, complex, np.complex128)
            then the matrix is multiplied by this scalar value.
            If other is a numpy array, then a matrix multiplication is applied.

        Returns
        -------
        out:
            New instance of ControlDense containing the result of the
            multiplication.

        Raises
        ------
        NotImplementedError:
            If the operation is not implemented for objects of others type.

        """
        if type(other) == np.ndarray:
            out = self.copy()
            np.matmul(other, self.data, out=out.data)
        elif type(other) in VALID_SCALARS:
            out = self.copy()
            out *= other
        else:
            raise NotImplementedError(str(type(other)))
        return out

    def __iadd__(self, other: 'OperatorDense') -> 'OperatorDense':
        """
        In place addition.

        Parameters
        ----------
        other: ControlDense or numpy array
            The addition is only implemented for other instances of ControlDense
            or numpy arrays of the same dimensions.
            Note that scalars are not interpreted as multiples of the identity
            matrix as it is the case for qutip Qobj!

        Returns
        -------
        self:
            As the operation is executed in place.

        Raises
        ------
        NotImplementedError:
            If the operation is not implemented for objects of others type.

        """
        if type(other) is OperatorDense:
            self.data += other.data
        elif type(other) == np.ndarray:
            self.data += other
        else:
            raise NotImplementedError(str(type(other)))
        return self

    def __isub__(self, other: 'OperatorDense') -> 'OperatorDense':
        """
        In place subtraction.

        Parameters
        ----------
        other: ControlDense or numpy array
            The subtraction is only implemented for other instances of
            ControlDense or numpy arrays of the same dimensions.
            Note that scalars are not interpreted as multiples of the identity
            matrix as it is the case for qutip Qobj!

        Returns
        -------
        self:
            As the operation is executed in place.

        Raises
        ------
        NotImplementedError:
            If the operation is not implemented for objects of others type.

        """

        if type(other) is OperatorDense:
            self.data -= other.data
        elif type(other) == np.ndarray:
            self.data -= other
        else:
            raise NotImplementedError(str(type(other)))
        return self

    def __getitem__(self, item: Tuple) -> complex:
        """See base class. """
        return self.data[item]

    def dag(self, copy_: bool = True) -> Optional['OperatorDense']:
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

    def conj(self, copy_: bool = True) -> Optional['OperatorDense']:
        """See base class. """
        if copy_:
            copy = self.copy()
            np.conj(copy.data, out=copy.data)
            return copy
        else:
            np.conj(self.data, out=self.data)
            return self

    def transpose(self, copy_: bool = True) -> Optional['OperatorDense']:
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

    def kron(self, other: 'OperatorDense') -> 'OperatorDense':
        """
        Computes the kronecker matrix product with another matrix.

        Parameters
        ----------
        other: OperatorDense or np.ndarray
            Second factor of the kronecker product.

        Returns
        -------
        out: OperatorDense
            Dense control matrix containing the product.

        Raises
        ------
        ValueError:
            If other is not of type ControlDense or np.ndarray.

        """
        if type(other) == OperatorDense:
            out = np.kron(self.data, other.data)
        elif type(other) == np.ndarray:
            out = np.kron(self.data, other)
        else:
            raise ValueError('The kronecker product of dense control matrices'
                             'is not defined for: ' + str(type(other)))
        return OperatorDense(out)

    def _exp_diagonalize(self, tau: complex = 1,
                         is_skew_hermitian: bool = False) -> 'OperatorDense':
        """ Calculates the matrix exponential by spectral decomposition.

        Refactored version of _spectral_decomp.

        Parameters
        ----------
        tau : complex
            The matrix is multiplied by tau.

        is_skew_hermitian : bool
            If True, the matrix is expected to be skew hermitian.

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

        return OperatorDense(exp)

    def _dexp_diagonalization(self, direction: 'OperatorDense', tau: complex = 1,
                              is_skew_hermitian: bool = False,
                              compute_expm: bool = False):
        """ Calculates the matrix exponential by spectral decomposition.

        Refactored version of _spectral_decomp.

        Parameters
        ----------
        direction: OperatorDense
            Direction in which the frechet derivative is calculated. Must be of
            the same shape as self.

        tau : complex
            The matrix is multiplied by tau.

        is_skew_hermitian : bool
            If True, the matrix is expected to be skew hermitian.

        compute_expm : bool
            If True, the matrix exponential is calculated as well.

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
        eig_val_diffs += np.eye(self._size)

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

    @needs_refactoring
    def _spectral_decomp(self, tau: complex = 1,
                         is_skew_hermitian: bool = False) -> None:
        """ Calculates the eigenvalues and right eigenvalues.

        Calculates the diagonalization of the dynamics generator
        generating lists of eigenvectors, propagators in the diagonalised
        basis, and the 'factormatrix' used in calculating the propagator
        gradient.

        Parameters
        ----------
        tau : complex
            The matrix is multiplied by tau.

        is_skew_hermitian : bool
            If True, the matrix is expected to be skew hermitian.

        """
        if is_skew_hermitian:
            eig_val, eig_vec = la.eigh(-1j * self.data)
            eig_val = 1j * eig_val
        else:
            eig_val, eig_vec = la.eig(self.data)

        eig_val_tau = eig_val * tau
        prop_eig = np.exp(eig_val_tau)

        o = np.ones([self._size, self._size])
        eig_val_cols = eig_val_tau * o
        eig_val_diffs = eig_val_cols - eig_val_cols.T

        prop_eig_cols = prop_eig * o
        prop_eig_diffs = prop_eig_cols - prop_eig_cols.T

        # degen_mask should essentially be the diagonal elements.

        degen_mask = np.abs(eig_val_diffs) < matrix_opt["fact_mat_round_prec"]
        eig_val_diffs[degen_mask] = 1
        factors = prop_eig_diffs / eig_val_diffs
        factors[degen_mask] = prop_eig_cols[degen_mask]

        self._factormatrix = factors
        self._prop_eigen = np.diagflat(prop_eig)
        self._eig_vec = eig_vec
        if matrix_opt["_mem_eigen_adj"] is not None:
            self._eig_vec_dag = eig_vec.conj().T

    @property
    def _eig_vec_adj(self) -> np.ndarray:
        """Adjoint eigenvectors. """
        if matrix_opt["_mem_eigen_adj"]:
            return self._eig_vec.conj().T
        else:
            return self._eig_vec_dag

    def exp(self, tau: complex = 1,
            method: Optional[str] = None,
            is_skew_hermitian: bool = False) -> 'OperatorDense':
        """
        Matrix exponential.

        Parameters
        ----------
        tau: complex
            The matrix is multiplied by tau before calculating the exponential.

        method: string
            Numerical method used for the calculation of the matrix exponential.
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
        prop: OperatorDense
            The matrix exponential.

        Raises
        ------
        NotImplementedError:
            If the method given as parameter is not implemented.

        """
        if method is None:
            method = matrix_opt['method']

        if matrix_opt["_mem_prop"] and self._prop is not None:
            return self._prop

        if method == "spectral":
            prop = self._exp_diagonalize(tau=tau,
                                         is_skew_hermitian=is_skew_hermitian)

        elif method == "spectral_unrefactored":
            if self._eig_vec is None:
                self._spectral_decomp(tau=tau,
                                      is_skew_hermitian=is_skew_hermitian)
            prop = self._eig_vec.dot(self._prop_eigen).dot(self._eig_vec_adj)

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

        if matrix_opt["_mem_prop"]:
            self._prop = OperatorDense(prop)
        return OperatorDense(prop)

    def prop(self, tau: complex = 1) -> 'OperatorDense':
        """See base class. """
        return OperatorDense(self.exp(tau))

    def dexp(self, direction: 'OperatorDense', tau: complex = 1,
             compute_expm: bool = False, method: Optional[str] = None,
             is_skew_hermitian: bool = False) \
            -> Union['OperatorDense', Tuple['OperatorDense']]:
        """
        Frechet derivative of the matrix exponential.

        Parameters
        ----------
        direction: OperatorDense
            Direction in which the frechet derivative is calculated. Must be of
            the same shape as self.

        tau: complex
            The matrix is multiplied by tau before calculating the exponential.

        compute_expm: bool
            If true, then the matrix exponential is calculated and returned as
            well.

        method: string
            Numerical method used for the calculation of the matrix exponential.
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

        Returns
        -------
        prop: OperatorDense
            The matrix exponential. Only returned if compute_expm is True!
        prop_grad: OperatorDense
            The frechet derivative d exp(Ax + B)/dx at x=0 where A is the
            direction and B is the matrix stored in self.

        Raises
        ------
        NotImplementedError:
            If the method given as parameter is not implemented.

        """
        prop = None

        if type(direction) != OperatorDense:
            direction = OperatorDense(direction)

        if method is None:
            method = matrix_opt['method']
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

        elif method == "spectral_unrefactored":
            if self._eig_vec is None:
                self._spectral_decomp(tau=tau,
                                      is_skew_hermitian=is_skew_hermitian)
            if compute_expm:
                prop = self.exp(tau=tau, is_skew_hermitian=is_skew_hermitian)
                # put control dyn_gen in combined dg diagonal basis
            cdg = self._eig_vec_dag.dot(direction.data).dot(self._eig_vec)
            # multiply (elementwise) by timeslice and factor matrix
            cdg = np.multiply(cdg * tau, self._factormatrix)
            # Return to canonical basis
            prop_grad = self._eig_vec.dot(cdg).dot(self._eig_vec_adj)

        elif method == "approx":
            d_m = (self.data + matrix_opt["epsilon"] * direction.data) * tau
            dprop = la.expm(d_m)
            prop = self.exp(tau)
            prop_grad = (dprop - prop) * (1 / matrix_opt["epsilon"])

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
                                 + self.data @ direction.data @ self.data) \
                         * (tau * tau * tau * 0.16666666666666666)
        else:
            raise NotImplementedError(
                'The specified method ' + method + "is not implemented!")
        if compute_expm:
            if type(prop) != OperatorDense:
                prop = OperatorDense(prop)
        if type(prop_grad) != OperatorDense:
            prop_grad = OperatorDense(prop_grad)
        if compute_expm:
            return prop, prop_grad
        else:
            return prop_grad

    def identity_like(self) -> 'OperatorDense':
        """See base class. """
        assert self.shape[0] == self.shape[1]
        return OperatorDense(np.eye(self.shape[0], dtype=complex))

    def truncate_to_subspace(
            self, subspace_indices: Optional[Sequence[int]],
            map_to_closest_unitary: bool = False
    ) -> 'OperatorDense':
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


class OperatorSparse(OperatorMatrix):
    """
    Sparse control matrix.

    The implementation is based on the fast CSR matrix implementation in
    qutip.fast_sparse which in turn uses the scipy csr_sparse matrix.

    Parameters
    ----------
    obj: Qobj or numpy array or scipy csr_matrix
        The matrix to be stored and handled as sparse matrix.

    Attributes
    ----------
    data: numpy array
        The data stored as scipy csr_matrix or qutip fast csr_matrix.

    Methods
    -------
    _spectral_decomp:
        Applies the spectral decomposition of the stored matrix.

    _eig_vec_adj
        Returns the adjoint eigenvectors.

    Raises
    ------
    ValueError
        If obj could not be broadcasted into a csr_matrix.

    Todo:
        * Investigate if data should always have the same type
        * Properly document the sparse functions

    """

    def __init__(self, obj=None):
        super().__init__()
        self.full = False
        if type(obj) is Qobj:
            self._size = obj.shape[0]
            self.data = obj.data
        elif type(obj) is np.ndarray:
            self._size = obj.shape[0]
            self.data = sp.csr_matrix(obj)
        elif type(obj) is sp.csr_matrix:
            self._size = obj.shape[0]
            self.data = obj
        else:
            raise ValueError('The object could not be broadcasted into a sparse'
                             'matrix! type: ' + str(type(obj)))
        self.method = "spectral"

    def copy(self):
        """See base class. """
        copy_ = OperatorSparse(self.data.copy())
        return copy_

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex)):
            out = self.copy()
            out *= other
        # elif isinstance(other, Qobj):
        #    out = self.copy()
        #    out.data = other.data * out.data
        # elif isinstance(other, sp.csr_matrix):
        #    out = self.copy()
        #    out.data = other.data * out.data
        elif type(other) == np.ndarray:
            if len(other.shape) == 1:
                out = spmv(self.data.T, other)
            else:
                out = (self.data.T * other.T).T
        else:
            raise ValueError("Control Dense matrix multiplication is not "
                             "implemented for objects of type:"
                             + str(type(other)))
        return out

    def __mul__(self, other):
        if isinstance(other, OperatorSparse):
            out = self.copy()
            out.data = self.data * other.data
        elif type(other) in [int, float, complex]:
            out = self.copy()
            out.data = self.data * other
        elif type(other) == np.ndarray:
            if len(other.shape) == 1:
                out = spmv(self.data, other)
            else:
                out = self.data * other

        else:
            raise NotImplementedError(type(other))
        return out

    def __imul__(self, other):
        if type(other) == OperatorSparse:
            self.data = self.data * other.data
        elif type(other) in [int, float, complex]:
            self.data = self.data * other
        # elif isinstance(other, np.ndarray):
        #    self.data = sp.csr_matrix(spmv(self.data, other))
        else:
            raise NotImplementedError(type(other))
        return self

    def __iadd__(self, other):
        if isinstance(other, OperatorSparse):
            self.data += other.data
        else:
            raise NotImplementedError(type(other))
        return self

    def __isub__(self, other):
        if isinstance(other, OperatorSparse):
            self.data -= other.data
        else:
            raise NotImplementedError(type(other))
        return self

    def __getitem__(self, item):
        return self.data[item]

    def dag(self):
        cp = self.copy()
        cp.data = zcsr_adjoint(self.data)
        return cp

    def tr(self):
        return zcsr_trace(self.data, 0)

    def conj(self, copy_=True):
        if copy_:
            return OperatorSparse(self.data.conj(copy=True))
        else:
            self.data = self.data.conj(copy=False)
            return self

    def flatten(self, out=None):
        return self.data.toarray().flatten()

    def kron(self, other):
        """
        Computes the kronecker matrix product with another matrix.

        Parameters
        ----------
        other: OperatorSparse or scipy.csr_sparse
            Second factor of the kronecker product.

        Returns
        -------
        out: OperatorSparse
            Sparse control matrix containing the product.

        Raises
        ------
        ValueError:
            If other is not of type ControlSparse or scipy.sparse.csr_matrix.

        """
        if type(other) == OperatorSparse:
            out = sp.kron(self.data, other.data)
        elif type(other) == sp.csr_matrix:
            out = sp.kron(self.data, other)
        else:
            raise ValueError('The kronecker product of sparse control matrices'
                             'is not defined for: ' + type(other))
        return OperatorSparse(out)

    def _spectral_decomp(self, tau):
        """
        Calculates the diagonalization of the dynamics generator
        generating lists of eigenvectors, propagators in the diagonalised
        basis, and the 'factormatrix' used in calculating the propagator
        gradient.
        """
        eig_val, eig_vec = sp_eigs(self.data, 0)
        # eig_val, eig_vec = sp_eigs(H.data, H.isherm,
        #                           sparse=self.sparse_eigen_decomp)
        eig_vec = eig_vec.T

        eig_val_tau = eig_val * tau
        prop_eig = np.exp(eig_val_tau)

        o = np.ones([self._size, self._size])
        eig_val_cols = eig_val_tau * o
        eig_val_diffs = eig_val_cols - eig_val_cols.T

        prop_eig_cols = prop_eig * o
        prop_eig_diffs = prop_eig_cols - prop_eig_cols.T

        degen_mask = np.abs(eig_val_diffs) < matrix_opt["fact_mat_round_prec"]
        eig_val_diffs[degen_mask] = 1
        factors = prop_eig_diffs / eig_val_diffs
        factors[degen_mask] = prop_eig_cols[degen_mask]

        self._factormatrix = factors
        self._prop_eigen = np.diagflat(prop_eig)
        self._eig_vec = eig_vec
        if not matrix_opt["_mem_eigen_adj"]:
            self._eig_vec_dag = eig_vec.conj().T

    @property
    def _eig_vec_adj(self):
        if matrix_opt["_mem_eigen_adj"]:
            return self._eig_vec.conj().T
        else:
            return self._eig_vec_dag

    def exp(self, tau=1, method=None):
        if matrix_opt["_mem_prop"] and self._prop:
            return self._prop

        if matrix_opt["method"] == "spectral":
            if self._eig_vec is None:
                self._spectral_decomp(tau)
            prop = self._eig_vec.dot(self._prop_eigen).dot(self._eig_vec_adj)
        elif matrix_opt["method"] in ["approx", "Frechet"]:
            if matrix_opt["sparse2dense"]:
                prop = la.expm(self.data.toarray() * tau)
            else:
                prop = sp_expm(self.data * tau,
                               sparse=matrix_opt["sparse_exp"])
        elif matrix_opt["method"] == "first_order":
            if matrix_opt["sparse2dense"]:
                prop = np.eye(self.data.shape[0]) + self.data.toarray() * tau
            else:
                prop = identity(self.data.shape[0], format='csr') + \
                       self.data * tau
        elif matrix_opt["method"] == "second_order":
            if matrix_opt["sparse2dense"]:
                M = self.data.toarray() * tau
                prop = np.eye(self.data.shape[0]) + M
                prop += M @ M * 0.5
            else:
                M = self.data * tau
                prop = identity(self.data.shape[0], format='csr') + M
                prop += M * M * 0.5
        elif matrix_opt["method"] == "third_order":
            if matrix_opt["sparse2dense"]:
                B = self.data.toarray() * tau
                prop = np.eye(self.data.shape[0]) + B
                BB = B @ B * 0.5
                prop += BB
                prop += BB @ B * 0.3333333333333333333
            else:
                B = self.data * tau
                prop = identity(self.data.shape[0], format='csr') + B
                BB = B * B * 0.5
                prop += BB
                prop += BB * B * 0.3333333333333333333

        if matrix_opt["_mem_prop"]:
            self._prop = prop
        return prop

    def prop(self, tau):
        if matrix_opt["sparse2dense"]:
            return OperatorDense(self.exp(tau))
        return OperatorSparse(self.exp(tau))

    def dexp(self, direction, tau=1, compute_expm=False, method=None):
        if method is None or method == "Frechet":
            A = (self.data * tau).toarray()
            E = (direction.data * tau).toarray()
            if compute_expm:
                prop_dense, prop_grad_dense = la.expm_frechet(A, E)
                prop = prop_dense
                # prop = sp.csr_matrix(prop_dense)
            else:
                prop_grad_dense = la.expm_frechet(A, E,
                                                  compute_expm=compute_expm)
            prop_grad = prop_grad_dense
            # prop_grad = sp.csr_matrix(prop_grad_dense)

        elif method == "spectral":
            if self._eig_vec is None:
                self._spectral_decomp(tau)
            if compute_expm:
                prop = self.exp(tau)
            # put control dyn_gen in combined dg diagonal basis
            cdg = self._eig_vec_adj.dot(direction.data.toarray()).dot(
                self._eig_vec)
            # multiply (elementwise) by timeslice and factor matrix
            cdg = np.multiply(cdg * tau, self._factormatrix)
            # Return to canonical basis
            prop_grad = self._eig_vec.dot(cdg).dot(self._eig_vec_adj)

        elif method == "approx":
            if matrix_opt["sparse2dense"]:
                dM = (self.data.toarray() + matrix_opt[
                    "epsilon"] * direction.data.toarray()) * tau
                dprop = la.expm(dM)
                prop = self.exp(tau)
                prop_grad = (dprop - prop) * (1 / matrix_opt["epsilon"])
            else:
                dM = (self.data + matrix_opt["epsilon"] * direction.data) * tau
                dprop = sp_expm(dM, sparse=matrix_opt["sparse_exp"])
                prop = self.exp(tau)
                prop_grad = (dprop - prop) * (1 / matrix_opt["epsilon"])

        elif method == "first_order":
            if compute_expm:
                prop = self.exp(tau)
            prop_grad = direction.data * tau

        elif method == "second_order":
            if compute_expm:
                prop = self.exp(tau)
            prop_grad = direction.data * tau
            prop_grad += (
                                 self.data * direction.data + direction.data * self.data) \
                         * (tau * tau * 0.5)

        elif method == "third_order":
            if compute_expm:
                prop = self.exp(tau)
            prop_grad = direction.data * tau
            A = self.data * direction.data
            B = direction.data * self.data
            prop_grad += (A + B) * (tau * tau * 0.5)
            prop_grad += (self.data * A + A * self.data + B * self.data) * \
                         (tau * tau * tau * 0.16666666666666666)

        if compute_expm:
            if matrix_opt["sparse2dense"]:
                return OperatorDense(prop), OperatorDense(prop_grad)
            else:
                return OperatorSparse(prop), OperatorSparse(prop_grad)
        else:
            if matrix_opt["sparse2dense"]:
                return OperatorDense(prop_grad)
            else:
                return OperatorSparse(prop_grad)


def convert_unitary_to_super_operator(unitary):
    """
    We assume that the unitary U shall be used to propagate a density matrix m
    like
        U m U^dag
    which is equivalent to
        ( U^\ast \otimes U) \vec{m}
    :param unitary:
    :return:

    """
    if type(unitary) in (OperatorDense, OperatorSparse):
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
