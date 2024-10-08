""" Decompositions of linear operators, e.g. LU, Cholesky, Lanczos, Arnoldi, SVD"""
from dataclasses import dataclass
from typing import Any, Optional

from plum import dispatch

import cola.linalg
from cola.annotations import Unitary
from cola.linalg.algorithm_base import Algorithm, Auto
from cola.linalg.decompositions.arnoldi import arnoldi
from cola.linalg.decompositions.lanczos import lanczos, lanczos_eigs
from cola.ops import (
    Array,
    BlockDiag,
    Diagonal,
    Identity,
    Kronecker,
    LinearOperator,
    Permutation,
    ScalarMul,
    Triangular,
)
from cola.utils import export

PRNGKey = Any


@export
class Cholesky(Algorithm):
    """
    Cholesky algorithm for decomposing a positive definite operator as
    :math:`A = L L^{*}`,
    where :math:`L` is a lower triangular operator.

    Example:
        >>> A = MyLinearOperator()
        >>> L = cola.linalg.decompositions.Cholesky()(A)
    """
    def __call__(self, A: LinearOperator):
        return cholesky(A)


@export
class LU(Algorithm):
    """
    LU algorithm for decomposing a general square operator as
    :math:`A = PLU`,
    where :math:`P` is a permutation operator,
    :math:`L` is a lower triangular operator
    and :math:`U` is an upper triangular operator.

    Example:
        >>> A = MyLinearOperator()
        >>> P,L,U = cola.linalg.decompositions.LU()(A)
    """
    def __call__(self, A: LinearOperator):
        return plu(A)


@export
@dataclass
class Arnoldi(Algorithm):
    """
    Arnoldi decomposition for a general square operator,
    :math:`H \\approx Q^{*} A Q`
    where :math:`H` is an upper Hessenberg operator.
    This algorithm is used to approximate eig(A) through eig(H).

    Args:
        start_vector (Array, optional): (n,) or (n, b) vector to start the algorithm.
        max_iters (int, optional): The maximum number of iterations to run.
        tol (float, optional): Relative error tolerance.
        pbar (bool, optional): Whether to show progress bar.
        key (PRNGKey, optional): Random key to use for the algorithm.
            PRNGKey for jax, long integer for numpy or pytorch.

    Example:
        >>> A = MyLinearOperator()
        >>> Q,H,info = Arnoldi(max_iters=100,pbar=True)(A)
    """
    start_vector: Array = None
    max_iters: int = 1_000
    tol: float = 1e-6
    pbar: bool = False
    key: Optional[PRNGKey] = None

    def __call__(self, A: LinearOperator):
        return arnoldi(A, **self.__dict__)


@dataclass
class LanczosSVD(Algorithm):
    """
    Does the SVD using the Lanczos decomposition,
    Args:
        start_vector (Array, optional): (n,) or (n, b) vector to start the algorithm.
        max_iters (int, optional): The maximum number of iterations to run.
        tol (float, optional): Relative error tolerance.
        pbar (bool, optional): Whether to show progress bar.
        key (PRNGKey, optional): Random key to use for the algorithm.
            PRNGKey for jax, long integer for numpy or pytorch.
    """
    start_vector: Array = None
    max_iters: int = 1_000
    tol: float = 1e-6
    pbar: bool = False
    key: Optional[PRNGKey] = None

    def __call__(self, A: LinearOperator):
        Sigma0_2, V, _ = lanczos_eigs(A.H @ A, **self.__dict__)
        Sigma1_2, U, _ = lanczos_eigs(A @ A.H, **self.__dict__)
        Sigma = A.xnp.sqrt(Sigma0_2)
        return Unitary(U), Diagonal(Sigma), Unitary(V)


@export
@dataclass
class Lanczos(Algorithm):
    """
    Lanczos decomposition for Symmetric (or Hermitian) operators,
    :math:`T = Q^{*} A Q`
    where :math:`T` is a tridiagional operator.
    This algorithm is used to approximate eig(A) through eig(T).

    Args:
        start_vector (Array, optional): (n,) or (n, b) vector to start the algorithm.
        max_iters (int, optional): The maximum number of iterations to run.
        tol (float, optional): Relative error tolerance.
        pbar (bool, optional): Whether to show progress bar.
        key (PRNGKey, optional): Random key to use for the algorithm.
            PRNGKey for jax, long integer for numpy or pytorch.

    Example:
        >>> A = MyLinearOperator()
        >>> Q,T,info = Lanczos(max_iters=100,pbar=True)(A)
    """
    start_vector: Array = None
    max_iters: int = 1_000
    tol: float = 1e-6
    pbar: bool = False
    key: Optional[PRNGKey] = None

    def __call__(self, A: LinearOperator):
        return lanczos(A, **self.__dict__)


@dispatch
def cholesky(A: LinearOperator):
    """ Performs a cholesky decomposition A=LL* of a linear operator A.
        The returned operator L with triangular structure.

        (Implicitly assumes A is PSD)
    """
    return Triangular(A.xnp.cholesky(A.to_dense()), lower=True)


@dispatch
def cholesky(A: Identity):
    return A


@dispatch
def cholesky(A: Diagonal | ScalarMul):
    return cola.linalg.sqrt(A, Auto())


@dispatch
def cholesky(A: Kronecker):
    # see https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/kronthesisschaecke04.pdf
    return Kronecker(*[cholesky(Ai) for Ai in A.Ms])


@dispatch
def cholesky(A: BlockDiag):
    return BlockDiag(*[cholesky(Ai) for Ai in A.Ms], multiplicities=A.multiplicities)


@dispatch
def plu(A: LinearOperator):
    """ Performs a cholesky decomposition A=PLU of a linear operator A.
        The returns P, L, U as linear operators, but represented using
        the triangular (and permutation) structure """
    p, L, U = A.xnp.lu(A.to_dense())
    P, L, U = Permutation(p), Triangular(L, lower=True), Triangular(U, lower=False)
    P, L, U = P.to(A.device), L.to(A.device), U.to(A.device)
    return P, L, U


@dispatch
def plu(A: Identity):
    return A, A, A


@dispatch
def plu(A: Diagonal | ScalarMul):
    S = cola.linalg.sqrt(A)
    return cola.ops.I_like(A), S, S


@dispatch
def plu(A: Kronecker):
    # see https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/kronthesisschaecke04.pdf
    P, L, U = zip(*[plu(Ai) for Ai in A.Ms])
    return Kronecker(*P), Kronecker(*L), Kronecker(*U)


@dispatch
def plu(A: BlockDiag):
    P, L, U = zip(*[plu(Ai) for Ai in A.Ms])
    BD = lambda *args: BlockDiag(*args, multiplicities=A.multiplicities)  # noqa
    return BD(*P), BD(*L), BD(*U)


def get_slice(num, which):
    if num == -1:
        raise ValueError(f"Number of eigenvalues {num} must be explicitly specified")
    if which == "SM":
        eig_slice = slice(0, num, None)
    elif which == "LM":
        id = -1 if num is None else -num
        eig_slice = slice(id, None, None)
    else:
        raise NotImplementedError(f"which={which} is not implemented")
    return eig_slice
