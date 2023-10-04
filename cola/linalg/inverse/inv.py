import numpy as np
from plum import dispatch
from cola.ops import LinearOperator
from cola.ops import Diagonal, Permutation
from cola.ops import Identity
from cola.ops import ScalarMul
from cola.ops import BlockDiag, Triangular
from cola.ops import Kronecker, Product
from plum import parametric
from cola.utils import export
from cola.annotations import PSD, Unitary
from cola.linalg.algorithm_base import Algorithm, Auto
from cola.linalg.decompositions.decompositions import Cholesky, LU
from cola.linalg.decompositions.decompositions import plu, cholesky


@export
def solve(A, b, alg=Auto()):
    """ Inverse expresse das a solve operation"""
    return inv(A, alg) @ b


@dispatch.abstract
@export
def inv(A: LinearOperator, alg: Algorithm = Auto()):
    """(lazily) computes the inverse of a linear operator, equivalent to solve.

    Args:
        A (LinearOperator): The linear operator to compute the inverse of.
        alg (Algorithm, optional): The algorithm to use for non dispatch base case.

    Returns:
        LinearOperator: The inverse of the linear operator.

    Example:
        >>> A = MyLinearOperator()
        >>> x = cola.inverse(A, alg=Auto(tol=1e-3)) @ b

    """


@dispatch(precedence=-1)
def inv(A: LinearOperator, alg: Auto = Auto()):
    """ Auto:
        - if A is PSD and small, use Cholesky
        - if A is PSD and large, use CG
        - if A is not PSD and small, use LU
        - if A is not PSD and large, use GMRES
    """
    from .cg import CG
    from .gmres import GMRES
    match (A.isa(PSD), bool(np.prod(A.shape) <= 1e6)):
        case (True, True):
            alg = Cholesky()
        case (True, False):
            alg = CG(**alg.__dict__)
        case (False, True):
            alg = LU()
        case (False, False):
            alg = GMRES(**alg.__dict__)
        case _:
            assert False
    return inv(A, alg)


@dispatch(precedence=-1)
def inv(A: LinearOperator, alg: Cholesky):
    L = cholesky(A)
    return inv(L) @ inv(L.H)


@dispatch(precedence=-1)
def inv(A: LinearOperator, alg: LU):
    P, L, U = plu(A)
    return inv(U) @ inv(L) @ inv(P)


@dispatch(cond=lambda A, *_: A.isa(Unitary))
def inv(A: LinearOperator, alg=Auto()):
    return Unitary(A.H)


@dispatch
def inv(A: Identity, alg=Auto()):
    return A


@dispatch
def inv(A: ScalarMul, alg=Auto()):
    return ScalarMul(1 / A.c, shape=A.shape, dtype=A.dtype)


@dispatch
def inv(A: Permutation, alg=Auto()):
    return Permutation(A.xnp.argsort(A.perm), A.dtype)


@dispatch(cond=lambda A, *_: all([M.shape[-2] == M.shape[-1] for M in A.Ms]))
def inv(A: Product, alg=Auto()):
    output = reversed([inv(M, alg) for M in A.Ms])
    return Product(*output)


@dispatch
def inv(A: BlockDiag, alg=Auto()):
    return BlockDiag(*[inv(M, alg) for M in A.Ms], multiplicities=A.multiplicities)


@dispatch
def inv(A: Kronecker, alg=Auto()):
    return Kronecker(*[inv(M, alg) for M in A.Ms])


@dispatch
def inv(A: Diagonal, alg=Auto()):
    return Diagonal(1. / A.diag)


@dispatch
def inv(A: Triangular, alg=Auto()):
    return TriangularInv(A)


@parametric
class TriangularInv(LinearOperator):
    def __init__(self, A: Triangular):
        super().__init__(A.dtype, A.shape)
        self.A = A.to_dense()
        self.lower = A.lower

    def _matmat(self, X):
        return self.xnp.solvetri(self.A, X, lower=self.lower)

    def _rmatmat(self, X):
        return self.xnp.solvetri(self.A.T, X.T, lower=not self.lower).T
