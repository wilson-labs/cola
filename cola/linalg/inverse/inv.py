import numpy as np
from plum import parametric
from plum import dispatch
from cola.ops.operators import LinearOperator
from cola.ops.operators import Diagonal, Permutation
from cola.ops.operators import Identity
from cola.ops.operators import ScalarMul
from cola.ops.operators import BlockDiag, Triangular
from cola.ops.operators import Kronecker, Product
from cola.utils import export
from cola.annotations import PSD, Unitary
from cola.linalg.algorithm_base import Algorithm, Auto
from cola.linalg.decompositions.decompositions import Cholesky, LU
from cola.linalg.decompositions.decompositions import plu, cholesky
from cola.linalg.algorithm_base import IterativeOperatorWInfo
from cola.linalg.inverse.cg import CG
from cola.linalg.inverse.gmres import GMRES


@export
def solve(A, b, alg=Auto()):
    """ Computes Linear solve of a linear operator. Equivalent to cola.inv

    Args:
        A (LinearOperator): The linear operator to compute the inverse of.
        b (Array): The right hand side of the linear system of shape (d, ) or (d, k)
        alg (Algorithm, optional): The algorithm to use for the solves.

    Returns:
        Array: The solution of the linear system of shape (d, ) or (d, k)

    Example:
        >>> A = MyLinearOperator()
        >>> x = cola.solve(A, b, alg=Auto(max_iters=10, pbar=True))
    """
    return inv(A, alg) @ b


@export
@dispatch.abstract
def inv(A: LinearOperator, alg: Algorithm = Auto()):
    """(lazily) computes the inverse of a linear operator, equivalent to solve.

    Args:
        A (LinearOperator): The linear operator to compute the inverse of.
        alg (Algorithm, optional): The algorithm to use for the solves.

    Returns:
        LinearOperator: The inverse of the linear operator.

    Example:
        >>> A = MyLinearOperator()
        >>> x = cola.inverse(A, alg=CG(tol=1e-3)) @ b

    """


@dispatch
def inv(A: LinearOperator, alg: GMRES):
    return IterativeOperatorWInfo(A, alg)


@dispatch(precedence=-1)
def inv(A: LinearOperator, alg: CG):
    assert A.isa(PSD), "CG only valid for PSD matrices, wrap in cola.PSD if desired"
    return IterativeOperatorWInfo(A, alg)


@export
@dispatch(precedence=-1)
def inv(A: LinearOperator, alg: Auto):
    """ Auto:
        - if A is PSD and small, use Cholesky
        - if A is PSD and large, use CG
        - if A is not PSD and small, use LU
        - if A is not PSD and large, use GMRES
    """
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
    assert A.isa(PSD), "Cholesky only valid for PSD matrices, wrap in cola.PSD if desired"
    L = cholesky(A)
    return inv(L.H) @ inv(L)


@dispatch(precedence=-1)
def inv(A: LinearOperator, alg: LU):
    P, L, U = plu(A)
    return inv(U) @ inv(L) @ inv(P)


@dispatch(cond=lambda A, *_: A.isa(Unitary))
def inv(A: LinearOperator, alg: Algorithm):
    return Unitary(A.H)


@dispatch
def inv(A: Identity, alg: Algorithm):
    return A


@dispatch
def inv(A: ScalarMul, alg: Algorithm):
    return ScalarMul(1 / A.c, shape=A.shape, dtype=A.dtype)


@dispatch
def inv(A: Permutation, alg: Algorithm):
    return Permutation(A.xnp.argsort(A.perm), A.dtype)


@dispatch(cond=lambda A, *_: all([M.shape[-2] == M.shape[-1] for M in A.Ms]))
def inv(A: Product, alg: Algorithm):
    output = reversed([inv(M, alg) for M in A.Ms])
    return Product(*output)


@dispatch
def inv(A: BlockDiag, alg: Algorithm):
    return BlockDiag(*[inv(M, alg) for M in A.Ms], multiplicities=A.multiplicities)


@dispatch
def inv(A: Kronecker, alg: Algorithm):
    return Kronecker(*[inv(M, alg) for M in A.Ms])


@dispatch
def inv(A: Diagonal, alg: Algorithm):
    return Diagonal(1. / A.diag)


@dispatch
def inv(A: Triangular, alg: Algorithm):
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
