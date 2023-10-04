""" Decompositions of linear operators, e.g. LU, Cholesky, Lanczos, Arnoldi, SVD"""

from typing import Callable
from plum import dispatch
import cola
from cola import Unitary
from cola.fns import lazify
from cola.ops.operator_base import LinearOperator
from cola.ops import Triangular, Permutation, Diagonal
from cola.ops import Identity, ScalarMul, Kronecker, BlockDiag
from cola.utils import export
# from cola.linalg import inv, eig, trace, apply_unary
import cola.linalg
from cola.linalg.algorithm_base import Algorithm, Auto


@export
class Cholesky(Algorithm):
    """ TODO: docstring"""
    def __call__(self, A: LinearOperator):
        return cholesky(A)


@export
class LU(Algorithm):
    """ TODO: docstring"""
    def __call__(self, A: LinearOperator):
        return plu(A)


@dispatch
@export
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
    return cola.linalg.sqrt(A)


@dispatch
def cholesky(A: Kronecker):
    # see https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/kronthesisschaecke04.pdf
    return Kronecker(*[cholesky(Ai) for Ai in A.Ms])


@dispatch
def cholesky(A: BlockDiag):
    return BlockDiag(*[cholesky(Ai) for Ai in A.Ms], multiplicities=A.multiplicities)


@dispatch
@export
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
