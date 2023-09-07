from typing import Callable
from plum import dispatch
import cola
from cola import Unitary
from cola.fns import lazify
from cola.ops.operator_base import LinearOperator
from cola.ops import Triangular, Permutation, Diagonal
from cola.ops import Identity, ScalarMul, Kronecker, BlockDiag
from cola.utils import export
from cola.linalg import inv, eig, trace, apply_unary


@dispatch
@export
def cholesky_decomposed(A: LinearOperator):
    """ Performs a cholesky decomposition A=LL* of a linear operator A.
        The returned operator LL* is the same as A, but represented using
        the triangular structure.

        (Implicitly assumes A is PSD)
    """
    L = Triangular(A.xnp.cholesky(A.to_dense()), lower=True)
    return L @ L.H


@dispatch
def cholesky_decomposed(A: Identity):
    return A


@dispatch
def cholesky_decomposed(A: Diagonal):
    return A


@dispatch
def cholesky_decomposed(A: ScalarMul):
    return A


@dispatch
def cholesky_decomposed(A: Kronecker):
    # see https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/kronthesisschaecke04.pdf
    return Kronecker(*[cholesky_decomposed(Ai) for Ai in A.Ms])


@dispatch
def cholesky_decomposed(A: BlockDiag):
    return BlockDiag(*[cholesky_decomposed(Ai) for Ai in A.Ms], multiplicities=A.multiplicities)


@dispatch
@export
def lu_decomposed(A: LinearOperator):
    """ Performs a cholesky decomposition A=PLU of a linear operator A.
        The returned operator PLU is the same as A, but represented using
        the triangular (and permutation) structure """
    p, L, U = A.xnp.lu(A.to_dense())
    P, L, U = Permutation(p), Triangular(L, lower=True), Triangular(U, lower=False)
    P, L, U = P.to(A.device), L.to(A.device), U.to(A.device)
    return P @ L @ U


@dispatch
def lu_decomposed(A: Identity):
    return A


@dispatch
def lu_decomposed(A: Diagonal):
    return A


@dispatch
def lu_decomposed(A: ScalarMul):
    return A


@dispatch
def lu_decomposed(A: Kronecker):
    # see https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/kronthesisschaecke04.pdf
    return Kronecker(*[lu_decomposed(Ai) for Ai in A.Ms])


@dispatch
def lu_decomposed(A: BlockDiag):
    return BlockDiag(*[lu_decomposed(Ai) for Ai in A.Ms], multiplicities=A.multiplicities)


@export
class UnitaryDecomposition(LinearOperator):
    """ Decomposition of form A = Q M Q^H.

    Convenient for computing inverses, eigs, traces, determinants.
    Assumes Q is unitary (or more precisely a Stiefel matrix): Q.H@Q = I,
    but not necessarily Q@Q.H = I and Q need not be square.
    """
    def __init__(self, Q, M):
        super().__init__(M.dtype, (Q.shape[0], Q.shape[0]))
        self.Q = cola.fns.lazify(Q)
        self.M = cola.fns.lazify(M)


@inv.dispatch
def inv(A: UnitaryDecomposition, **kwargs):
    Q, M = A.Q, A.M
    return Q @ inv(M, **kwargs) @ Q.H


@eig.dispatch
def eig(QH: UnitaryDecomposition, **kwargs):
    Q, H, xnp = QH.Q, QH.M, QH.xnp
    eig_vals, eig_vecs = eig(H, **kwargs)
    eig_vecs = xnp.cast(Q.to_dense(), dtype=eig_vecs.dtype) @ eig_vecs.to_dense()
    eig_vecs = Unitary(lazify(eig_vecs))
    return eig_vals, eig_vecs


@trace.dispatch
def trace(A: UnitaryDecomposition, **kwargs):
    _, M = A.Q, A.M
    return trace(M, **kwargs)


@apply_unary.dispatch
def apply_unary(fn: Callable, A: UnitaryDecomposition, **kwargs):
    # Need to think carefully about the case where Q is not full rank
    Q, M = A.Q, A.M
    return Q @ apply_unary(fn, M, **kwargs) @ Q.H
