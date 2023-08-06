from typing import Callable
import cola
from cola import Unitary
from cola.fns import lazify
from cola.ops.operator_base import LinearOperator
from cola.ops import Triangular, Permutation
from cola.utils import export
from cola.linalg import inverse, eig, trace, apply_unary

@export
def cholesky_decomposed(A: LinearOperator):
    """ Performs a cholesky decomposition A=LL* of a linear operator A.
        The returned operator LL* is the same as A, but represented using
        the triangular structure """
    L = Triangular(A.ops.cholesky(A.to_dense()), lower=True)
    return L@L.H

@export
def lu_decomposed(A: LinearOperator):
    """ Performs a cholesky decomposition A=PLU of a linear operator A.
        The returned operator PLU is the same as A, but represented using
        the triangular (and permutation) structure """
    p, L, U = A.ops.lu(A.to_dense())
    P, L, U = Permutation(p), Triangular(L,lower=True), Triangular(U, lower=False)
    P, L, U = P.to(A.device), L.to(A.device), U.to(A.device)
    return P@L@U

@export
class UnitaryDecomposition(LinearOperator):
    """ Decomposition of form A = Q M Q^H.

    Convenient for computing inverses, eigs, traces, determinants.
    Assumes Q is unitary (or more precisely a Stiefel matrix): Q.H@Q = I,
    but not necessarily Q@Q.H = I and Q need not be square.
    """
    def __init__(self, Q, M):
        super().__init__(M.dtype, (Q.shape[0],Q.shape[0]))
        self.Q = cola.fns.lazify(Q)
        self.M = cola.fns.lazify(M)


@inverse.dispatch
def inverse(A: UnitaryDecomposition, **kwargs):
    Q, M = A.Q, A.M
    return Q @ inverse(M, **kwargs) @ Q.H


@eig.dispatch
def eig(QH: UnitaryDecomposition, **kwargs):
    Q, H, xnp = QH.Q, QH.M, QH.ops
    eig_vals, eig_vecs = eig(H, **kwargs)
    eig_vecs = xnp.cast(Q.to_dense(), dtype=eig_vecs.dtype) @ eig_vecs.to_dense()
    eig_vecs = Unitary(lazify(eig_vecs))
    return eig_vals, eig_vecs


@trace.dispatch
def trace(A: UnitaryDecomposition, **kwargs):
    Q, M = A.Q, A.M
    return trace(M, **kwargs)


@apply_unary.dispatch
def apply_unary(fn: Callable, A: UnitaryDecomposition, **kwargs):
    # Need to think carefully about the case where Q is not full rank
    Q, M = A.Q, A.M
    return Q @ apply_unary(fn, M, **kwargs) @ Q.H
