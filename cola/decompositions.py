from typing import Callable
import cola
from cola import Unitary
from cola.fns import lazify
from cola.ops.operator_base import LinearOperator
from cola.utils import export
from cola.linalg import inverse, eig, trace, apply_unary


@export
class UnitaryDecomposition(LinearOperator):
    """ Decomposition of form Q A Q^H.

    Convenient for computing inverses, eigs, traces, determinants.
    Assumes Q is unitary (or more precisely a Stiefel matrix): Q.H@Q = I,
    but not necessarily Q@Q.H = I and Q need not be square.
    """
    def __init__(self, Q, HH):
        super().__init__(HH.dtype, HH.shape)
        self.Q = cola.fns.lazify(Q)
        self.HH = cola.fns.lazify(HH)


@inverse.dispatch
def inverse(A: UnitaryDecomposition, **kwargs):
    Q, A = A.Q, A.A
    return Q @ inverse(A, **kwargs) @ Q.H


@eig.dispatch
def eig(QH: UnitaryDecomposition, **kwargs):
    Q, H, xnp = QH.Q, QH.HH, QH.ops
    eig_vals, eig_vecs = eig(H, **kwargs)
    eig_vecs = xnp.cast(Q.to_dense(), dtype=eig_vecs.dtype) @ eig_vecs.to_dense()
    eig_vecs = Unitary(lazify(eig_vecs))
    return eig_vals, eig_vecs


@trace.dispatch
def trace(A: UnitaryDecomposition, **kwargs):
    Q, A = A.Q, A.A
    return trace(A, **kwargs)


@apply_unary.dispatch
def apply_unary(fn: Callable, A: UnitaryDecomposition, **kwargs):
    # Need to think carefully about the case where Q is not full rank
    Q, A = A.Q, A.A
    return Q @ apply_unary(fn, A, **kwargs) @ Q.H
