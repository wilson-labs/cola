import numpy as np
from plum import dispatch

import cola
from cola.linalg.algorithm_base import Algorithm, Auto, IterativeOperatorWInfo
from cola.linalg.inverse.cg import CG
from cola.ops.operators import Diagonal, Identity, LinearOperator, Permutation, ScalarMul, I_like
from cola.annotations import PSD
from cola.utils import export
from cola.utils.utils_linalg import get_precision

_SIZE = 1e6


class LSTSQ(Algorithm):
    """
    Least-squares algorithm for computing the solve of a linear equation
    """
    def __call__(self, A: LinearOperator):
        return LSTSQSolve(A)


class LSTSQSolve(LinearOperator):
    def __init__(self, A: LinearOperator):
        super().__init__(A.dtype, (A.shape[-1], A.shape[-2]))
        self.A = A.to_dense()

    def _matmat(self, X):
        return self.xnp.lstsq(self.A, X)


@export
@dispatch.abstract
def pinv(A: LinearOperator, alg: Algorithm = Auto()):
    """(lazily) computes the pseudo-inverse of a linear operator.

    Args:
        A (LinearOperator): The linear operator to compute the inverse of.
        alg (Algorithm, optional): The algorithm to use for the solves.

    Returns:
        LinearOperator: The pseudo-inverse of the linear operator.

    Example:
        >>> A = MyLinearOperator()
        >>> x = cola.pseudo(A) @ b

    """


@export
@dispatch(precedence=-1)
def pinv(A: LinearOperator, alg: Auto):
    """ Auto:
        - if A is small, use dense algorithms
        - if A is large, use iterative algorithms
    """
    match bool(np.prod(A.shape) <= _SIZE):
        case True:
            alg = LSTSQ()
        case False:
            alg = CG(**alg.__dict__)
    return pinv(A, alg)


@dispatch
def pinv(A: LinearOperator, alg: CG):
    xnp = A.xnp
    M = A.H @ A
    cons = get_precision(xnp, A.dtype) * xnp.sqrt(cola.eigmax(M))
    Op = IterativeOperatorWInfo(M, alg)
    return PSD(Op + cons * I_like(M)) @ A.H


@dispatch
def pinv(A: LinearOperator, alg: LSTSQ):
    return LSTSQSolve(A)


@dispatch
def pinv(A: Identity, alg: Algorithm):
    return A


@dispatch
def pinv(A: ScalarMul, alg: Algorithm):
    return ScalarMul(1 / A.c, shape=A.shape, dtype=A.dtype)


@dispatch
def pinv(A: Diagonal, alg: Algorithm):
    return Diagonal(1. / A.diag)


@dispatch
def pinv(A: Permutation, alg: Algorithm):
    return Permutation(A.xnp.argsort(A.perm), A.dtype)
