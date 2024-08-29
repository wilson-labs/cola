import numpy as np
from plum import dispatch

from cola.linalg.algorithm_base import Algorithm, Auto, IterativeOperatorWInfo
from cola.linalg.inverse.gmres import LSTSQ_GMRES
from cola.ops.operators import Diagonal, Identity, LinearOperator, Permutation, ScalarMul
from cola.utils import export

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
def pseudo(A: LinearOperator, alg: Algorithm = Auto()):
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
def pseudo(A: LinearOperator, alg: Auto):
    """ Auto:
        - if A is small, use dense algorithms
        - if A is large, use iterative algorithms
    """
    if not hasattr(alg, "method"):
        alg.method = "dense" if bool(np.prod(A.shape)) else "iterative"

    match alg.method:
        case "dense":
            alg = LSTSQ()
        case "iterative":
            alg.__dict__.pop("method")
            alg = LSTSQ_GMRES(**alg.__dict__)
    return pseudo(A, alg)


@dispatch
def pseudo(A: LinearOperator, alg: LSTSQ_GMRES):
    Op = IterativeOperatorWInfo(A, alg)
    Op.shape = (A.shape[-1], A.shape[-2])
    return Op


@dispatch
def pseudo(A: LinearOperator, alg: LSTSQ):
    return LSTSQSolve(A)


@dispatch
def pseudo(A: Identity, alg: Algorithm):
    return A


@dispatch
def pseudo(A: ScalarMul, alg: Algorithm):
    return ScalarMul(1 / A.c, shape=A.shape, dtype=A.dtype)


@dispatch
def pseudo(A: Diagonal, alg: Algorithm):
    return Diagonal(1. / A.diag)


@dispatch
def pseudo(A: Permutation, alg: Algorithm):
    return Permutation(A.xnp.argsort(A.perm), A.dtype)
