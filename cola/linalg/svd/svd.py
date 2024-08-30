import numpy as np
from plum import dispatch
from cola.annotations import Unitary
from cola.ops.operator_base import LinearOperator
from cola.ops.operators import Diagonal
from cola.ops.operators import I_like
from cola.ops.operators import Identity
from cola.linalg.algorithm_base import Algorithm, Auto
from cola.linalg.decompositions.decompositions import SVD, LanczosSVD
from cola.utils import export


@export
@dispatch.abstract
def svd(A: LinearOperator, alg: Algorithm = Auto()):
    """
    Computes the SVD of the linear operator A.

    Args:
        A (LinearOperator): The linear operator to decompose.
        alg (Algorithm): (Auto, SVD, LanczosSVD)

    Returns:
        Tuple[LinearOperator]: A tuple U, D, V, such that U D V^{*} = A.
    """


@dispatch(precedence=-1)
def svd(A: LinearOperator, alg: Auto):
    """ Auto:
        - if A is small, use dense SVD
        - if A is large, use dense Lanczos SVD
    """
    if not hasattr(alg, "method"):
        alg.method = "dense" if bool(np.prod(A.shape) <= 1e6) else "iterative"
    match alg.method:
        case "dense":
            alg = SVD()
        case "iterative":
            alg.__dict__.pop("method")
            alg = LanczosSVD(**alg.__dict__)
    return svd(A, alg)


@dispatch
def svd(A: LinearOperator, alg: SVD):
    U, Sigma, V = alg(A)
    return U, Sigma, V


@dispatch
def svd(A: LinearOperator, alg: LanczosSVD):
    U, Sigma, V = alg(A)
    return U, Sigma, V


@dispatch
def svd(A: Identity, alg: Algorithm):
    ones = A.xnp.ones(A.shape[0], device=A.device, dtype=A.dtype)
    return Unitary(I_like(A)), Diagonal(ones), Unitary(I_like(A))


@dispatch
def svd(A: Diagonal, alg: Algorithm):
    return Unitary(I_like(A)), A, Unitary(I_like(A))
