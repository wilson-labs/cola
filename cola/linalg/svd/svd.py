import numpy as np
from plum import dispatch

from cola.annotations import Unitary
from cola.fns import lazify
from cola.linalg.algorithm_base import Algorithm, Auto
from cola.linalg.decompositions.decompositions import Lanczos
from cola.linalg.inverse.inv import inv
from cola.ops.operator_base import LinearOperator
from cola.ops.operators import Dense, Diagonal, I_like, Identity
from cola.utils import export


@export
class DenseSVD(Algorithm):
    """
    Performs SVD on A.
    """
    def __call__(self, A: LinearOperator):
        return svd(A)


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
        - if A is large, use Lanczos
    """
    match bool(np.prod(A.shape) <= 1e6):
        case True:
            alg = DenseSVD()
        case False:
            alg = Lanczos(**alg.__dict__)
    return svd(A, alg)


@dispatch
def svd(A: LinearOperator, alg: DenseSVD):
    U, Sigma, V = A.xnp.svd(A.to_dense(), full_matrices=True)
    idx = A.xnp.argsort(Sigma, axis=-1)
    return Unitary(Dense(U[:, idx])), Diagonal(Sigma[..., idx]), Unitary(Dense(V[:, idx]))


@dispatch
def svd(A: LinearOperator, alg: Lanczos):
    xnp = A.xnp
    Q, T, _ = alg(A.H @ A)
    eigvals, eigvectors = xnp.eigh(T.to_dense())
    idx = xnp.argsort(eigvals, axis=-1)
    V = Unitary(Q @ lazify(eigvectors[:, idx]))
    Sigma = Diagonal(xnp.sqrt(eigvals[..., idx]))
    U = Unitary(lazify((A @ V @ inv(Sigma)).to_dense()))
    return U, Sigma, V


@dispatch
def svd(A: Identity, alg: Algorithm):
    ones = A.xnp.ones(A.shape[0], device=A.device, dtype=A.dtype)
    return Unitary(I_like(A)), Diagonal(ones), Unitary(I_like(A))


@dispatch
def svd(A: Diagonal, alg: Algorithm):
    return Unitary(I_like(A)), A, Unitary(I_like(A))
