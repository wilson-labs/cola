import numpy as np
from plum import dispatch

from cola.annotations import Unitary
from cola.fns import lazify
from cola.linalg.algorithm_base import Algorithm, Auto
from cola.linalg.decompositions.decompositions import Lanczos, get_slice
from cola.linalg.decompositions.lanczos import lanczos_eigs
from cola.linalg.eig.lobpcg import LOBPCG, lobpcg
from cola.linalg.inverse.inv import inv
from cola.ops.operator_base import LinearOperator
from cola.ops.operators import (
    BlockDiag,
    Dense,
    Diagonal,
    I_like,
    Identity,
    Kronecker,
    Permutation,
    ScalarMul,
)
from cola.utils import export


@export
class DenseSVD(Algorithm):
    """
    Performs SVD on A.
    """


@export
@dispatch.abstract
def svd(A: LinearOperator, k: int, which: str = "LM", alg: Algorithm = Auto()):
    """
    Computes the SVD of the linear operator A.

    Args:
        A (LinearOperator): The linear operator to decompose.
        alg (Algorithm): (Auto, SVD, LanczosSVD)

    Returns:
        Tuple[LinearOperator]: A tuple U, D, V, such that U D V^{*} = A.
    """


@dispatch(precedence=-1)
def svd(A: LinearOperator, k: int, which: str, alg: Auto):
    """ Auto:
        - if A is small, use dense SVD
        - if A is large, use Lanczos
    """
    match bool(np.prod(A.shape) <= 1e6):
        case True:
            alg = DenseSVD()
        case False:
            alg = Lanczos(**alg.__dict__)
    return svd(A, k, which, alg)


@dispatch
def svd(A: LinearOperator, k: int, which: str, alg: DenseSVD):
    U, Sigma, V = A.xnp.svd(A.to_dense(), full_matrices=True)
    idx = A.xnp.argsort(Sigma, axis=-1)
    return Unitary(Dense(U[:, idx])), Diagonal(Sigma[..., idx]), Unitary(Dense(V[:, idx]))


@dispatch
def svd(A: LinearOperator, k: int, which: str, alg: Lanczos):
    xnp = A.xnp
    eig_slice = get_slice(k, which)
    if A.shape[1] <= A.shape[0]:
        eig_vals, V, _ = lanczos_eigs(A.H @ A, **alg.__dict__)
        V = Unitary(V[:, eig_slice])
        Sigma = Diagonal(xnp.sqrt(eig_vals[eig_slice]))
        U = Unitary(lazify((A @ V @ inv(Sigma)).to_dense()))
    else:
        eig_vals, U, _ = lanczos_eigs(A @ A.H, **alg.__dict__)
        U = Unitary(U[:, eig_slice])
        Sigma = Diagonal(xnp.sqrt(eig_vals[eig_slice]))
        V = Unitary(lazify((inv(Sigma) @ U.H @ A).to_dense().conj().T))
    return U, Sigma, V


@dispatch
def svd(A: LinearOperator, k: int, which: str, alg: LOBPCG):
    xnp = A.xnp
    eig_slice = get_slice(k, which)
    eig_vals, V = lobpcg(A.H @ A, **alg.__dict__)
    V = Unitary(V[:, eig_slice])
    Sigma = Diagonal(xnp.sqrt(eig_vals[eig_slice]))
    U = Unitary(lazify((A @ V @ inv(Sigma)).to_dense()))
    return U, Sigma, V


@dispatch
def svd(A: Identity, k: int, which: str, alg: Algorithm):
    ones = A.xnp.ones(A.shape[0], device=A.device, dtype=A.dtype)
    return Unitary(I_like(A)), Diagonal(ones), Unitary(I_like(A))


@dispatch
def svd(A: Diagonal, k: int, which: str, alg: Algorithm):
    return Unitary(I_like(A)), A, Unitary(I_like(A))


@dispatch(precedence=1)
def svd(A: ScalarMul, k: int, which: str, alg: Algorithm):
    ones = A.xnp.ones((int(A.shape[0]), ), dtype=A.dtype, device=A.device)
    phase = A.c / A.xnp.abs(A.c)
    U = Diagonal(phase * ones)
    Sigma = Diagonal(A.xnp.abs(A.c) * ones)
    V = Diagonal(ones)
    return U, Sigma, V


@dispatch(precedence=1)
def svd(A: Permutation, k: int, which: str, alg: Algorithm):
    ones = A.xnp.ones((int(A.shape[0]), ), dtype=A.dtype, device=A.device)
    U = A  # Permutation is unitary
    Sigma = Diagonal(ones)
    V = Diagonal(ones)
    return U, Sigma, V


@dispatch(precedence=1)
def svd(A: BlockDiag, k: int, which: str, alg: Algorithm):
    # U = BlockDiag(Ui), S = Diagonal(concat(si)), V = BlockDiag(Vi)
    results = [svd(M, k, which, alg) for M in A.Ms]
    Us, Ss, Vs = zip(*results)
    return BlockDiag(*Us), Diagonal(A.xnp.concat([S.diag for S in Ss])), BlockDiag(*Vs)


@dispatch(precedence=1)
def svd(A: Kronecker, k: int, which: str, alg: Algorithm):
    # A = (U1 ⊗ U2)(S1 ⊗ S2)(V1 ⊗ V2)^H
    results = [svd(M, k, which, alg) for M in A.Ms]
    Us, Ss, Vs = zip(*results)
    return Kronecker(*Us), Kronecker(*Ss), Kronecker(*Vs)
