import numpy as np
from plum import dispatch
from cola.annotations import Unitary, Stiefel
from cola.fns import lazify
from cola.annotations import SelfAdjoint
from cola.linalg.trace.diag_trace import diag
from cola.ops.operator_base import LinearOperator
from cola.ops.operators import Diagonal
from cola.ops.operators import I_like
from cola.ops.operators import Identity
from cola.ops.operators import Triangular
from cola.linalg.decompositions.lanczos import lanczos_eigs
from cola.linalg.decompositions.arnoldi import arnoldi_eigs
from cola.linalg.algorithm_base import Algorithm, Auto
from cola.linalg.decompositions.decompositions import Arnoldi, Lanczos
from cola.linalg.unary.unary import Eig, Eigh
from cola.utils import export


@export
@dispatch.abstract
def eig(A: LinearOperator, k: int = -1, which: str = 'LM', alg: Algorithm = Auto()):
    """
    Computes eigenvalues and eigenvectors of a linear operator.

    Args:
        A (LinearOperator): The linear operator for which eigenvalues and eigenvectors are computed.
        k (int): The desired number of eigenvalues and eigenvectors. Must be specified.
        which (str): From what part of the spectrum would de eigenvalues be fetched.
         Default is 'LM' (largest in magnitude) but alternatively you can use 'SM'
         (smallest in magnitude).
        alg (Algorithm): (Auto, Eig, Eigh, Arnoldi, Lanczos)

    Returns:
        Tuple[Array, LinearOperator]: A tuple containing eigenvalues and eigenvectors.
         The eigenvalues are given by eig_vals and the eigenvectors are given
         by eig_vecs.

    Example:
        >>> A = MyLinearOperator()
        >>> eig_vals, eig_vecs = eig(A, k=6, which='LM', alg=Auto(tol=1e-4))
    """


@dispatch
def eig(A: LinearOperator, k: int, which: str, alg: Arnoldi):
    eig_slice = get_slice(k, which)
    eig_vals, eig_vecs, _ = arnoldi_eigs(A, **alg.__dict__)
    return eig_vals[eig_slice], eig_vecs[:, eig_slice]


@dispatch
def eig(A: LinearOperator, k: int, which: str, alg: Lanczos):
    assert A.isa(SelfAdjoint), "Lanczos only valid for SelfAdjoint, wrap in cola.SelfAdjoint if desired"
    eig_slice = get_slice(k, which)
    eig_vals, eig_vecs, _ = lanczos_eigs(A, **alg.__dict__)
    return eig_vals[eig_slice], eig_vecs[:, eig_slice]


@dispatch(precedence=-1)
def eig(A: LinearOperator, k: int, which: str, alg: Auto):
    """ Auto:
        - if A is Hermitian and small, use Eigh
        - if A is Hermitian and large, use Lanczos
        - if A is not Hermitian and small, use Eig
        - if A is not Hermitian and large, use Arnoldi
    """
    match (A.isa(SelfAdjoint), bool(np.prod(A.shape) <= 1e6)):
        case (True, True):
            algorithm = Eigh()
        case (True, False):
            algorithm = Lanczos(**alg.__dict__)
        case (False, True):
            algorithm = Eig()
        case (False, False):
            algorithm = Arnoldi(**alg.__dict__)
    return eig(A, k, which, algorithm)


@dispatch(precedence=-1)
def eig(A: LinearOperator, k: int, which: str, alg: Eig):
    eig_slice = get_slice(k, which)
    eig_vals, eig_vecs = A.xnp.eig(A.to_dense())
    return eig_vals[eig_slice], lazify(eig_vecs[:, eig_slice])


@dispatch(precedence=-1)
def eig(A: LinearOperator, k: int, which: str, alg: Eigh):
    eig_slice = get_slice(k, which)
    eig_vals, eig_vecs = A.xnp.eigh(A.to_dense())
    return eig_vals[eig_slice], Stiefel(lazify(eig_vecs[:, eig_slice]))


@dispatch
def eig(A: Identity, k: int, which: str, alg: Algorithm):
    eig_slice = get_slice(k, which)
    xnp = A.xnp
    eig_vals = xnp.ones(shape=(A.shape[0], ), dtype=A.dtype, device=A.device)
    eig_vecs = A.to_dense()[:, xnp.arange(A.shape[0] - 1, -1, -1)]
    return eig_vals[eig_slice], Unitary(lazify(eig_vecs[:, eig_slice]))


@dispatch
def eig(A: Triangular, k: int, which: str, alg: Algorithm):
    # TODO: take out compute_lower_triangular_eigvecs
    eig_slice = get_slice(k, which)
    xnp = A.xnp
    eig_vals = diag(A)
    sorted_ind = xnp.argsort(eig_vals)
    eig_vals = eig_vals[sorted_ind]
    eig_vecs = compute_lower_triangular_eigvecs(np.array(A.A))
    eig_vecs = xnp.array(eig_vecs, dtype=A.dtype, device=A.device)[:, sorted_ind]
    return eig_vals[eig_slice], Unitary(lazify(eig_vecs[:, eig_slice]))


def compute_lower_triangular_eigvecs(L):
    eigvals = np.diag(L)
    eigvecs = np.eye(L.shape[0])
    for i in range(1, L.shape[0]):
        A = L[:i, :i] - eigvals[i] * np.eye(i)
        out = np.linalg.solve(A, -L[:i, i])
        eigvecs[:i, i] = out
    return eigvecs


@dispatch
def eig(A: Diagonal, k: int, which: str, alg: Algorithm):
    eig_slice = get_slice(k, which)
    xnp = A.xnp
    sorted_ind = xnp.argsort(A.diag)
    eig_vals = A.diag[sorted_ind]
    eig_vecs = I_like(A).to_dense()[:, sorted_ind]
    return eig_vals[eig_slice], Unitary(lazify(eig_vecs[:, eig_slice]))


def get_slice(num, which):
    if num == -1:
        raise ValueError(f"Number of eigenvalues {num} must be explicitly specified")
    if which == "SM":
        eig_slice = slice(0, num, None)
    elif which == "LM":
        id = -1 if num is None else -num
        eig_slice = slice(id, None, None)
    else:
        raise NotImplementedError(f"which={which} is not implemented")
    return eig_slice
