from typing import Tuple
from math import prod
from cola.basic_operations import densify
from cola.basic_operations import diag
from cola.ops import LinearOperator
from cola.ops import Array
from cola.ops import SelfAdjoint
from cola.ops import LowerTriangular
from cola.ops import Diagonal
from cola.ops import I_like
from cola.algorithms.lanczos import lanczos_eig
from cola.algorithms.arnoldi import arnoldi_eig
from cola.algorithms import power_iteration
from plum import dispatch
from cola.utils import export

@export
@dispatch
def eig(A: LinearOperator, eig_slice=slice(0, None, None), tol=1e-6, pbar=False, method='auto',
        info=False, max_iters=1000) -> Tuple[Array, Array]:
    xnp = A.ops
    if method == 'dense' or (method == 'auto' and prod(A.shape) < 1e6):
        eig_vals, eig_vecs = xnp.eig(A.to_dense())
        return eig_vals[eig_slice], eig_vecs[:, eig_slice]
    elif method == 'arnoldi' or (method == 'auto' and prod(A.shape) >= 1e6):
        rhs = xnp.randn(A.shape[1], 1, dtype=A.dtype)
        eig_vals, eig_vecs, Q = arnoldi_eig(A=A, rhs=rhs, max_iters=max_iters, tol=tol)
        return eig_vals[eig_slice], eig_vecs[:, eig_slice], Q[:, eig_slice]
    else:
        raise ValueError(f"Unknown method {method}")


@dispatch
def eig(A: LowerTriangular, eig_slice=slice(0, None, None), method="dense", *args, **kwargs):
    xnp = A.ops
    if method == "dense":
        eig_vals = diag(A.A)[eig_slice]
        eig_vecs = xnp.eye(eig_vals.shape[0], eig_vals.shape[0])
        return eig_vals, eig_vecs
    else:
        raise ValueError(f"Unknown method {method}")


@dispatch
def eig(A: SelfAdjoint, eig_slice=slice(0, None, None), tol=1e-6, pbar=False, method='auto',
        info=False, max_iters=1000) -> Tuple[Array, Array]:
    xnp = A.ops
    if method == 'dense' or (method == 'auto' and prod(A.shape) < 1e6):
        eig_vals, eig_vecs = xnp.eigh(A.to_dense())
    elif method == 'lanczos' or (method == 'auto' and prod(A.shape) >= 1e6):
        rhs = xnp.randn(A.shape[1], 1, dtype=A.dtype)
        eig_vals, eig_vecs = lanczos_eig(A, rhs, max_iters=max_iters, tol=tol)
    else:
        raise ValueError(f"Unknown method {method}")
    return eig_vals[eig_slice], eig_vecs[:, eig_slice]


@dispatch
def eig(A: Diagonal, eig_slice=slice(0, None, None), **kwargs):
    xnp = A.ops
    eig_vecs = I_like(A).to_dense()
    sorted_ind = xnp.argsort(A.diag)
    eig_vals = A.diag[sorted_ind]
    eig_vecs = eig_vecs[:, sorted_ind]
    return eig_vals[eig_slice], eig_vecs[:, eig_slice]


def eigenvalues(A: LinearOperator, info=False, pbar=False):
    pass

@export
def eigmax(A: LinearOperator, tol=1e-7, max_iters=1000, pbar=False, info=False):
    """ Returns eigenvalue with largest magnitude of A
        up to specified tolerance tol."""
    return power_iteration(A, tol=tol, max_iter=max_iters, pbar=pbar, info=info)

#@export
def eigmin(A: LinearOperator, tol=1e-7):
    """ Returns eigenvalue with smallest magnitude of A
        up to specified tolerance tol."""
    raise NotImplementedError


