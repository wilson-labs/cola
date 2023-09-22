import numpy as np
from math import prod
from plum import dispatch
from cola import SelfAdjoint
from cola import Unitary, Stiefel
from cola.fns import lazify
from cola.linalg import diag
from cola.ops import LinearOperator
from cola.ops import Diagonal
from cola.ops import I_like
from cola.ops import Identity
from cola.ops import Triangular
from cola.algorithms import power_iteration
from cola.algorithms.lanczos import lanczos_eigs
from cola.algorithms.arnoldi import arnoldi_eigs
from cola.utils import export


@dispatch
@export
def eig(A: LinearOperator, **kwargs):
    """
    Computes eigenvalues and eigenvectors of a linear operator.

    Args:
        A (LinearOperator): The linear operator for which eigenvalues and eigenvectors are computed.
        num (int): The desired number of eigenvalues and eigenvectors.
         Default is None which gets all available.
        which (str): From what part of the spectrum would de eigenvalues be fetched.
         Default is 'LM' (largest in magnitude) but alternatively you can use 'SM'
         (smallest in magnitude).
        tol (float): Optional. Tolerance for convergence. Default is 1e-6.
        pbar (bool): Optional. Whether to display a progress bar during computation.
         Default is False.
        method (str): Optional. Method to use for computation.
         'dense' computes eigenvalues and eigenvectors using dense matrix operations.
         'iterative' computes using Lanczos or Arnoldi iteration. 'auto' automatically selects the
          method based on the size of the linear operator. Default is 'auto'.
        max_iters (int): Optional. Maximum number of iterations for Arnoldi method. Default is 1000.

    Returns:
        Tuple[Array, Array]: A tuple containing eigenvalues and eigenvectors.
         The eigenvalues are given by eig_vals and the eigenvectors are given
         by eig_vecs.

    Example:
        >>> A = MyLinearOperator()
        >>> eig_vals, eig_vecs = eig(A, num=6, which='LM', tol=1e-4)
    """
    kws = dict(num=None, which="LM", tol=1e-6, pbar=False, method='auto', max_iters=1000)
    assert not kwargs.keys() - kws.keys(), f"Unknown kwargs {kwargs.keys()-kws.keys()}"
    kws.update(kwargs)
    method = kws.pop('method')
    eig_slice = get_slice(kws.pop('num'), kws.pop('which'))
    xnp = A.xnp
    if method == 'dense' or (method == 'auto' and prod(A.shape) < 1e6):
        eig_vals, eig_vecs = xnp.eig(A.to_dense())
        return eig_vals[eig_slice], lazify(eig_vecs[:, eig_slice])
    elif method in ('arnoldi', 'iterative') or (method == 'auto' and prod(A.shape) >= 1e6):
        eig_vals, eig_vecs, _ = arnoldi_eigs(A, **kws)
        return eig_vals, eig_vecs
    else:
        raise ValueError(f"Unknown method {method}")


@dispatch(cond=lambda A, **kwargs: A.isa(SelfAdjoint))
def eig(A: LinearOperator, **kwargs):
    kws = dict(num=None, which="LM", tol=1e-6, pbar=False, method='auto', max_iters=1000)
    assert not kwargs.keys() - kws.keys(), f"Unknown kwargs {kwargs.keys()-kws.keys()}"
    kws.update(kwargs)
    method = kws.pop('method')
    eig_slice = get_slice(kws.pop('num'), kws.pop('which'))
    xnp = A.xnp
    if method == 'dense' or (method == 'auto' and prod(A.shape) < 1e6):
        eig_vals, eig_vecs = xnp.eigh(A.to_dense())
        return eig_vals[eig_slice], Stiefel(lazify(eig_vecs[:, eig_slice]))
    elif method in ('lanczos', 'iterative') or (method == 'auto' and prod(A.shape) >= 1e6):
        eig_vals, eig_vecs, _ = lanczos_eigs(A, **kws)
        return eig_vals, eig_vecs
    else:
        raise ValueError(f"Unknown method {method} for SelfAdjoint operator")


@dispatch
def eig(A: Identity, num=None, which="LM", **kwargs):
    eig_slice = get_slice(num, which)
    xnp = A.xnp
    eig_vals = xnp.ones(shape=(A.shape[0], ), dtype=A.dtype, device=A.device)
    eig_vecs = A.to_dense()
    return eig_vals[eig_slice], Unitary(lazify(eig_vecs[:, eig_slice]))


@dispatch
def eig(A: Triangular, num=None, which="LM", **kwargs):
    # TODO: take out compute_lower_triangular_eigvecs
    eig_slice = get_slice(num, which)
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
def eig(A: Diagonal, num=None, which="LM", **kwargs):
    eig_slice = get_slice(num, which)
    xnp = A.xnp
    sorted_ind = xnp.argsort(A.diag)
    eig_vals = A.diag[sorted_ind]
    eig_vecs = I_like(A).to_dense()[:, sorted_ind]
    return eig_vals[eig_slice], Unitary(lazify(eig_vecs[:, eig_slice]))


@export
def eigmax(A: LinearOperator, tol=1e-7, max_iters=1000, pbar=False, return_vec=False):
    """ Returns eigenvalue with largest magnitude of A up to specified tolerance tol.
        If return_vec=True, also returns the corresponding eigenvector.

        Args:
            A (LinearOperator): The linear operator for which to compute the eigenvalue.
            tol (float, optional): Tolerance for convergence. Default is 1e-7.
            max_iters (int, optional): Maximum number of iterations. Default is 1000.
            pbar (bool, optional): Whether to display a progress bar. Default is False.
            return_vec (bool, optional): Whether to compute and return the
             corresponding eigenvector. Default is False.

        Returns:
            float or tuple: The eigenvalue with the largest magnitude of A.
             If `return_vec` is True, a tuple (eigenvalue, eigenvector) is returned.

        Example:
            >>> eig = eigmax(A, tol=1e-3)
            >>> eig, vec = eigmax(A, tol=1e-3, return_vec=True)
    """
    v0, e0, _ = power_iteration(A, tol=tol, max_iter=max_iters, pbar=pbar)
    return e0 if not return_vec else (e0, v0)


# @export
def eigmin(A: LinearOperator, tol=1e-7):
    """ Returns eigenvalue with smallest magnitude of A
        up to specified tolerance tol."""
    raise NotImplementedError


def get_slice(num, which):
    if which == "LM":
        eig_slice = slice(0, num, None)
    elif which == "SM":
        id = -1 if num is None else -num
        eig_slice = slice(id, None, None)
    else:
        raise NotImplementedError(f"which={which} is not implemented")
    return eig_slice
