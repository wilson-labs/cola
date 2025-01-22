import logging

import numpy as np
from plum import dispatch

from cola.backends import get_library_fns
from cola.ops import Array, LinearOperator
from cola.utils import export

eigmax = None  # TODO: fix


def orthogonal_complement(C, tol=1e-5):
    """ Computes the orthogonal complement to a given matrix proj"""
    xnp = get_library_fns(C.dtype)
    _, S, VH = xnp.svd(C, full_matrices=True)
    rank = (S > tol).sum()
    return VH[rank:].conj().T


@dispatch
@export
def nullspace(C: LinearOperator, tol=1e-5, pbar=True, info=False, method='auto') -> Array:
    """Computes the nullspace of a linear operator C.

    Args:
        C (LinearOperator): The linear operator to compute the nullspace for.
        tol (float, optional): Tolerance for the computation. Default is 1e-5.
        pbar (bool, optional): Whether to display a progress bar. Default is True.
        info (bool, optional): Whether to return additional information. Default is False.
        method (str, optional): Method to use for computation. Options are 'dense', 'iterative'
        and 'auto'. 'auto' chooses based on the C matrix size. Default is 'auto'.

    Returns:
        Array: The nullspace of C, shape (C.shape[1], rank(C)).

    Example:
        >>> C = MyLinearOperator()
        >>> Q = nullspace(C, method='auto', pbar=False)
        >>> C@Q # should be zero

    .. warning::
        This function is not yet well tested and does not yet include composition rules.
    """

    if method == 'dense' or (method == 'auto' and np.prod(C.shape) < 3e7):
        Q = orthogonal_complement(C.to_dense(), tol=tol)
        return Q
    if method == 'iterative' or (method == 'auto' and np.prod(C.shape) > 3e7):
        Q, inf = krylov_constraint_solve(C, tol=tol, pbar=pbar, info=True)
        Q.info = inf
        return Q
    else:
        raise ValueError(f"Unknown method {method}")


def krylov_constraint_solve(C, tol=1e-5, pbar=False, info=False):
    """ Computes the solution basis Q for the linear constraint CQ=0  and QᵀQ=I
        up to specified tolerance with C expressed as a LinearOperator. """
    r = 5
    if C.shape[0] * r * 2 > 2e9:
        raise MemoryError(f"Solns for contraints {C.shape} too large to fit in memory")
        # TODO: output nullspace implicitly
    found_rank = 5
    while found_rank == r:  # TODO: jaxify for loop
        r *= 2  # Iterative doubling of rank until large enough to include the full solution space
        if C.shape[0] * r > 2e9:
            raise MemoryError(f"Solns for contraints {C.shape} too large to fit in memory")
        Q, inf = krylov_constraint_solve_upto_r(C, r, tol, pbar=pbar, info=True)
        found_rank = Q.shape[-1]
    return Q, inf if info else Q


def krylov_constraint_solve_upto_r(C, r, tol=1e-5, max_iter=10000, pbar=False, info=False):
    """ Iterative routine to compute the solution basis to the constraint CQ=0 and QᵀQ=I
        up to the rank r, with given tolerance. Uses gradient descent (+ momentum) on the
        objective |CQ|^2, which provably converges at an exponential rate."""
    xnp = C.ops
    W = xnp.randn(C.shape[-1], r, dtype=C.dtype) / np.sqrt(C.shape[-1])  # if W0 is None else W0

    lr = 1.5 / eigmax(C.T @ C, tol=5e-2)
    tol /= 5

    @xnp.jit
    def body_fn(state):
        i, W, loss = state
        CW = C @ W
        grad = C.T @ CW
        W = W - lr * grad
        # TODO: add back in optimal momentum
        err = ((xnp.abs(CW)**2).sum() / 2)**.5
        return i + 1, W, err

    def cond_fn(state):
        i, W, err = state
        return (err > tol) & (i < max_iter)

    while_loop, inf = xnp.while_loop_winfo(lambda s: s[-1], tol, pbar=pbar)
    i, W, err = while_loop(cond_fn, body_fn, (0, W, 1e10))
    assert err < tol, f"Err {err:.2e} failed to converge to tol {tol:.2e} in {max_iter} iterations"
    # Orthogonalize solution at the end
    U, S, _ = xnp.svd(W, full_matrices=False)
    rank = (S > 3 * tol).sum()
    Q = U[:, :rank]
    final_error = body_fn((0, Q, 0))[-1]
    if final_error > 5 * tol:
        logging.warning(f"Normalized basis has too high error {final_error:.2e} for tol {tol:.2e}")
    scutoff = (S[rank] if r > rank else 0)
    text = f"Singular value gap too small: {S[rank - 1]:.2e}"
    text += "above cutoff {scutoff:.2e} below cutoff. Final L, earlier {S[rank-5:rank]}"
    assert rank == 0 or scutoff < S[rank - 1] / 100, text

    return Q, inf if info else Q


class ConvergenceError(Exception):
    pass


class MemoryError(Exception):
    pass
