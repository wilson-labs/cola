from functools import reduce
from cola.utils import export, dispatch
from cola.ops.operators import LinearOperator, I_like, Diagonal, Identity
from cola.ops.operators import BlockDiag, ScalarMul, Sum, Dense, Array
from cola.ops.operators import Kronecker, KronSum
from cola.linalg.algorithm_base import Algorithm, Auto
from cola.linalg.trace.diagonal_estimation import Hutch, HutchPP, Exact
import numpy as np


@dispatch.abstract
@export
def diag(A: LinearOperator, k: int, alg: Algorithm = Auto()):
    """
    Computes diagonal
    """


@dispatch
def diag(v: Array, k: int = 0):
    """ Constructs a diagonal matrix with the given vector on the diagonal. """
    assert k == 0, "Off diagonal diag not yet supported"
    assert not isinstance(v, LinearOperator), "This diag is for constructing diagonal matrices"
    assert len(v.shape) == 1, f"Unknown input {v.shape}"
    return Diagonal(v)


# ########### BASE CASES #############
@dispatch(precedence=-1)
def diag(A: LinearOperator, k: int = 0, alg: Auto = Auto()):
    tol = alg.__dict__.get("tol", 1e-6)
    exact_faster = tol < 1 / np.sqrt(10 * np.prod(A.shape))
    if exact_faster:
        return diag(A, k, Exact())
    else:
        return diag(A, k, Hutch(**alg.__dict__))


@dispatch(precedence=-1)
def diag(A: LinearOperator, k: int = 0, alg: Hutch | HutchPP | Exact = Exact()):
    return alg(A, k)


# ############ Dispatch Rules ############


@dispatch
def diag(A: Dense, k: int = 0, alg: Algorithm = Auto()):
    xnp = A.xnp
    return xnp.diag(A.A, diagonal=k)


@dispatch
def diag(A: Identity, k: int = 0, alg: Algorithm = Auto()):
    if k == 0:
        return A.xnp.ones((A.shape[0], ), A.dtype, device=A.device)
    else:
        return A.xnp.zeros((A.shape[0] - abs(k), ), A.dtype, device=A.device)


@dispatch
def diag(A: Diagonal, k: int = 0, alg: Algorithm = Auto()):
    if k == 0:
        return A.diag
    else:
        return A.xnp.zeros((A.shape[0] - abs(k), ), A.dtype, device=A.device)


@dispatch
def diag(A: Sum, k: int = 0, alg: Algorithm = Auto()):
    out = sum(diag(M, k, alg) for M in A.Ms)
    return out


@dispatch
def diag(A: BlockDiag, k: int = 0, alg: Algorithm = Auto()):
    assert k == 0, "Havent filled this case yet, need to pad with 0s"
    diags = [[diag(M, k, alg)] * m for M, m in zip(A.Ms, A.multiplicities)]
    return A.xnp.concatenate([item for sublist in diags for item in sublist])


@dispatch
def diag(A: ScalarMul, k: int = 0, alg: Algorithm = Auto()):
    return A.c * diag(I_like(A), k, alg)


def product(c):
    return reduce(lambda a, b: a * b, c)


@dispatch
def diag(A: Kronecker, k: int = 0, alg: Algorithm = Auto()):
    assert k == 0, "Need to verify correctness of rule for off diagonal case"
    ds = [diag(M, k, alg) for M in A.Ms]
    # compute outer product of the diagonals
    slices = [[None] * i + [slice(None)] + [None] * (len(ds) - i - 1) for i in range(len(ds))]
    return product([d[tuple(s)] for d, s in zip(ds, slices)]).reshape(-1)


@dispatch
def diag(A: KronSum, k: int = 0, alg: Algorithm = Auto()):
    assert k == 0, "Need to verify correctness of rule for off diagonal case"
    ds = [diag(M, k, alg) for M in A.Ms]
    # compute outer product of the diagonals
    slices = [[None] * i + [slice(None)] + [None] * (len(ds) - i - 1) for i in range(len(ds))]
    return sum([d[tuple(s)] for d, s in zip(ds, slices)]).reshape(-1)


@dispatch
@export
def trace(A: LinearOperator, alg: Algorithm = Auto()):
    r""" Compute the trace of a linear operator tr(A).

    Uses either :math:`O(\tfrac{1}{\delta^2})` time stochastic estimation (Hutchinson estimator)
    or a deterministic :math:`O(n)` time algorithm if :math:`\delta < 1/\sqrt{10n}`, where
    :math:`\delta=` tol is the standard deviation of the estimate.
    If you unly need unbiased estimates, set tol to be very high.

    Args:
        A (LinearOperator): The linear operator to compute the logdet of.
        tol (float, optional): Tolerance for the variance (std) of the solution,
         returns a stochastic estimate if large enough to save a substantial computation.
         If you want the stochastic estimate, you will need to choose fairly large values,
         e.g. 1e-2. Default: 1e-6.
        pbar (bool, optional): Whether to show a progress bar. Defaults to False.
        method (str, optional): Directly specify method, defaults to 'auto',
         options are 'auto', 'exact', 'approx'.

    Returns:
        Array: trace"""
    assert A.shape[0] == A.shape[1], "Can't trace non square matrix"
    return diag(A, 0, alg).sum()


@dispatch
def trace(A: Kronecker, alg: Algorithm = Auto()):
    return product([trace(M, alg) for M in A.Ms])
