from functools import reduce
from cola.utils import export, dispatch
from cola.ops.operators import LinearOperator, I_like, Diagonal, Identity
from cola.ops.operators import BlockDiag, ScalarMul, Sum, Dense
from cola.ops.operators import Kronecker, KronSum
from cola.linalg.algorithm_base import Algorithm, Auto
from cola.linalg.trace.diagonal_estimation import Hutch, HutchPP, Exact
import numpy as np


@export
@dispatch.abstract
def diag(A: LinearOperator, k: int = 0, alg: Algorithm = Auto()):
    r""" Extract the (kth) diagonal of a linear operator.

    Can use either the :math:`O(\tfrac{1}{\delta^2})` time stochastic estimation (alg=Hutch())
    or a deterministic :math:`O(n)` time algorithm (alg =Exact()).

    If only unbiased estimates of the diagonal are needed, use the Hutchinson algorithm.

    Args:
        A (LinearOperator): The linear operator to compute the diagonal of.
        k (int, optional): Specify to compute the kth off diagonal diagonal.

    Returns:
        Array: diag
    """


# ########### BASE CASES #############
@dispatch(precedence=-1)
def diag(A: LinearOperator, k, alg: Auto):
    tol = alg.__dict__.get("tol", 1e-6)
    exact_faster = tol < 1 / np.sqrt(10 * np.prod(A.shape))
    if exact_faster:
        return diag(A, k, Exact())
    else:
        return diag(A, k, Hutch(**alg.__dict__))


@dispatch(precedence=-1)
def diag(A: LinearOperator, k, alg: Hutch | HutchPP | Exact):
    return alg(A, k)


# ############ DISPATCH RULES ############
@dispatch
def diag(A: Dense, k, alg: Algorithm):
    xnp = A.xnp
    return xnp.diag(A.A, diagonal=k)


@dispatch
def diag(A: Identity, k, alg: Algorithm):
    if k == 0:
        return A.xnp.ones((A.shape[0], ), A.dtype, device=A.device)
    else:
        return A.xnp.zeros((A.shape[0] - abs(k), ), A.dtype, device=A.device)


@dispatch
def diag(A: Diagonal, k, alg: Algorithm):
    if k == 0:
        return A.diag
    else:
        return A.xnp.zeros((A.shape[0] - abs(k), ), A.dtype, device=A.device)


@dispatch
def diag(A: Sum, k, alg: Algorithm):
    out = sum(diag(M, k, alg) for M in A.Ms)
    return out


@dispatch
def diag(A: BlockDiag, k, alg: Algorithm):
    assert k == 0, "Havent filled this case yet, need to pad with 0s"
    diags = [[diag(M, k, alg)] * m for M, m in zip(A.Ms, A.multiplicities)]
    return A.xnp.concat([item for sublist in diags for item in sublist])


@dispatch
def diag(A: ScalarMul, k, alg: Algorithm):
    return A.c * diag(I_like(A), k, alg)


def product(c):
    return reduce(lambda a, b: a * b, c)


@dispatch
def diag(A: Kronecker, k, alg: Algorithm):
    assert k == 0, "Need to verify correctness of rule for off diagonal case"
    ds = [diag(M, k, alg) for M in A.Ms]
    # compute outer product of the diagonals
    slices = [[None] * i + [slice(None)] + [None] * (len(ds) - i - 1) for i in range(len(ds))]
    return product([d[tuple(s)] for d, s in zip(ds, slices)]).reshape(-1)


@dispatch
def diag(A: KronSum, k, alg: Algorithm):
    assert k == 0, "Need to verify correctness of rule for off diagonal case"
    ds = [diag(M, k, alg) for M in A.Ms]
    # compute outer product of the diagonals
    slices = [[None] * i + [slice(None)] + [None] * (len(ds) - i - 1) for i in range(len(ds))]
    return sum([d[tuple(s)] for d, s in zip(ds, slices)]).reshape(-1)


@export
@dispatch.abstract
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


@dispatch
def trace(A: LinearOperator, alg: Algorithm):
    assert A.shape[0] == A.shape[1], "Can't trace non square matrix"
    return diag(A, 0, alg).sum()


@dispatch
def trace(A: Kronecker, alg: Algorithm):
    return product([trace(M, alg) for M in A.Ms])
