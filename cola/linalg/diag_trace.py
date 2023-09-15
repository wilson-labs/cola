import numpy as np
from functools import reduce
from cola.utils import export, dispatch
from cola.ops import LinearOperator, I_like, Diagonal, Identity
from cola.ops import BlockDiag, ScalarMul, Sum, Dense, Array
from cola.ops import Kronecker, KronSum, Product
from cola.algorithms import exact_diag, approx_diag


@dispatch
@export
def diag(v: Array, k=0, **kwargs):
    """ Constructs a diagonal matrix with the given vector on the diagonal. """
    assert k == 0, "Off diagonal diag not yet supported"
    assert len(v.shape) == 1, f"Unknown input {v.shape}"
    return Diagonal(v)


@dispatch
def diag(A: LinearOperator, k=0, **kwargs):
    r""" Extract the (kth) diagonal of a linear operator.

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
        Array: diag"""
    kws = dict(tol=1e-6, pbar=False, max_iters=5000, method='auto')
    kws.update(kwargs)
    method = kws.pop('method')
    exact_faster = (kws['tol'] < 1 / np.sqrt(10 * A.shape[-1]))
    if method == 'exact' or (method == 'auto' and exact_faster):
        out, info = exact_diag(A, k=k, **kws)
    elif method == 'approx' or (method == 'auto' and exact_faster):
        out, info = approx_diag(A, k=k, **kws)
    out.info = info
    return out


@dispatch
def diag(A: Dense, k=0, **kwargs):
    xnp = A.xnp
    return xnp.diag(A.A, diagonal=k)


@dispatch
def diag(A: Identity, k=0, **kwargs):
    if k == 0:
        return A.xnp.ones((A.shape[0], ), A.dtype, device=A.device)
    else:
        return A.xnp.zeros((A.shape[0] - abs(k), ), A.dtype, device=A.device)


@dispatch
def diag(A: Diagonal, k=0, **kwargs):
    if k == 0:
        return A.diag
    else:
        return A.xnp.zeros((A.shape[0] - abs(k), ), A.dtype, device=A.device)


@dispatch
def diag(A: Sum, k=0, **kwargs):
    out = sum(diag(M, k=k, **kwargs) for M in A.Ms)
    # out.info = {'sum': [(M.info if hasattr(M,'info') else {}) for M in A.Ms]}
    return out


@dispatch
def diag(A: BlockDiag, k=0, **kwargs):
    assert k == 0, "Havent filled this case yet, need to pad with 0s"
    diags = [[diag(M, **kwargs)] * m for M, m in zip(A.Ms, A.multiplicities)]
    return A.xnp.concatenate([item for sublist in diags for item in sublist])


@dispatch
def diag(A: ScalarMul, k=0, **kwargs):
    return A.c * diag(I_like(A), k=k, **kwargs)


def product(c):
    return reduce(lambda a, b: a * b, c)


@dispatch
def diag(A: Kronecker, k=0, **kwargs):
    assert k == 0, "Need to verify correctness of rule for off diagonal case"
    ds = [diag(M, **kwargs) for M in A.Ms]
    # compute outer product of the diagonals
    slices = [[None] * i + [slice(None)] + [None] * (len(ds) - i - 1) for i in range(len(ds))]
    return product([d[tuple(s)] for d, s in zip(ds, slices)]).reshape(-1)


@dispatch
def diag(A: KronSum, k=0, **kwargs):
    assert k == 0, "Need to verify correctness of rule for off diagonal case"
    ds = [diag(M, **kwargs) for M in A.Ms]
    # compute outer product of the diagonals
    slices = [[None] * i + [slice(None)] + [None] * (len(ds) - i - 1) for i in range(len(ds))]
    return sum([d[tuple(s)] for d, s in zip(ds, slices)]).reshape(-1)


@dispatch
@export
def trace(A: LinearOperator, **kwargs):
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
    return diag(A, k=0, **kwargs).sum()


@dispatch
def trace(A: Kronecker, **kwargs):
    return product([trace(M, **kwargs) for M in A.Ms])


@dispatch(cond=lambda A, **kwargs: A.Ms[0].shape[0] > min(M.shape[0] for M in A.Ms))
def trace(A: Product, **kwargs):
    shift_idx = min(enumerate(M.shape[0] for M in A.Ms), key= lambda x: x[1])[0]
    return trace(Product(*(A.Ms[shift_idx::] + A.Ms[:shift_idx:])), **kwargs)
