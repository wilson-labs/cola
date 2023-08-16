from cola.utils import export, dispatch
from cola.ops import LinearOperator, I_like, Diagonal, Identity, BlockDiag, ScalarMul, Sum, Dense, Array
from cola.ops import Kronecker, KronSum
from cola.algorithms import exact_diag,approx_diag
import numpy as np


@dispatch
@export
def diag(v: Array, k=0, **kwargs):
    """ Return a diagonal matrix with the given vector on the diagonal. """
    assert k == 0, "Off diagonal diag not yet supported"
    assert len(v.shape) == 1, f"Unknown input {v.shape}"
    return Diagonal(v)

@dispatch
def diag(A: LinearOperator, k=0, **kwargs):
    """ Extract the (kth) diagonal of a linear operator. 
        Method options: auto, exact, approx"""
    kws = dict(tol=1e-1, pbar=False, max_iters=5000, method='auto')
    kws.update(kwargs)
    method = kws.pop('method')
    if method == 'exact' or (method == 'auto' and (np.prod(A.shape) <= 1e6 or kws['tol']<3e-2)):
        out, info = exact_diag(A, k=k, **kws)
    elif method == 'approx' or (method == 'auto' and (np.prod(A.shape) > 1e6 and kws['tol']>=3e-2)):
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
        return A.xnp.ones(A.shape[0], A.dtype)
    else:
        return A.xnp.zeros(A.shape[0] - k, A.dtype)

@dispatch
def diag(A: Diagonal, k=0, **kwargs):
    if k == 0:
        return A.v
    else:
        return A.xnp.zeros(A.shape[0] - k, A.dtype)

@dispatch
def diag(A: Sum, k=0, **kwargs):
    out = sum(diag(M, **kwargs) for M in A.Ms)
    #out.info = {'sum': [(M.info if hasattr(M,'info') else {}) for M in A.Ms]}
    return out


@dispatch
def diag(A: BlockDiag, k=0, **kwargs):
    assert k == 0, "Havent filled this case yet, need to pad with 0s"
    return A.xnp.concatenate([diag(M, **kwargs) for M in A.Ms])


@dispatch
def diag(A: ScalarMul, k=0, **kwargs):
    return A.c * diag(I_like(A), k=k, **kwargs)

from functools import reduce
def product(c):
    return reduce(lambda a, b: a * b, c)

@dispatch
def diag(A: Kronecker, k=0, **kwargs):
    assert k==0, "Need to verify correctness of rule for off diagonal case"
    ds = [diag(M, **kwargs) for M in A.Ms]
    # compute outer product of the diagonals
    slices = [[None]*i + [slice(None)] + [None] * (len(ds) - i - 1) for i in range(len(ds))]
    return product([d[tuple(s)] for d, s in zip(ds, slices)]).reshape(-1)


@dispatch
def diag(A: KronSum, k=0, **kwargs):
    assert k==0, "Need to verify correctness of rule for off diagonal case"
    ds = [diag(M, **kwargs) for M in A.Ms]
    # compute outer product of the diagonals
    slices = [[None]*i + [slice(None)] + [None] * (len(ds) - i - 1) for i in range(len(ds))]
    return sum([d[tuple(s)] for d, s in zip(ds, slices)]).reshape(-1)

@dispatch
@export
def trace(A: LinearOperator, **kwargs):
    assert A.shape[0] == A.shape[1], "Can't trace non square matrix"
    return diag(A, k=0, **kwargs).sum()

@dispatch
def trace(A: Kronecker, **kwargs):
    return product([trace(M, **kwargs) for M in A.Ms])