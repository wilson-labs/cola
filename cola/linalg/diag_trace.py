from cola.utils import export, dispatch
from cola.ops import LinearOperator, I_like, Diagonal, Identity, BlockDiag, ScalarMul, Sum, Dense, Array


@export
@dispatch
def diag(v: Array, k=0, **kwargs):
    """ Return a diagonal matrix with the given vector on the diagonal. """
    assert k == 0, "Off diagonal diag not yet supported"
    assert len(v.shape) == 1, f"Unknown input {v.shape}"
    return Diagonal(v)




@dispatch
def diag(A: LinearOperator, k=0, **kwargs):
    """ Extract the (kth) diagonal of a linear operator. 
        Method options: auto, exact, approx"""
    kws = dict(tol=1e-2, pbar=False, max_iters=5000, method='auto', info=False)
    kws.update(kwargs)
    method = kws.pop('method')
    if method == 'exact' or (method == 'auto' and (np.prod(A.shape) <= 1e6 or tol<1e-3)):
        pass



@dispatch
def diag(A: Dense, k=0, **kwargs):
    xnp = A.ops
    return xnp.diag(A.A, diagonal=k)


@dispatch
def diag(A: Identity, k=0, **kwargs):
    if k == 0:
        return A.ops.ones(A.shape[0], A.dtype)
    else:
        return A.ops.zeros(A.shape[0] - k, A.dtype)


@dispatch
def diag(A: Sum, k=0, **kwargs):
    return Sum(diag(M) for M in A.Ms)


@dispatch
def diag(A: BlockDiag, k=0):
    assert k == 0, "Havent filled this case yet, need to pad with 0s"
    return A.ops.concatenate([diag(M) for M in A.Ms])


@dispatch
def diag(A: ScalarMul, k=0):
    return A.c * diag(I_like(A), k=k)