from typing import Tuple
from cola.ops import LinearOperator
from cola.ops import Array
from plum import dispatch
from cola.utils import export
import numpy as np
import cola

@dispatch
@export
def svd(X: LinearOperator, rank= None, top=True, tol=1e-7, method='auto') -> Tuple[Array]:
    """ Computes the singular value decomposition of a linear operator A.

    Args:
        A: The linear operator to decompose
        k: The rank of the svd to compute
        top: Whether to compute the largest or smallest singular values
        tol: The tolerance to use for the svd
        method: The method to use for the svd (auto, dense, lanczos)

    Returns: A tuple of (U, S, VH) where U is shape (m, k), S is shape (k, k), and VH is shape (k, n)
    """
    # kws = dict(k: int, top=True, tol=1e-7, method='auto')
    # kws.update(kwargs)
    # method = kws.pop('method', 'auto')
    k=rank
    #assert top, "Only top singular values are supported at this time"
    xnp = X.ops
    if method == 'dense' or (method == 'auto' and np.prod(X.shape) <= 1e6):
        U,S,Vh = xnp.svd(X.to_dense())
        return U[:,:k], S[:k], Vh[:k,:]
    elif method == 'lanczos' or (method == 'auto' and np.prod(X.shape) > 1e6):
        Cov = X.H@X/X.shape[0]
        slc = slice(0,k) if not top else slice(-k,None)
        eigs, V = cola.eig(cola.ops.Symmetric(Cov),slc)#,slice(0,k))
        #TODO: reverse order if other side is bigger
        U = X@V # shape (n, k)
        # singular values are the norms
        S = xnp.norm(U,axis=0)
        U = U / S
        return U, S, V.H
    else:
        raise ValueError(f"Unknown method {method} or CoLA didn't fit any selection criteria")
