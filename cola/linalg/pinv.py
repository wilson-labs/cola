from typing import Tuple
from cola.ops import LinearOperator, I_like
from cola.ops import Array
from plum import dispatch
from cola.utils import export
import numpy as np
import cola
from cola.linalg.inverse import inverse


@dispatch
@export
def pinv(A: LinearOperator, **kwargs):
    """Computes the Moore-Penrose pseudoinverse of a linear operator A.

    Args:
        A (LinearOperator): The linear operator to compute the pseudoinverse for.
        **kwargs: Additional keyword arguments, including 'tol', 'P', 'x0', 'pbar', 'info', and 'max_iters'. 
                  These are used to customize the inverse computation. If not provided, they take default values.

    Returns:
        LinearOperator: The pseudoinverse of A.

    Example:
        A = LinearOperator((3, 5), jnp.float32, lambda x: x[:3])
        A_pinv = pinv(A, tol=1e-4, max_iters=1000)

    .. warning:: 
        This function is not yet well tested and does not yet include composition rules.
    """
    kws = dict(tol=1e-6, P=None, x0=None, pbar=False, max_iters=5000)
    kws.update(kwargs)
    n,m = A.shape
    if n > m:
        M = A.H@A
        return inverse(cola.PSD(M+kws['tol']*I_like(M)/10), **kws)@A.H
    else:
        M = A@A.H
        return A.H@inverse(cola.PSD(M+kws['tol']*I_like(M)/10), **kws)
