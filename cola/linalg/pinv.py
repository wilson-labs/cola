from typing import Tuple
from cola.ops import LinearOperator
from cola.ops import Array
from plum import dispatch
from cola.utils import export
import numpy as np
import cola
from cola.linalg.inverse import inverse


@export
@dispatch
def pinv(A: LinearOperator, **kwargs):
    kws = dict(tol=1e-6, P=None, x0=None, pbar=False, info=False, max_iters=5000)
    kws.update(kwargs)
    n,m = A.shape
    if n > m:
        return inverse(A.H@A, **kws)@A.H
    else:
        return A.H@inverse(A@A.H, **kws)
