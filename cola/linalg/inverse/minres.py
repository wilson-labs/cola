import numpy as np
from scipy.sparse.linalg import LinearOperator as LO
from scipy.sparse.linalg import minres as sp_minres

from cola.ops import Array, LinearOperator
from cola.utils import export
from cola.utils.utils_linalg import get_numpy_dtype


@export
def minres(
    A: LinearOperator,
    rhs: Array,
    start_vector: Array = None,
    tol: float = 1e-5,
    max_iters: int = None,
    pbar: bool = False,
):
    xnp = A.xnp
    np_dtype = get_numpy_dtype(A.dtype)
    del pbar

    v0 = None
    if start_vector is not None:
        v0 = np.array(start_vector, dtype=np_dtype)

    def matvec(x):
        X = xnp.array(x, dtype=A.dtype, device=A.device)
        out = A @ X
        return np.array(out.cpu(), dtype=np_dtype)

    A2 = LO(shape=A.shape, dtype=np_dtype, matvec=matvec)
    rhs = np.array(rhs.cpu(), dtype=np_dtype)
    soln, is_success = sp_minres(
        A2,
        b=rhs,
        x0=v0,
        rtol=tol,
        shift=0.0,
        maxiter=max_iters,
        M=None,
    )
    soln = xnp.array(soln, dtype=A.dtype, device=A.device)
    info = {"success": is_success}
    return soln, info
