import numpy as np
from scipy.sparse.linalg import LinearOperator as LO
from scipy.sparse.linalg import eigsh

from cola.ops import Array, Dense, LinearOperator
from cola.utils import export
from cola.utils.utils_linalg import get_numpy_dtype


@export
def irl(A: LinearOperator, start_vector: Array = None, eig_n: int = 6, max_iters: int = 100, tol: float = 1e-7,
        which: str = "LM", pbar: bool = False):
    xnp = A.xnp
    np_dtype = get_numpy_dtype(A.dtype)
    max_iters = None if max_iters == -1 else max_iters
    del pbar
    v0 = None
    if start_vector is not None:
        v0 = np.array(start_vector, dtype=np_dtype)

    def matvec(x):
        X = xnp.array(x, dtype=A.dtype, device=A.device)
        out = A @ X
        return np.array(out.cpu(), dtype=np_dtype)

    A2 = LO(shape=A.shape, dtype=np_dtype, matvec=matvec)
    eigvals, eigvecs = eigsh(A2, k=eig_n, M=None, sigma=None, which=which, v0=v0, tol=tol, maxiter=max_iters)
    eigvals = xnp.array(eigvals, dtype=A.dtype, device=A.device)
    eigvecs = xnp.array(eigvecs, dtype=A.dtype, device=A.device)
    info = {}
    return eigvals, Dense(eigvecs), info
