import numpy as np
from scipy.sparse.linalg import LinearOperator as LO
from scipy.sparse.linalg import eigsh
from cola.ops import LinearOperator
from cola.ops import Array
from cola.ops import Dense
from cola.utils import export


@export
def iram(A: LinearOperator, start_vector: Array = None, max_iters: int = 100, tol: float = 1e-7, pbar: bool = False):
    xnp = A.xnp
    del pbar
    del start_vector

    def matvec(x):
        X = xnp.array(x, dtype=A.dtype, device=A.device)
        out = A @ X
        return np.array(out, dtype=np.float32)

    A2 = LO(shape=A.shape, dtype=np.float32, matvec=matvec)
    k = min(A.shape[0] - 1, max_iters)
    eigvals, eigvecs = eigsh(A2, k=k, M=None, sigma=None, which="LM", v0=None, tol=tol)
    eigvals, eigvecs = xnp.array(eigvals, dtype=A.dtype,
                                 device=A.device), xnp.array(eigvecs, dtype=A.dtype, device=A.device)
    info = {}
    return eigvals, Dense(eigvecs), info
