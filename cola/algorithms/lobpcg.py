import numpy as np
from scipy.sparse.linalg import LinearOperator as LO
from scipy.sparse.linalg import lobpcg as lobpcg_sp
from cola.ops import LinearOperator
from cola.ops import Array
from cola.ops import Dense
from cola.utils import export


@export
def lobpcg(A: LinearOperator, start_vector: Array = None, max_iters: int = 100, tol: float = 1e-7,
           pbar: bool = False):
    xnp = A.xnp
    del pbar, start_vector, tol

    def matvec(x):
        X = xnp.array(x, dtype=A.dtype, device=A.device)
        out = A @ X
        return np.array(out, dtype=np.float32)

    A2 = LO(shape=A.shape, dtype=np.float32, matvec=matvec)
    k = min(A.shape[0] - 1, max_iters)
    X = np.random.normal(size=(A.shape[0], k)).astype(np.float32)
    eigvals, eigvecs = lobpcg_sp(A2, X)
    eigvals = xnp.array(np.copy(eigvals), dtype=A.dtype, device=A.device)
    eigvecs = xnp.array(np.copy(eigvecs), dtype=A.dtype, device=A.device)
    idx = xnp.argsort(eigvals, axis=-1)
    info = {}
    return eigvals[idx], Dense(eigvecs[:, idx]), info
