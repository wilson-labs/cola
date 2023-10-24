import numpy as np
from dataclasses import dataclass
from cola.linalg.algorithm_base import Algorithm
from scipy.sparse.linalg import LinearOperator as LO
from scipy.sparse.linalg import lobpcg as lobpcg_sp
from cola.ops import LinearOperator
from cola.ops import Dense
from cola.utils import export


@export
def lobpcg(A: LinearOperator, max_iters: int = 100):
    xnp = A.xnp

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
    return eigvals[idx], Dense(eigvecs[:, idx])


@export
@dataclass
class LOBPCG(Algorithm):
    """
    Runs Locally Optimal Block Preconditioned Conjugate Gradients (LOBPCG).
    LOBPCG is a preconditioned eigensolver for large real symmetric and
    complex Hermitian definite generalized eigenproblems.

    Args:
        max_iters (int, optional): The maximum number of iterations to run.
    """
    max_iters: int = 100
