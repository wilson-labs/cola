from typing import Callable
from cola.ops import LinearOperator
from cola.algorithms.lanczos import lanczos_parts
from cola.algorithms.lanczos import construct_tridiagonal_batched
# from cola.algorithms.lanczos import construct_tridiagonal
from cola.utils import export


@export
def stochastic_lanczos_quad(A: LinearOperator, fun: Callable, num_samples: int, max_iters: int,
                            tol: float = 1e-7):
    """Approximates trace(f(A)) for a positive definite operator A.

    Args:
        A (LinearOperator): The linear operator representing the matrix A.
        fun (Callable): The function to apply to the eigenvalues.
        num_samples (int): The number of samples to use for the approximation.
        max_iters (int): The maximum number of iterations to run Lanczos.
        tol (float, optional): The tolerance criteria to stop Lanczos. Defaults to 1e-7.

    Returns:
        float: The approximate value of trace(f(A)).
    """
    # TODO: how can we jit here given the iter shape change?
    xnp = A.ops
    rhs = xnp.randn(A.shape[1], num_samples, dtype=A.dtype)
    alpha, beta, iters, _ = lanczos_parts(A, rhs, max_iters, tol)
    alpha, beta = alpha[..., :iters - 1], beta[..., :iters]
    T = construct_tridiagonal_batched(alpha, beta, alpha)
    eigvals, Q = xnp.eigh(T)
    tau = Q[..., 0, :]
    approx = xnp.sum(tau**2 * fun(eigvals), axis=-1)
    estimate = A.shape[-2] * approx
    return xnp.mean(estimate, axis=0)
