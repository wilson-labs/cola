from typing import Callable
from linops.operator_base import LinearOperator
from linops.algorithms.lanczos import lanczos_parts
from linops.algorithms.lanczos import construct_tridiagonal_batched
# from linops.algorithms.lanczos import construct_tridiagonal


def stochastic_lanczos_quad(A: LinearOperator, fun: Callable, num_samples: int, max_iters: int,
                            tol: float = 1e-7):
    """
    Approximates trace(f(A)) for a positive definite operator A. For example, if f(t) = log(t)
    then the algorithm approximates log(det(A)).

    A: LinearOperator (n, n) positive definite
    fun: Callable | function to apply to the eigenvalues
    num_samples: int | number of samples to use to approximate for the approximation
    max_iters: int | maximum number of iters to run lanczos
    tol: float | tolerance criteria to stop lanczos
    """
    # TODO: ask about what to do with the jit compilation here given the iter shape change
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
