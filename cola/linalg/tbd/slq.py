from typing import Callable

from cola.linalg.decompositions.lanczos import lanczos
from cola.linalg.inverse.cg import cg
from cola.ops import LinearOperator
from cola.utils import export
from cola.utils.custom_autodiff import iterative_autograd


def slq_bwd(res, grads, unflatten, *args, **kwargs):
    op_args, *_ = res
    num_samples = kwargs["num_samples"]

    A = unflatten(op_args)
    xnp = A.xnp
    key = kwargs.get("key", xnp.PRNGKey(0))
    key = xnp.PRNGKey(0) if key is None else key
    probes = xnp.randn(A.shape[1], num_samples, dtype=A.dtype, key=key, device=A.device)
    probes_solves, _ = cg(A, probes, tol=1e-6, max_iters=100)
    # problem here. Assumes f is log. Need to generalize, or specialize outer level function
    coef = 1.0 / probes.shape[-1]
    g = grads[0] if (isinstance(grads, (tuple, list)) or len(grads.shape) > 0) else grads
    d_solves = coef * g * probes_solves

    def fun(*theta):
        Aop = unflatten(theta)
        return Aop @ probes

    d_params = xnp.vjp_derivs(fun=fun, primals=op_args, duals=d_solves)
    dA = unflatten(d_params)
    return (dA, )


@iterative_autograd(slq_bwd)
def slq_fwd(A, fun, num_samples, max_iters, tol, pbar, key):
    xnp = A.xnp
    _mp = xnp.finfo(A.dtype).eps
    key = xnp.PRNGKey(0) if key is None else key
    rhs = xnp.randn(A.shape[1], num_samples, dtype=A.dtype, key=key, device=A.device)
    _, T, _ = lanczos(A, rhs, max_iters, tol, pbar)
    T = xnp.vmap(T.__class__.to_dense)(T)
    eigvals, Q = xnp.eigh(T)
    tau = Q[..., 0, :]
    const = 10 * _mp * xnp.max(eigvals, axis=1, keepdims=True)
    fn_vals = xnp.where(xnp.abs(eigvals) > const, fun(eigvals), xnp.zeros_like(eigvals))
    approx = xnp.sum(tau**2 * fn_vals, axis=-1)
    estimate = A.shape[-2] * approx
    return xnp.mean(estimate, axis=0)


@export
def stochastic_lanczos_quad(A: LinearOperator, fun: Callable, max_iters: int = 15, tol: float = 1e-5, vtol=0.2,
                            pbar: bool = False, key=None):
    """
    Approximates trace(f(A)) for a positive definite operator A and a given function
    f().

    Args:
        A (LinearOperator): A symmetric linear operator of size (n, n).
        fun (Callable): The function to apply to the eigenvalues.
        max_iters (int): The maximum number of iterations to run Lanczos.
        tol (float, optional): Lanczos stopping criteria. (Bias tolerance)
        vtol (float, optional): Variance tolerance for the stochastic estimate.
        pbar (bool, optional): Show a progress bar.

    Returns:
        float: The approximate value of trace(f(A)).
    """
    # TODO: bwds has a bug because it assumes f is log
    num_samples = max(int(1 / vtol**2), 1)
    return slq_fwd(A, fun, num_samples=num_samples, max_iters=max_iters, tol=tol, pbar=pbar, key=key)
