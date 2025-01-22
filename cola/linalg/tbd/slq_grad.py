from cola.linalg.inverse.cg import cg
from cola.ops import LinearOperator
from cola.utils import export
from cola.utils.custom_autodiff import iterative_autograd


def slq_bwd_grad(res, grads, unflatten, *args, **kwargs):
    op_args, *_ = res
    A = unflatten(op_args)
    xnp = A.xnp

    num_samples = kwargs["num_samples"]
    P = kwargs["P"]
    max_iters = kwargs["max_iters"]
    tol = kwargs["tol"]
    key = kwargs.get("key", xnp.PRNGKey(0))
    key = xnp.PRNGKey(0) if key is None else key

    probes = xnp.randn(A.shape[1], num_samples, dtype=A.dtype, key=key, device=A.device)
    probes_solves, _ = cg(A, probes, tol=tol, max_iters=max_iters, P=P)
    coef = 1.0 / probes.shape[-1]
    g = grads[0] if (isinstance(grads, (tuple, list)) or len(grads.shape) > 0) else grads
    d_solves = coef * g * probes_solves

    def fun(*theta):
        Aop = unflatten(theta)
        return Aop @ probes

    d_params = xnp.vjp_derivs(fun=fun, primals=op_args, duals=d_solves)
    dA = unflatten(d_params)
    return (dA, )


@iterative_autograd(slq_bwd_grad)
def slq_fwd_grad(A, num_samples, P, max_iters, tol, key):
    xnp = A.xnp
    key = xnp.PRNGKey(0) if key is None else key
    rhs = xnp.randn(A.shape[1], num_samples, dtype=A.dtype, key=key, device=A.device)
    estimate = rhs
    return 0.0 * xnp.mean(estimate)


@export
def slq_grad_only(
    A: LinearOperator,
    max_iters: int = 15,
    tol: float = 1e-5,
    vtol=0.2,
    P=None,
    key=None,
):
    num_samples = max(int(1 / vtol**2), 1)
    out = slq_fwd_grad(
        A,
        num_samples=num_samples,
        P=P,
        max_iters=max_iters,
        tol=tol,
        key=key,
    )
    return out
