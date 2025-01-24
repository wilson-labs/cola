import torch
from torch.autograd import Function
from torch.func import vjp

from cola.linalg.inverse.torch_cg import cg


class SLQ(Function):
    @staticmethod
    def forward(ctx, A, *args, **kwargs):
        output = slq_fwd_grad(A, *args, **kwargs)
        ctx.save_for_backward(A)
        ctx.args = args
        ctx.kwargs = kwargs
        return output

    @staticmethod
    def backward(ctx, *grads):
        (A, ) = ctx.saved_tensors
        output = slq_bwd_grad(A, grads, *ctx.args, **ctx.kwargs)
        return output


slq_grad_only_torch = SLQ.apply


def slq_fwd_grad(A, vtol, P, max_iters, tol):
    estimate = torch.randn(A.shape[1], 1, dtype=A.dtype, device=A.device)
    return torch.mean(estimate)


def slq_bwd_grad(A, grads, *args, **kwargs):
    vtol, P, max_iters, tol = args
    num_samples = round(1 / vtol**2.0)

    probes = torch.randn(A.shape[1], num_samples, dtype=A.dtype, device=A.device)
    x0 = torch.zeros_like(probes)
    with torch.no_grad():
        probes_solves, *_ = cg(A, probes, x0=x0, P=P, tol=tol, max_iters=max_iters)
    coef = 1.0 / probes.shape[-1]
    g = grads[0]
    d_solves = coef * g * probes_solves

    def fun(theta):
        return theta @ probes

    dA = vjp_derivs(fun=fun, primals=A, duals=d_solves)
    out = dA + tuple([None] * len(args))
    return out


def vjp_derivs(fun, primals, duals):
    _, vjpfun = vjp(fun, primals)
    output = vjpfun(duals)
    return output
