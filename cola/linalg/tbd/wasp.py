import torch
from torch.autograd import Function
from torch.func import vjp

from cola.linalg.inverse.torch_cg import cg
from cola.ops.operators import Dense


class WASP(Function):
    @staticmethod
    def forward(ctx, A, rhs, P, tol, max_iters, vtol):
        out, logdet, quad, A, all_rhs, all_soln = logdet_quad_fwd(A, rhs, P, tol, max_iters, vtol)
        ctx.save_for_backward(A, all_rhs, all_soln)
        return out, logdet, quad

    @staticmethod
    def backward(ctx, *grads):
        A, all_rhs, all_soln = ctx.saved_tensors
        out = logdet_quad_bwd(grads, A, all_rhs, all_soln)
        return out


wasp_ldq = WASP.apply


def logdet_quad_fwd(A, rhs, P, tol, max_iters, vtol):
    if vtol > 0.0:
        num_samples = round(1 / vtol**2.0)
        probes = torch.randn(A.shape[1], num_samples, dtype=A.dtype, device=A.device)
        coef = 1.0 / probes.shape[-1]
    else:
        probes = torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
        coef = 1.0
    all_rhs = torch.concatenate((rhs, probes), dim=-1)
    logdet = 0.0
    AOp, POp = Dense(A), Dense(P)
    all_soln, *_ = cg(AOp, all_rhs, x0=torch.zeros_like(all_rhs), P=POp, tol=tol, max_iters=max_iters)
    all_rhs = torch.concatenate((all_soln[:, [0]], coef * probes), dim=-1)
    quad = torch.sum(rhs * all_soln[:, [0]])
    out = logdet - quad
    return out, logdet, quad, A, all_rhs, all_soln


def logdet_quad_bwd(grads, A, all_rhs, all_soln):
    def fun(theta):
        return theta @ all_soln

    dA = vjp_derivs(fun=fun, primals=A, duals=grads[0] * all_rhs)
    out = dA + tuple([None] * 5)
    return out


def vjp_derivs(fun, primals, duals):
    _, vjpfun = vjp(fun, primals)
    output = vjpfun(duals)
    return output
