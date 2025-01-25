import torch
from torch.autograd import Function
from torch.func import vjp


class LogdetQuad(Function):
    @staticmethod
    def forward(ctx, A, rhs, vtol):
        out, logdet, quad, A, L, soln = logdet_quad_fwd(A, rhs)
        ctx.save_for_backward(A, L, soln)
        ctx.vtol = vtol
        return out, logdet, quad

    @staticmethod
    def backward(ctx, *grads):
        A, L, soln = ctx.saved_tensors
        vtol = ctx.vtol
        out = logdet_quad_bwd(grads, A, L, soln, vtol)
        return out


logdet_quad = LogdetQuad.apply


def logdet_quad_fwd(A, rhs):
    with torch.no_grad():
        L = torch.linalg.cholesky(A)
        logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))
        soln = chol_solve(L, rhs)
        quad = torch.sum(rhs * soln)
        out = logdet + quad
    return out, logdet, quad, A, L, soln


def logdet_quad_bwd(grads, A, L, soln, vtol):
    if vtol > 0.0:
        num_samples = round(1 / vtol**2.0)
        probes = torch.randn(A.shape[1], num_samples, dtype=A.dtype, device=A.device)
        coef = 1.0 / probes.shape[-1]
    else:
        probes = torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
        coef = 1.0
    with torch.no_grad():
        L_soln = torch.linalg.solve_triangular(L.T, probes, upper=True)
    all_soln = torch.concatenate((soln, L_soln), dim=-1)
    all_rhs = torch.concatenate((-soln, coef * L_soln), dim=-1)
    d_solves = grads[0] * all_soln

    def fun(theta):
        return theta @ all_rhs

    dA = vjp_derivs(fun=fun, primals=A, duals=d_solves)
    out = dA + tuple([None] * 3)
    return out


def vjp_derivs(fun, primals, duals):
    _, vjpfun = vjp(fun, primals)
    output = vjpfun(duals)
    return output


def chol_solve(L, rhs):
    y = torch.linalg.solve_triangular(L, rhs, upper=False)
    soln = torch.linalg.solve_triangular(L.T, y, upper=True)
    return soln
