from cola.ops import LinearOperator
from cola.ops import Array
from cola.ops import Diagonal
from cola.ops import Kronecker
from cola.fns import lazify
from cola.linalg.inverse import inverse
from cola.linalg.sqrt import sqrt
from cola.linalg.eigs import eig
from cola.linalg.logdet import logdet
from cola.algorithms.cg import run_batched_cg
from cola.algorithms import stochastic_lanczos_quad
from torch.autograd import Function


def construct_rffs(X, ls, xnp, num_features):
    weights = xnp.randn(X.shape[-1], num_features // 2, dtype=X.dtype)
    wX = X @ (weights / ls)
    Z = xnp.concat((xnp.cos(wX), xnp.sin(wX)), axis=-1)
    Z /= xnp.sqrt(xnp.array(num_features, dtype=X.dtype))
    return lazify(Z)


def log_determinant(A: LinearOperator) -> Array:
    symm_prod = []
    for k_op, d_op in zip(A.Ms[0].Ms, A.Ms[1]):
        d_op = inverse(sqrt(d_op))
        out = d_op @ k_op @ d_op
        symm_prod.append(out)
    symm_prod = Kronecker(symm_prod)
    eig_vals, _ = eig(symm_prod)
    diag_op = Diagonal(eig_vals + 1.0)
    out = logdet(A.Ms[1])
    out += logdet(diag_op)
    return out


class InvQuad(Function):
    @staticmethod
    def forward(ctx, unflatten, *args):
        inv_quad_rhs, cg_args, *op_args = args
        A = unflatten(op_args)
        inv_quad_solves, *_ = run_batched_cg(A, inv_quad_rhs, *cg_args)
        # inv_quad_solves = A.xnp.solve(A.to_dense(), inv_quad_rhs)
        inv_quad_term = (inv_quad_solves * inv_quad_rhs).sum(-2)

        ctx.A = A
        ctx.unflatten = unflatten
        ctx.cg_args = cg_args
        ctx.soln = inv_quad_solves

        return inv_quad_term

    @staticmethod
    def backward(ctx, grads):
        soln, A = ctx.soln, ctx.A
        grad_vec = -(grads * soln)
        # soln_grad, *_ = run_batched_cg(A, grad_vec, *ctx.cg_args)
        # soln_grad = A.xnp.solve(A.to_dense(), grad_vec)
        dA = A._bilinear_derivative(grad_vec, soln)
        matrix_args_grads = [None] * 3
        out_grads = tuple(matrix_args_grads + list(dA))
        return out_grads


class LogDet(Function):
    @staticmethod
    def forward(ctx, unflatten, *args):
        cg_args, *op_args = args
        A = unflatten(op_args)
        xnp = A.xnp
        dtype = A.dtype

        probes = xnp.randn(A.shape[1], 10, dtype=dtype)

        num_samples, max_iters, tol = 20, 25, 1e-7
        logdet = stochastic_lanczos_quad(A, xnp.log, num_samples, max_iters, tol)
        probes_solves, *_ = run_batched_cg(A, probes, *cg_args)

        ctx.A = A
        ctx.unflatten = unflatten
        ctx.cg_args = cg_args
        ctx.probes = probes
        ctx.probes_solves = probes_solves

        return logdet

    @staticmethod
    def backward(ctx, grads):
        A = ctx.A
        probes = ctx.probes
        probes_solves = ctx.probes_solves

        coef = 1.0 / probes.shape[-1]
        grad_vec = coef * grads * probes_solves
        # dA = A._bilinear_derivative(grad_vec, probes)
        dA = bilinear_derivative(A, grad_vec, probes)

        matrix_args_grads = [None] * 2
        out_grads = tuple(matrix_args_grads + list(dA))
        return out_grads


def bilinear_derivative(A, dual, primal):
    from collections import deque

    params = A.flatten()[0]
    params_with_grads = tuple(arg for arg in params if arg.requires_grad)

    xnp = A.xnp
    with xnp.autograd.enable_grad():
        loss = (dual * (A @ primal)).sum()
        actual_grads = deque(xnp.autograd.grad(loss, params_with_grads, allow_unused=True))

    grads = []
    for arg in params:
        if arg.requires_grad:
            grads.append(actual_grads.popleft())
        else:
            grads.append(None)

    return tuple(grads)
