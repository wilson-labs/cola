import torch

_small_value = 1e-40


@torch.no_grad
def cg(A, rhs, x0, P, tol, max_iters):
    fn = torch.compile(run_batched_cg)
    out = fn(A, rhs, x0, max_iters, tol, P)
    return out


def run_batched_cg(A, b, x0, max_iters, tol, preconditioner):
    mult = torch.linalg.norm(b, axis=-2, keepdims=True)
    b_norm = do_safe_div(b, mult)
    init_val = initialize(A=A, b=b_norm, preconditioner=preconditioner, x0=x0)
    _, _, r0, *_ = init_val
    tol = tol * torch.linalg.norm(r0, axis=-2, keepdims=True) + tol

    def cond(state):
        flag = cond_fun(state, tol, max_iters)
        return flag

    def body_fun(state):
        state = take_cg_step(state, A, preconditioner)
        return state

    def track_res(state):
        return torch.linalg.norm(state[2], axis=-2).mean()

    state = while_loop(cond_fun=cond, body_fun=body_fun, init_val=init_val)
    return state[0] * mult, state[2] * mult, state[1]


def while_loop(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val


def initialize(A, b, preconditioner, x0):
    r0 = b - A @ x0
    z0 = preconditioner @ r0
    p0 = z0
    device = A.device
    gamma0 = torch.sum(r0 * z0, axis=-2, keepdims=True)
    alpha0 = torch.zeros(gamma0.shape, dtype=r0.dtype, device=device)
    beta0 = torch.zeros(gamma0.shape, dtype=r0.dtype, device=device)
    return (x0, 0, r0, p0, alpha0, beta0, gamma0)


def cond_fun(value, tol, max_iters):
    _, k, r, *_ = value
    rs = torch.linalg.norm(r, axis=-2, keepdims=True)
    res_meet = torch.any(rs > tol)
    flag = (res_meet) & (k < max_iters)
    return flag


def take_cg_step(state, A, preconditioner):
    x0, k, r0, p0, _, _, gamma0 = state
    eps = torch.tensor(_small_value, dtype=p0.real.dtype, device=A.device)
    has_converged = torch.linalg.norm(r0, axis=-2, keepdims=True) < eps
    Ap0 = A @ p0

    alpha = update_alpha(gamma0, p0, Ap0, has_converged)
    x1 = x0 + alpha * p0
    r1 = r0 - alpha * Ap0
    z1 = preconditioner @ r1
    gamma1, beta = update_gamma_beta(r1, z1, gamma0, has_converged)
    p1 = z1 + beta * p0

    return (x1, k + 1, r1, p1, alpha, beta, gamma1)


def update_alpha(gamma, p, Ap, has_converged):
    denom = torch.sum(p * Ap, axis=-2, keepdims=True)
    alpha = do_safe_div(gamma, denom)
    zero = torch.tensor(0.0, dtype=p.dtype, device=p.device)
    alpha = torch.where(has_converged, zero, alpha)
    return alpha


def update_gamma_beta(r, z, gamma0, has_converged):
    gamma1 = torch.sum(r * z, axis=-2, keepdims=True)
    beta = do_safe_div(gamma1, gamma0)
    zero = torch.tensor(0.0, dtype=r.dtype, device=r.device)
    beta = torch.where(has_converged, zero, beta)
    return gamma1, beta


def do_safe_div(num, denom):
    is_zero = torch.abs(denom) < torch.tensor(_small_value, dtype=num.real.dtype, device=num.device)
    denom = torch.where(is_zero, _small_value, denom)
    output = num / denom
    return output
