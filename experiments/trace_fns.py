from cola.utils.control_flow import while_loop

_small_value = 1e-40


def trace_cg(A, b, x0, max_iters, tol, preconditioner):
    xnp = A.ops
    mult = xnp.norm(b, axis=-2, keepdims=True)
    b_norm = do_safe_div(b, mult, xnp=xnp)
    init_val = initialize(A=A, b=b_norm, preconditioner=preconditioner, x0=x0, xnp=xnp)

    def cond(state):
        flag = cond_fun(state, tol, max_iters, xnp=xnp)
        return flag

    def body_fun(state):
        state = take_cg_step(state, A, preconditioner, xnp=xnp)
        return state

    while_fn, info = while_loop, {}
    state = while_fn(cond_fun=cond, body_fun=body_fun, init_val=init_val)
    return state[0] * mult, state[2] * mult, state[1], info


def initialize(A, b, preconditioner, x0, xnp):
    r0 = b - A @ x0
    z0 = preconditioner @ r0
    p0 = z0
    gamma0 = xnp.sum(xnp.conj(r0) * z0, axis=-2, keepdims=True)
    alpha0 = xnp.zeros(shape=gamma0.shape, dtype=r0.dtype)
    beta0 = xnp.zeros(shape=gamma0.shape, dtype=r0.dtype)
    return (x0, xnp.array(0, dtype=xnp.int32), r0, p0, alpha0, beta0, gamma0)


def cond_fun(value, tol, max_iters, xnp):
    _, k, r, *_ = value
    rs = xnp.norm(r, axis=-2, keepdims=True)
    res_meet = xnp.all(rs > tol)
    flag = (res_meet) & (k < max_iters)
    return flag


def take_cg_step(state, A, preconditioner, xnp):
    x0, k, r0, p0, _, _, gamma0 = state
    eps = xnp.array(_small_value, dtype=p0.real.dtype)
    has_converged = xnp.norm(r0, axis=-2, keepdims=True) < eps
    Ap0 = A @ p0

    alpha = update_alpha(gamma0, p0, Ap0, has_converged, xnp=xnp)
    x1 = x0 + alpha * p0
    r1 = r0 - alpha * Ap0
    z1 = preconditioner @ r1
    gamma1, beta = update_gamma_beta(r1, z1, gamma0, has_converged, xnp=xnp)
    p1 = z1 + beta * p0

    return (x1, k + 1, r1, p1, alpha, beta, gamma1)


def update_alpha(gamma, p, Ap, has_converged, xnp):
    denom = xnp.sum(xnp.conj(p) * Ap, axis=-2, keepdims=True)
    alpha = do_safe_div(gamma, denom, xnp=xnp)
    alpha = xnp.where(has_converged, x=xnp.array(0.0, dtype=p.dtype), y=alpha)
    return alpha


def update_gamma_beta(r, z, gamma0, has_converged, xnp):
    gamma1 = xnp.sum(xnp.conj(r) * z, axis=-2, keepdims=True)
    beta = do_safe_div(gamma1, gamma0, xnp=xnp)
    beta = xnp.where(has_converged, x=xnp.array(0.0, dtype=r.dtype), y=beta)
    return gamma1, beta


def do_safe_div(num, denom, xnp):
    is_zero = xnp.abs(denom) < xnp.array(_small_value, dtype=num.real.dtype)
    denom = xnp.where(is_zero, _small_value, denom)
    output = num / denom
    return output
