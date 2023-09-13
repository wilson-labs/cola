from cola.ops import Array
from cola.ops import LinearOperator
from cola.ops import I_like
from cola.utils.custom_autodiff import iterative_autograd
# from cola.utils.control_flow import while_loop
from cola.utils import export

_small_value = 1e-40


@export
def cg(A: LinearOperator, rhs: Array, x0=None, P=None, tol=1e-6, max_iters=5000, pbar=False):
    """
    Solves Ax=b or AX=B using conjugate gradients (CG).

    Args:
        A (LinearOperator): A positive definite linear operator of size (n, n).
        rhs (Array): A single right hand side (n,) or multiple right hand sides (n, b).
        x0 (Array, optional): (n,) or (n, b) initial solution guess.
         Defaults to the zero vector.
        P (LinearOperator, optional): Preconditioner. Defaults to the identity.
        tol (float, optional): Stopping criteria.
        max_iters (int, optional): The maximum number of iterations to run.
        pbar (bool, optional): Show a progress bar.

    Returns:
        tuple:
            - soln (Array): solution to the linear system,  either (n,) or (n, b)
            - info (dict): General information about the iterative procedure.
    """
    xnp = A.xnp
    is_vector = len(rhs.shape) == 1
    if x0 is None:
        x0 = xnp.zeros_like(rhs)
    if is_vector:
        rhs = rhs[..., None]
        x0 = x0[..., None]
    if P is None:
        P = I_like(A)
    # soln, res, iters, infodict = run_batched_cg(A, rhs, x0, max_iters, tol, P, pbar=pbar)
    # cg_fn = xnp.jit(run_cg, static_argnums=(0, 3,4,5,6))
    cg_fn = run_cg
    # cg_fn = run_batched_cg
    # TODO: check why the performance degrades so much when adding 5
    # cg_fn = xnp.jit(run_batched_cg, static_argnums=(0, 5, 6))
    soln, *_, infodict = cg_fn(A, rhs, x0, max_iters, tol, P, pbar=pbar)
    soln = soln.reshape(-1) if is_vector else soln
    # infodict['residuals'] = res
    # infodict['iterations'] = iters
    return soln, infodict


def cg_bwd(res, grads, unflatten, *args, **kwargs):
    # TODO: add static and dynamic
    y_grads = grads[0]
    op_args, output = res
    soln = output[0]
    A = unflatten(op_args)
    xnp = A.xnp
    db, *_ = run_batched_cg(A, y_grads, *args[1:], **kwargs)

    def fun(*theta):
        Aop = unflatten(theta)
        return Aop @ soln

    d_params = xnp.vjp_derivs(fun, op_args, -db)
    dA = unflatten(d_params)
    return (dA, db)


@iterative_autograd(cg_bwd)
def run_cg(A, b, x0, max_iters, tol, preconditioner, pbar):
    return run_batched_cg(A, b, x0, max_iters, tol, preconditioner, pbar)


def run_batched_cg(A, b, x0, max_iters, tol, preconditioner, pbar):
    xnp = A.xnp
    mult = xnp.norm(b, axis=-2, keepdims=True)
    b_norm = do_safe_div(b, mult, xnp=xnp)
    init_val = initialize(A=A, b=b_norm, preconditioner=preconditioner, x0=x0, xnp=xnp)

    @xnp.jit
    def cond(state):
        flag = cond_fun(state, tol, max_iters, xnp=xnp)
        return flag

    @xnp.jit
    def body_fun(state):
        state = take_cg_step(state, A, preconditioner, xnp=xnp)
        return state

    @xnp.jit
    def track_res(state):
        return xnp.norm(state[2], axis=-2).mean()

    while_fn, info = xnp.while_loop_winfo(track_res, tol, max_iters, pbar=pbar)
    # while_fn, info = while_loop, {}
    # while_fn, info = xnp.while_loop, {}
    state = while_fn(cond_fun=cond, body_fun=body_fun, init_val=init_val)
    return state[0] * mult, state[2] * mult, state[1], info


def initialize(A, b, preconditioner, x0, xnp):
    r0 = b - A @ x0
    z0 = preconditioner @ r0
    p0 = z0
    device = A.device
    gamma0 = xnp.sum(xnp.conj(r0) * z0, axis=-2, keepdims=True)
    alpha0 = xnp.zeros(shape=gamma0.shape, dtype=r0.dtype, device=device)
    beta0 = xnp.zeros(shape=gamma0.shape, dtype=r0.dtype, device=device)
    return (x0, 0, r0, p0, alpha0, beta0, gamma0)


def cond_fun(value, tol, max_iters, xnp):
    _, k, r, *_ = value
    rs = xnp.norm(r, axis=-2, keepdims=True)
    res_meet = xnp.all(rs > tol)
    flag = (res_meet) & (k < max_iters)
    return flag


def take_cg_step(state, A, preconditioner, xnp):
    x0, k, r0, p0, _, _, gamma0 = state
    eps = xnp.array(_small_value, dtype=p0.real.dtype, device=A.device)
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
    device = xnp.get_device(p)
    alpha = xnp.where(has_converged, xnp.array(0.0, dtype=p.dtype, device=device), alpha)
    return alpha


def update_gamma_beta(r, z, gamma0, has_converged, xnp):
    gamma1 = xnp.sum(xnp.conj(r) * z, axis=-2, keepdims=True)
    beta = do_safe_div(gamma1, gamma0, xnp=xnp)
    device = xnp.get_device(r)
    beta = xnp.where(has_converged, xnp.array(0.0, dtype=r.dtype, device=device), beta)
    return gamma1, beta


def do_safe_div(num, denom, xnp):
    device = xnp.get_device(num)
    is_zero = xnp.abs(denom) < xnp.array(_small_value, dtype=num.real.dtype, device=device)
    denom = xnp.where(is_zero, _small_value, denom)
    output = num / denom
    return output


def run_batched_tracking_cg(A, b, x0, max_iters, tol, preconditioner):
    xnp = A.xnp
    mult = xnp.norm(b, axis=-2, keepdims=True)
    b_norm = do_safe_div(b, mult, xnp=xnp)
    init_val = initialize_track(A=A, b=b_norm, preconditioner=preconditioner, x0=x0, max_iters=max_iters, xnp=xnp)

    def cond(state):
        flag = cond_fun_track(state, tol, max_iters, xnp=xnp)
        return flag

    def body_fun(state):
        state = take_cg_step_and_track(state, A, preconditioner, xnp=xnp)
        return state

    while_fn = xnp.while_loop
    state, tracker = while_fn(cond_fun=cond, body_fun=body_fun, init_val=init_val)
    return state[0] * mult, state[2] * mult, state[1], tracker


def initialize_track(A, b, preconditioner, x0, max_iters, xnp):
    state = initialize(A=A, b=b, preconditioner=preconditioner, x0=x0, xnp=xnp)
    *_, gamma0 = state
    device = A.device
    alphas = xnp.zeros(shape=(max_iters, ) + gamma0.shape, dtype=b.dtype, device=device)
    betas = xnp.zeros(shape=(max_iters, ) + gamma0.shape, dtype=b.dtype, device=device)
    rs = xnp.zeros(shape=(max_iters, ) + gamma0.shape, dtype=b.dtype, device=device)
    return (state, (rs, alphas, betas))


def cond_fun_track(state, tol, max_iters, xnp):
    value, _ = state
    return cond_fun(value=value, tol=tol, max_iters=max_iters, xnp=xnp)


def take_cg_step_and_track(state_track, A, preconditioner, xnp):
    state, track = state_track
    state = take_cg_step(state=state, A=A, preconditioner=preconditioner, xnp=xnp)
    _, k, r, _, alpha, beta, _ = state

    rs, alphas, betas = track
    rs = xnp.update_array(rs, xnp.norm(r, axis=-2, keepdims=True), k - 1)
    alphas = xnp.update_array(alphas, alpha, k - 1)
    betas = xnp.update_array(betas, beta, k - 1)

    return (state, (rs, alphas, betas))
