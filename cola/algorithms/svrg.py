import numpy as np
import cola
# from cola.linalg.eigs import eigmax
from cola.ops import Sum, Product, Dense
from cola.ops import I_like
from cola.utils.control_flow import while_loop
from cola.utils import export
# import standard Union type


def oja_svrg_fns():
    def oja_grad(A, W):
        return -(A @ W - W @ (W.T @ W))

    def oja_grad_vrdiff(Ahat, W, anchor_W):
        aW = anchor_W
        return -(Ahat @ (W - aW) - (W @ (W.T @ W) - (aW @ (aW.T @ aW))))

    return oja_grad, oja_grad_vrdiff


def cg_svrg_fns(b):
    def cg_grad(A, W):
        return (A @ W - b)

    def cg_grad_vrdiff(Ahat, W, anchor_W):
        return (Ahat @ (W - anchor_W))

    return cg_grad, cg_grad_vrdiff


# def minres_svrg_fns(b):
#     def minres_grad(A,W):
#         return A.T@(A@W - b)
#     def minres_grad_vrdiff(Ahat,W,anchor_W):
#         return Ahat.T@(Ahat@(W-anchor_W)) # not quite right
#     return minres_grad, minres_grad_vrdiff


# @export
def solve_svrg_symmetric(A: Sum, b, tol=1e-6, P=None, x0=None, pbar=False, info=False, max_iters=5000, bs=50):
    import jax
    # mult = jnp.linalg.norm(b, axis=0)
    # b_norm = do_safe_div(b, mult)
    assert isinstance(A, Sum), f"A (of type {type(A)}) must be directly a Sum"
    xnp = A.xnp
    if x0 is None:
        x0 = xnp.zeros_like(b)
    if P is None:
        P = I_like(A)
    lr, beta = get_optimal_learning_rate(A, P, bs=bs, stochastic=True)

    # lr*=5
    # lr/=20.
    # print(f"Using learning rate {lr} and momentum {beta}")
    n = len(A.Ms)
    N = n // bs
    assert n % bs == 0, "batch size must divide number of samples in current implementation"

    # @xnp.jit
    def body(state):
        w, v, anchor_w, anchor_grad, residual, k, key = state
        # #TODO make work with torch
        perm = jax.random.permutation(key, n).reshape(-1, bs)

        # perm = np.random.permutation(np.arange(n)).reshape(-1,bs)
        def vr_sgd_update(i, wk):
            w, mean_w, v = wk
            # ids = xnp.dynamic_slice(perm,i*bs,bs)
            # ids = perm[i*bs:(i+1)*bs]
            Mi = sum(A.Ms[idd] for idd in perm[i]) * (n / bs)
            grad_vr = Mi @ (w - anchor_w) + anchor_grad
            v = beta * v - lr * P @ grad_vr
            w = w + v
            mean_w += (w - mean_w) / (i + 1)
            return w, mean_w, v

        # w, anchor_w, v = jax.lax.fori_loop(0, N, vr_sgd_update, (w, 0 * w, v))
        # # convert fori into python for loop

        mean_w = 0 * w
        for i in range(N):
            w, mean_w, v = vr_sgd_update(i, (w, mean_w, v))
        anchor_w = mean_w

        anchor_grad = (A @ w - b)
        # w = w - 10*lr * P@anchor_grad
        # anchor_w = w
        # anchor_loss = residual = xnp.norm(anchor_grad)
        key = jax.random.split(key)[0]
        return (w, v, anchor_w, anchor_grad, residual, k + 1, key)

    # @xnp.jit
    def cond(state):
        _, _, _, _, residual, k, _ = state
        return (residual > tol) & (k < max_iters)

    state = (x0, 0 * x0, x0, 0 * x0, xnp.norm(b), 0, jax.random.PRNGKey(38))
    inf = {}
    # while_loop, inf = xnp.while_loop_winfo(lambda s: s[-3], tol, pbar=pbar)
    # while_loop = xnp.while_loop
    _, _, anchor_w, _, residual, _, _ = while_loop(cond, body, state)
    return anchor_w, inf if info else anchor_w


def get_optimal_learning_rate(M, P, bs=1, stochastic=True):
    try:
        e_max = P.preconditioned_eigmax
        e_min = P.preconditioned_eigmin
    except AttributeError:
        e_max = cola.linalg.eigs.eigmax(M @ P, tol=5e-2)
        e_min = 1e-5 * e_max
    lr = .1 / (e_max + e_min)
    lr *= np.sqrt(bs) / 50.
    lr *= 10.
    # if mom:
    #     f = .95
    #     beta = (1 - 2 / (f * (jnp.sqrt(kappa) + 1)))**2
    #     lr *= (1 + beta)
    # else:
    #     beta = 0
    beta = 0
    kappa = e_max / e_min
    text = f"Using learning rate {lr:.3e} and momentum {beta:.3e} "
    text += f"from kappa {kappa:.3e} and eigmax {e_max:.2e}"
    print(text)
    return lr, beta


@export
def svrg_eigh_max(A: Product[Dense, Dense], k=1, tol=1e-6, pbar=False, info=False, max_iters=5000, bs=50, lr_scale=1.):
    """ Use SVRG to find the largest k eigenvalues and eigenvectors of an hermitian matrix A.
        Assumes A = B@C where B and C are dense matrices, decomposes the sum over the
        inner dimension.

        Returns:
            (eigvals, V), info
    """
    x0 = A.xnp.randn(A.shape[1], k, dtype=A.dtype)
    out, info = solve_svrg_generic(A, oja_svrg_fns(), tol=tol, pbar=pbar, info=info, max_iters=max_iters, bs=bs,
                                   lr_scale=lr_scale, x0=x0)
    eigvals = A.xnp.vmap(lambda v: v.T @ v)(out.T)
    V = A.xnp.vmap(lambda v: v / A.xnp.sqrt(v.T @ v))(out.T).T
    return (eigvals, V), info


@export
def svrg_solveh(A: Product[Dense, Dense], b, tol=1e-6, pbar=False, info=False, max_iters=5000, bs=50, lr_scale=1.):
    """ Use SVRG solve the linear system Ax=b assuming an hermitian matrix A.
        Assumes A = B@C where B and C are dense matrices, decomposes the sum over
        the inner dimension.

        Returns:
            x, info
    """
    # x0 = A.xnp.randn(A.shape[1], k, dtype=A.dtype)
    x0 = A.xnp.randn(A.shape[1], b.shape[1], dtype=A.dtype)
    out, info = solve_svrg_generic(A, cg_svrg_fns(b), tol=tol, pbar=pbar, info=info, max_iters=max_iters, bs=bs,
                                   lr_scale=lr_scale, x0=x0)
    return out, info


def solve_svrg_generic(A: Product[Dense, Dense], grad_fns, x0, tol=1e-6, P=None, pbar=False, info=False, max_iters=5000,
                       bs=50, lr_scale=1.):
    import jax  # TODO: enable support for pytorch
    gradients, vrdiffs = grad_fns
    # assert isinstance(A, Sum), f"A (of type {type(A)}) must be directly a Sum"
    xnp = A.xnp

    if P is None:
        P = I_like(A)
    lr, beta = get_optimal_learning_rate(A, P, bs=bs, stochastic=True)
    lr *= lr_scale
    # lr*=5
    # lr/=20.
    M1, M2 = A.Ms
    M1 = M1.A
    M2 = M2.A
    print(f"Using learning rate {lr} and momentum {beta}")
    n = M1.shape[1]
    N = n // bs
    text = f"batch size {bs} must divide number of samples {n} in current implementation"
    assert n % bs == 0, text

    @xnp.jit
    def body(state):
        w, v, anchor_w, anchor_grad, residual, k, key = state
        # #TODO make work with torch
        perm = jax.random.permutation(key, n).reshape(-1, bs)

        # perm = np.random.permutation(np.arange(n)).reshape(-1,bs)
        def vr_sgd_update(i, wk):
            w, mean_w, v = wk
            # ids = xnp.dynamic_slice(perm,i*bs,bs)
            # slice_arr = lambda x: jax.lax.dynamic_slice(x,)
            # A_sample = jax.lax.slice_in_dim(M1,perm[i],0,bs)
            A_sample = Dense(M1[:, perm[i]]) @ Dense(M2[perm[i], :]) * (n / bs)

            grad_vr = vrdiffs(A_sample, w, anchor_w) + anchor_grad
            v = beta * v - lr * P @ grad_vr
            w = w + v
            mean_w += (w - mean_w) / (i + 1)
            return w, mean_w, v

        w, anchor_w, v = jax.lax.fori_loop(0, N, vr_sgd_update, (w, 0 * w, v))

        anchor_grad = gradients(A, w)
        key = jax.random.split(key)[0]
        return (w, v, anchor_w, anchor_grad, xnp.norm(anchor_grad), k + 1, key)

    # @xnp.jit
    def cond(state):
        _, _, _, _, residual, k, _ = state
        return (residual > tol) & (k < max_iters)

    state = (x0, 0 * x0, x0, 0 * x0, 1., 0, jax.random.PRNGKey(38))
    while_loop, inf = xnp.while_loop_winfo(lambda s: s[-3], tol, pbar=pbar)
    _, _, anchor_w, *_ = while_loop(cond, body, state)
    return anchor_w, inf


def solve_svrg_rff(A, rhs, tol=1e-6, P=None, pbar=False, info=False, max_iters=5000, bs=50, lr_scale=1.):
    import jax
    xnp = A.xnp
    x0 = xnp.randn(A.shape[1], rhs.shape[1], dtype=A.dtype)
    x0 /= xnp.norm(x0)
    xnp = A.xnp
    if P is None:
        P = I_like(A)
    lr, beta = get_optimal_learning_rate(A, P, bs=bs, stochastic=True)
    lr *= lr_scale
    # lr, beta = lr_scale, 0.
    PR, D = A.Ms
    M1, M2 = PR.Ms
    M1, M2 = M1.A, M2.A
    print(f"Using learning rate {lr} and momentum {beta}")
    n = M1.shape[1]
    N = n // bs
    text = f"batch size {bs} must divide number of samples {n} in current implementation"
    assert n % bs == 0, text

    @xnp.jit
    def body(state):
        w, v, anchor_w, anchor_grad, _, k, key = state
        perm = jax.random.permutation(key, n).reshape(-1, bs)

        def vr_sgd_update(i, wk):
            w, mean_w, v = wk
            # A_sample = (Dense(M1[:, perm[i]]) @ Dense(M2[perm[i], :])) * (n / bs)
            A_sample = (Dense(M1[:, perm[i]]) @ Dense(M2[perm[i], :])) * (n / bs) + D

            grad_vr = A_sample @ (w - anchor_w) + anchor_grad
            v = beta * v - lr * P @ grad_vr
            w = w + v
            mean_w += (w - mean_w) / (i + 1)
            return w, mean_w, v

        w, anchor_w, v = jax.lax.fori_loop(0, N, vr_sgd_update, (w, 0 * w, v))

        anchor_grad = A @ w - rhs
        key = jax.random.split(key)[0]
        return (w, v, anchor_w, anchor_grad, xnp.norm(anchor_grad), k + 1, key)

    def cond(state):
        _, _, _, _, residual, k, _ = state
        return (residual > tol) & (k < max_iters)

    state = (x0, 0 * x0, x0, 0 * x0, 1., 0, jax.random.PRNGKey(38))
    while_loop, info = xnp.while_loop_winfo(lambda s: s[-3], tol, pbar=pbar)
    _, _, anchor_w, _, _, _, _ = while_loop(cond, body, state)
    return anchor_w, info
