from linops.operator_base import LinearOperator
from linops.operator_base import Array
from linops.operators import Householder, Product
# from linops.operators import get_householder_vec
from linops.operators import get_householder_vec_simple
from linops.utils.control_flow import for_loop


def arnoldi_eig(A: LinearOperator, rhs: Array, max_iters: int, tol: float = 1e-7,
                use_householder=False):
    xnp = A.ops
    if use_householder:
        Q, H = run_householder_arnoldi(A=A, rhs=rhs, max_iters=max_iters)
    else:
        # Q, H, _ = get_arnoldi_matrix(A=A, rhs=rhs, max_iters=max_iters, tol=tol)
        fn = xnp.jit(get_arnoldi_matrix, static_argnums=(0, 2, 3))
        Q, H, _ = fn(A=A, rhs=rhs, max_iters=max_iters, tol=tol)
        H = H[:-1, :]
    eigvals, eigvectors = xnp.eig(H)
    # aux = eigvectors @ xnp.diag(eigvals) @ xnp.inv(eigvectors) - H
    return eigvals, eigvectors, Q


def run_householder_arnoldi(A: LinearOperator, rhs: Array, max_iters: int):
    xnp = A.ops
    dtype = A.dtype
    Ps = [Householder(xnp.zeros((rhs.shape[-2], 1), dtype=dtype)) for _ in range(max_iters + 2)]

    def body_fun(idx, state):
        Q, H, zj = state
        # vec, beta = get_householder_vec(zj, idx - 1, xnp)
        vec, beta = get_householder_vec_simple(zj, idx - 1, xnp)
        Ps[idx].vec, Ps[idx].beta = vec[:, None], beta
        aux = Ps[idx] @ zj
        H = xnp.update_array(H, aux[:max_iters], ..., idx - 1)
        Q = xnp.update_array(Q, 1., idx - 1, idx)
        Reflect = Product(*[Ps[jdx] for jdx in range(1, idx + 1)])
        # Reflect = Product([Ps[jdx] for jdx in range(1, max_iters + 1)])
        Q = xnp.update_array(Q, Reflect @ Q[:, idx], ..., idx)
        Reflect = Product(*[Ps[jdx] for jdx in range(idx + 1, 0, -1)])
        # Reflect = Product([Ps[jdx] for jdx in range(max_iters + 1, 0, -1)])
        zj = Reflect @ A @ Q[:, idx]
        return Q, H, zj

    def last_iter_fun(state):
        Q, H, zj = state
        H = xnp.update_array(H, zj[:max_iters], ..., max_iters)
        return Q, H, zj

    init_val = initialize_householder_arnoldi(xnp, rhs, max_iters=max_iters, dtype=A.dtype)
    # state = xnp.while_loop(cond_fun, body_fun, init_val)
    state = for_loop(1, max_iters + 1, body_fun, init_val)
    state = last_iter_fun(state)
    Q, H, *_ = state
    return Q[:, 1:], H[:, 1:]


def initialize_householder_arnoldi(xnp, rhs, max_iters, dtype):
    H = xnp.zeros(shape=(max_iters, max_iters + 1), dtype=dtype)
    Q = xnp.zeros(shape=(rhs.shape[-2], max_iters + 1), dtype=dtype)
    rhs = rhs / xnp.norm(rhs)
    Q = xnp.update_array(Q, xnp.copy(rhs[:, 0]), ..., 0)
    zj = Q[:, 0]
    return Q, H, zj


def get_arnoldi_matrix(A: LinearOperator, rhs: Array, max_iters: int, tol: float = 1e-7):
    xnp = A.ops

    def cond_fun(state):
        idx, *_, norm = state
        is_not_max = idx < max_iters
        is_large = norm >= tol
        return is_not_max & is_large

    def body_fun(state):
        idx, Q, H, _ = state
        new_vec = A @ Q[..., [idx]]
        h_vec = xnp.sum(xnp.conj(Q) * new_vec, axis=-2, keepdims=True)
        new_vec = new_vec - xnp.sum(Q * h_vec, axis=-1, keepdims=True)
        norm = xnp.norm(new_vec)
        new_vec /= norm
        h_vec = xnp.permute(h_vec, axes=(1, 0))
        h_vec = xnp.update_array(h_vec, norm, idx + 1)
        Q = xnp.update_array(Q, new_vec[..., 0], ..., idx + 1)
        H = xnp.update_array(H, h_vec[..., 0], ..., idx)
        return idx + 1, Q, H, norm

    def last_iter_fun(state):
        idx, Q, H, _ = state
        new_vec = A @ Q[..., [idx]]
        h_vec = xnp.sum(xnp.conj(Q) * new_vec, axis=-2, keepdims=True)
        new_vec = new_vec - xnp.sum(Q * h_vec, axis=-1, keepdims=True)
        norm = xnp.norm(new_vec)
        new_vec /= norm
        h_vec = xnp.permute(h_vec, axes=(1, 0))
        H = xnp.update_array(H, h_vec[..., 0], ..., idx)
        return idx + 1, Q, H, norm

    init_val = initialize_arnoldi(xnp, rhs, max_iters=max_iters, dtype=A.dtype)
    state = xnp.while_loop(cond_fun, body_fun, init_val)
    state = last_iter_fun(state)
    idx, Q, H, _ = state
    return Q, H[:max_iters + 1, :max_iters], idx


def initialize_arnoldi(xnp, rhs, max_iters, dtype):
    idx = xnp.array(0, dtype=xnp.int32)
    H = xnp.zeros(shape=(max_iters + 1, max_iters + 1), dtype=dtype)
    Q = xnp.zeros(shape=(rhs.shape[-2], max_iters + 1), dtype=dtype)
    rhs = rhs / xnp.norm(rhs)
    Q = xnp.update_array(Q, xnp.copy(rhs[:, 0]), ..., 0)
    norm = xnp.array(1., dtype=rhs.real.dtype)
    return idx, Q, H, norm
