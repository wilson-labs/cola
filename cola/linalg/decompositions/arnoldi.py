from typing import Tuple

# from cola.utils import export
from cola import Stiefel, lazify
from cola.ops import Array, Dense, Householder, LinearOperator, Product

# def arnoldi_eigs_bwd(res, grads, unflatten, *args, **kwargs):
#     val_grads, eig_grads, _ = grads
#     op_args, (eig_vals, eig_vecs, _) = res
#     A = unflatten(op_args)
#     xnp = A.xnp

#     e = eig_vals
#     V = eig_vecs  # (n, m)
#     W = eig_grads  # (n, m)

#     def altogether(*theta):
#         Aop = unflatten(theta)
#         AV = Aop @ V
#         eigs = (AV * V).sum(axis=-2)  # output 1
#         out1 = xnp.sum(eigs * val_grads)
#         VHAV = V.conj().T @ AV
#         diff = xnp.nan_to_num(1 / (e[:, None] - e[None, :]), nan=0., posinf=0., neginf=0.)
#         C = (W.conj().T @ V) * diff
#         out2 = (C.T * VHAV).sum()
#         return out1 + out2

#     d_params = xnp.grad(altogether)(*op_args)
#     dA = unflatten([d_params])
#     return (dA, )


# @export
# @iterative_autograd(arnoldi_eigs_bwd)
def arnoldi_eigs(A: LinearOperator, start_vector: Array = None, max_iters: int = 100, tol: float = 1e-7,
                 use_householder: bool = False, pbar: bool = False, key=None):
    """
    Computes eigenvalues and eigenvectors using Arnoldi.

    Args:
        A (LinearOperator): A linear operator of size (n, n).
        start_vector (Array, optional): An initial vector to start Arnoldi of size (n, ).
         Defaults to a random probe.
        max_iters (int): The maximum number of iterations to run.
        tol (float, optional): Stopping criteria.
        use_householder (bool, optional): Use Householder Arnoldi variant.
        pbar (bool, optional): Show a progress bar.
        key (PNRGKey, optional): PRNGKey for random number generation.

    Returns:
        tuple:
            - eigvals (Array): eigenvalues of shape (max_iters,).
            - eigvectors (LinearOperator): eigenvectors of shape (n, max_iters).
            - info (dict): General information about the iterative procedure.
    """
    Q, H, info = arnoldi(A=A, start_vector=start_vector, max_iters=max_iters, tol=tol, use_householder=use_householder,
                         pbar=pbar, key=key)
    Q, H = Q[:, :-1], H[:-1]
    xnp = A.xnp
    eigvals, vs = xnp.eig(H.to_dense())
    eigvectors = Q @ lazify(vs)
    return eigvals, eigvectors, info


def ira(A: LinearOperator, start_vector=None, eig_n: int = 5, which: str = "LM", max_size: int = 20,
        max_iters: int = 100, tol: float = 1e-7, pbar: bool = False):
    """
    Runs the Implicitly Restarted Arnoldi Method (IRAM), which basically
    finds a factorization A V = V H using constant memory.

    Args:
        A (LinearOperator): A linear operator of size (n, n).
        start_vector (Array, optional): An initial vector to start Arnoldi of size (n, ).
         Defaults to a random probe.
        eig_n (int): The number of eigenvalues to estimate.
        max_size (int): The maximum number of inner Arnoldi iterations to run.
        max_iters (int): The maximum number of outer iterations to run.
        tol (float, optional): Stopping criteria.
        pbar (bool, optional): Show a progress bar.

    Returns:
        tuple:
            - V (Array): Unitary matrix of size (n, max_iters) or (b, n, max_iters)
            - H (Array): The upper Hessenberg matrix of size (max_iters, max_iters) or (b, max_iters, max_iters)
            - idx (Array): The number of iterations ran
            - info (dict): General information about the iterative procedure.
    """
    xnp = A.xnp
    init_val = init_arnoldi(xnp=xnp, rhs=start_vector, max_iters=max_size, dtype=A.dtype)

    def cond_fun(state):
        _, H, idx, norm = state
        is_not_max = idx < max_iters
        is_large = (norm > tol)
        return is_not_max & xnp.any(is_large)

    def body_fun(state):
        V, H, idx, _ = state
        V, H, *_ = arnoldi_fact(A, state, max_iters=max_size, tol=tol, pbar=pbar)
        V, H = V[0], H[0]
        eigvals, _ = xnp.eig(H[:-1])
        eig_slice = get_deflation_eig_slice(eigvals, which=which, eig_n=eig_n, xnp=xnp)
        eigvals = xnp.cast(eigvals[eig_slice], dtype=A.dtype)
        vec = H[-1, -1] * V[:, [-1]]
        H, Q = run_shift(H[:-1], eigvals, xnp)
        beta = H[eig_n, eig_n - 1]
        sigma = Q[-1, eig_n - 1]
        new_vec = beta * V[:, [eig_n]] + sigma * vec
        V0 = V[:, :-1] @ Q[:, :eig_n]
        H0 = H[:eig_n, :eig_n]
        init_val = init_arnoldi_from_vec(H0, V0, xnp, new_vec.T, rest=eig_n, max_iters=max_size)
        V, H, *_ = init_val
        norm = xnp.norm(A @ V0 - V0 @ H0)
        return V, H, idx + 1, norm[None]

    while_fn, info = xnp.while_loop_winfo(lambda s: s[-1][0], tol, max_iters, pbar=pbar)
    state = while_fn(cond_fun, body_fun, init_val)
    V, H, idx, _ = state
    return V, H, idx, info


def get_deflation_eig_slice(eigvals, which, eig_n, xnp):
    total_n = eigvals.shape[-1]
    nq = total_n - eig_n
    match which:
        case "LM":
            idx = xnp.argsort(xnp.abs(eigvals))
            eig_slice = slice(None, nq, None)
            return idx[eig_slice]
        case "SM":
            idx = xnp.argsort(xnp.abs(eigvals))
            eig_slice = slice(total_n - nq, total_n, None)
            return idx[eig_slice]


def run_shift(H, shifts, xnp):
    dtype, device = H.dtype, H.device
    Id = xnp.eye(*H.shape, dtype=dtype, device=device)
    max_iters = shifts.shape[0]

    def body_fun(idx, state):
        H, V = state
        Q, _ = xnp.qr(H - shifts[idx] * Id, full_matrices=True)
        H = Q.conj().T @ H @ Q
        V = V @ Q
        return H, V

    init_val = (H, xnp.eye(*H.shape, dtype=dtype, device=device))
    H, V = xnp.for_loop(0, max_iters, body_fun, init_val)
    return H, V


def init_arnoldi_from_vec(H, V, xnp, new_vec, rest, max_iters):
    dtype, device = H.dtype, H.device
    idx, norm = xnp.array(rest, dtype=xnp.int32, device=H.device), xnp.norm(new_vec)
    H1 = xnp.zeros(shape=(1, max_iters + 1, max_iters), dtype=dtype, device=device)
    Q1 = xnp.zeros(shape=(1, V.shape[0], max_iters + 1), dtype=dtype, device=device)

    H1 = xnp.update_array(H1, H[None][:, :rest, :rest], ..., slice(None, rest, None), slice(None, rest, None))
    H1 = xnp.update_array(H1, norm, ..., rest, rest - 1)
    Q1 = xnp.update_array(Q1, V[None][:, :, :rest], ..., slice(None, rest, None))
    Q1 = xnp.update_array(Q1, new_vec / xnp.clip(norm, 1e-10), ..., idx)
    return Q1, H1, idx, norm[None]


def arnoldi(A: LinearOperator, start_vector=None, max_iters=100, tol: float = 1e-7, use_householder: bool = False,
            pbar: bool = False, key=None):
    """
    Computes the Arnoldi decomposition of the linear operator A, A = QHQ^*.

    Args:
        A (LinearOperator): A linear operator of size (n, n).
        start_vector (Array, optional): An initial vector to start Arnoldi of size (n, ).
         Defaults to a random probe.
        max_iters (int): The maximum number of iterations to run.
        tol (float, optional): Stopping criteria.
        use_householder (bool, optional): Use Householder Arnoldi iteration.
        pbar (bool, optional): Show a progress bar.
        key (PNRGKey, optional): PRNGKey for random number generation.

    Returns:
        tuple:
            - Q (Array): Unitary matrix of size (n, max_iters) or (b, n, max_iters)
            - H (Array): The upper Hessenberg matrix of size (max_iters, max_iters) or (b, max_iters, max_iters)
            - info (dict): General information about the iterative procedure.
    """
    xnp = A.xnp
    if start_vector is None:
        key = xnp.PRNGKey(42) if key is None else key
        start_vector = xnp.randn(A.shape[-1], dtype=A.dtype, device=A.device, key=key)
    if len(start_vector.shape) == 1:
        rhs = start_vector[:, None]
    else:
        rhs = start_vector
    if use_householder:
        Q, H, infodict = run_householder_arnoldi(A=A, rhs=rhs, max_iters=max_iters)
    else:
        init_val = init_arnoldi(xnp, rhs, max_iters=max_iters, dtype=A.dtype)
        Q, H, _, infodict = arnoldi_fact(A=A, init_val=init_val, max_iters=max_iters, tol=tol, pbar=pbar)
    if len(start_vector.shape) == 1:
        return Stiefel(Dense(Q[0])), Dense(H[0]), infodict
    else:
        H = xnp.vmap(Dense)(H)
        Q = Stiefel(xnp.vmap(Dense)(Q))
        return Q, H, infodict


def get_householder_vec_simple(x, idx, xnp):
    indices = xnp.arange(x.shape[0])
    vec = xnp.where(indices >= idx, x, 0.)
    x_norm = xnp.norm(vec)
    vec = xnp.update_array(vec, vec[idx] - x_norm, idx)
    beta = xnp.nan_to_num(2. / xnp.norm(vec)**2., posinf=0., neginf=0., nan=0.)
    return vec, beta


def get_householder_vec(x, idx, xnp):
    # indices = xnp.arange(x.shape[0])
    # aux = xnp.where(indices >= idx + 1, x=x, y=0.)
    # sigma_2 = xnp.norm(aux)**2.
    sigma_2 = xnp.norm(x[idx + 1:])**2.
    vec = xnp.zeros_like(x)
    vec = xnp.update_array(vec, x[idx:], slice(idx, None, None))
    if sigma_2 == 0 and x[idx] >= 0:
        beta = 0
    elif sigma_2 == 0 and x[idx] < 0:
        beta = -2
    else:
        x_norm_partial = xnp.sqrt(x[idx]**2 + sigma_2)
        if x[idx] <= 0:
            vec = xnp.update_array(vec, x[idx] - x_norm_partial, idx)
        else:
            vec = xnp.update_array(vec, -sigma_2 / (x[idx] + x_norm_partial), idx)
        beta = 2 * vec[idx]**2 / (sigma_2 + vec[idx]**2)
        vec = vec / vec[idx]
        vec = xnp.update_array(vec, vec[idx:] / vec[idx], slice(idx, None, None))
    return vec, beta


def run_householder_arnoldi(A: LinearOperator, rhs: Array, max_iters: int):
    xnp = A.xnp
    dtype = A.dtype
    device = A.device
    Ps = [Householder(xnp.zeros((rhs.shape[-2], 1), dtype=dtype, device=device)) for _ in range(max_iters + 2)]

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
    state = xnp.for_loop(1, max_iters + 1, body_fun, init_val)
    state = last_iter_fun(state)
    Q, H, *_ = state
    infodict = {}
    Q, H = Q[:, 1:], H[:, 1:]
    Q = xnp.permute(Q, (2, 0, 1))
    H = xnp.permute(H, (2, 0, 1))
    return Q, H, infodict


def initialize_householder_arnoldi(xnp, rhs, max_iters, dtype):
    device = xnp.get_device(rhs)
    H = xnp.zeros(shape=(max_iters, max_iters + 1), dtype=dtype, device=device)
    Q = xnp.zeros(shape=(rhs.shape[-2], max_iters + 1), dtype=dtype, device=device)
    rhs = rhs / xnp.norm(rhs)
    Q = xnp.update_array(Q, xnp.copy(rhs[:, 0]), ..., 0)
    zj = Q[:, 0]
    return Q, H, zj


def arnoldi_fact(A: LinearOperator, init_val: Tuple, max_iters: int, tol: float, pbar: bool):
    xnp = A.xnp
    max_iters = min(max_iters, A.shape[0])

    def cond_fun(state):
        _, H, idx, norm = state
        is_not_max = idx < max_iters
        is_large = (norm > tol * H[:, 1, 0].real) | (idx <= 0)
        return is_not_max & xnp.any(is_large)

    def body_fun(state):
        Q, H, idx, _ = state
        new_vec = (A @ Q[..., idx].T).T
        h_vec = xnp.zeros(shape=(H.shape[0], H.shape[1]), dtype=new_vec.dtype, device=xnp.get_device(new_vec))

        def inner_loop(jdx, result):
            new_vec, h_vec = result
            angle = xnp.sum(xnp.conj(Q[..., jdx]) * new_vec, axis=-1)
            h_vec = xnp.update_array(h_vec, angle, ..., jdx)
            new_vec = new_vec - h_vec[..., [jdx]] * Q[..., jdx]
            return (new_vec, h_vec)

        new_vec, h_vec = xnp.for_loop(0, idx + 1, inner_loop, (new_vec, h_vec))
        # new_vec, h_vec = do_dgks_correction(new_vec, h_vec, Q[:, :, :idx + 1], xnp)

        norm = xnp.norm(new_vec, axis=-1, keepdims=True)
        new_vec /= xnp.clip(norm, a_min=tol / 2.)
        h_vec = xnp.update_array(h_vec, norm[:, 0], ..., idx + 1)
        H = xnp.update_array(H, h_vec, ..., idx)
        Q = xnp.update_array(Q, new_vec, ..., idx + 1)
        return Q, H, idx + 1, norm[:, 0]

    while_fn, info = xnp.while_loop_winfo(lambda s: s[-1][0], tol, max_iters, pbar=pbar)
    state = while_fn(cond_fun, body_fun, init_val)
    Q, H, idx, _ = state
    return Q, H, idx, info


def init_arnoldi(xnp, rhs, max_iters, dtype):
    device = xnp.get_device(rhs)
    idx = xnp.array(0, dtype=xnp.int32, device=device)
    H = xnp.zeros(shape=(rhs.shape[-1], max_iters + 1, max_iters), dtype=dtype, device=device)
    Q = xnp.zeros(shape=(rhs.shape[-1], rhs.shape[-2], max_iters + 1), dtype=dtype, device=device)
    norm = xnp.norm(rhs, axis=-2)
    rhs = rhs / norm
    Q = xnp.update_array(Q, xnp.copy(rhs.T), ..., 0)
    return Q, H, idx, norm


def do_dgks_correction(v, h, V, xnp):
    corr = xnp.sum(xnp.conj(V) * v[:, :, None], axis=-2)
    v = v - xnp.sum(V * corr[:, None, :], axis=-1)
    zeros = xnp.zeros(shape=(h.shape[-2], (h.shape[-1] - corr.shape[-1])), dtype=h.dtype, device=h.device)
    corr = xnp.concat((corr, zeros), axis=-1)
    h = h + corr
    return v, h
