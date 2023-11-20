from cola import SelfAdjoint, Unitary
from cola.fns import lazify
from cola.ops import Array, LinearOperator, Dense, Tridiagonal
import cola


def lanczos_eig_bwd(res, grads, unflatten, *args, **kwargs):
    val_grads, eig_grads, _ = grads
    op_args, (eig_vals, eig_vecs, _) = res
    A = unflatten(op_args)
    xnp = A.xnp

    e = eig_vals
    V = eig_vecs.to_dense()  # (n, m)
    W = eig_grads  # (n, m)

    def altogether(*theta):
        Aop = unflatten(theta)
        AV = Aop @ V
        eigs = (AV * V).sum(axis=-2)  # output 1
        out1 = xnp.sum(eigs * val_grads)
        VHAV = V.conj().T @ AV
        diff = xnp.nan_to_num(1 / (e[:, None] - e[None, :]), nan=0., posinf=0., neginf=0.)
        C = (W.conj().T @ V) * diff
        out2 = (C.T * VHAV).sum()
        return out1 + out2

    d_params = xnp.grad(altogether)(*op_args)
    dA = unflatten([d_params])
    return (dA, )


# @iterative_autograd(lanczos_eig_bwd)
def lanczos_eigs(A: LinearOperator, start_vector: Array = None, max_iters: int = 100, tol: float = 1e-7,
                 pbar: bool = False, key=None):
    """
    Computes the eigenvalues and eigenvectors using Lanczos.

    Args:
        A (LinearOperator): A symmetric linear operator of size (n, n).
        start_vector (Array, optional): An initial vector to start Lanczos of size (n, ).
         Defaults to a random probe.
        max_iters (int): The maximum number of iterations to run.
        tol (float, optional): Stopping criteria.
        pbar (bool, optional): Show a progress bar.
        key (PNRGKey, optional): PRNGKey for random number generation.

    Returns:
        tuple:
            - eigvals (Array): eigenvalues of shape (max_iters,).
            - eigvectors (LinearOperator): eigenvectors of shape (n, max_iters).
            - info (dict): General information about the iterative procedure.

    """
    xnp = A.xnp
    Q, T, info = lanczos(A=A, start_vector=start_vector, max_iters=max_iters, tol=tol, pbar=pbar, key=key)
    eigvals, eigvectors = xnp.eigh(T.to_dense())
    idx = xnp.argsort(eigvals, axis=-1)
    V = Q @ lazify(eigvectors[:, idx])
    eigvals = eigvals[..., idx]
    return eigvals, V, info


def LanczosDecomposition(A: LinearOperator, start_vector=None, max_iters=100, tol=1e-7, pbar=False):
    """ Provides the Lanczos decomposition of a matrix A = Q T Q^*.
    LinearOperator form of lanczos, see lanczos for arguments."""
    Q, T, info = lanczos(A=A, start_vector=start_vector, max_iters=max_iters, tol=tol, pbar=pbar)
    A_approx = cola.UnitaryDecomposition(lazify(Q), SelfAdjoint(lazify(T)))
    A_approx.info = info
    return A_approx


def irl(A: LinearOperator, start_vector=None, eig_n: int = 5, which: str = "LM", max_size: int = 20,
        max_iters: int = 100, tol: float = 1e-7, pbar: bool = False):
    """
    Runs the Implicitly Restarted Lanczos (IRL), which basically
    finds a factorization A V = V H using constant memory.

    Args:
        A (LinearOperator): A linear operator of size (n, n).
        start_vector (Array, optional): An initial vector to start Lanczos of size (n, ).
         Defaults to a random probe.
        eig_n (int): The number of eigenvalues to estimate.
        max_size (int): The maximum number of inner Lanczos iterations to run.
        max_iters (int): The maximum number of outer iterations to run.
        tol (float, optional): Stopping criteria.
        pbar (bool, optional): Show a progress bar.

    Returns:
        tuple:
            - V (Array): Unitary matrix of size (n, max_iters) or (b, n, max_iters)
            - H (Array): The Triangular  matrix of size (max_iters, max_iters) or (b, max_iters, max_iters)
            - idx (Array): The number of iterations ran
            - info (dict): General information about the iterative procedure.
    """
    xnp = A.xnp
    init_val = init_lanczos(xnp=xnp, rhs=start_vector, max_iters=max_size, dtype=A.dtype)

    def cond_fun(state):
        *_, idx, norm = state
        is_not_max = idx < max_iters
        is_large = (norm > tol)
        return is_not_max & xnp.any(is_large)

    def body_fun(state):
        V, diag, subdiag, idx, _ = state
        breakpoint()
        V, diag, subdiag, *_ = lanczos_fact(A, state, max_iters=max_size, tol=tol, pbar=pbar)
        T = Tridiagonal(subdiag, diag, subdiag).to_dense()
        V, T = V[0], T[0]
        eigvals, _ = xnp.eig(T)
        eig_slice = get_deflation_eig_slice(eigvals, which=which, eig_n=eig_n, xnp=xnp)
        eigvals = xnp.array(eigvals[eig_slice], dtype=A.dtype, device=A.device)
        vec = diag[-1] * V[:, [-1]]
        T, Q = run_shift(T[:-1], eigvals, xnp)

        beta = T[eig_n, eig_n - 1]
        sigma = Q[-1, eig_n - 1]
        new_vec = beta * V[:, [eig_n]] + sigma * vec
        new_vec = beta * V[:, [eig_n]]
        V0 = V[:, :-1] @ Q[:, :eig_n]
        T0 = T[:eig_n, :eig_n]
        norm = xnp.norm(A @ V0 - V0 @ T0)

        init_val = init_lanczos_from_vec(diag, subdiag, V0, xnp, new_vec.T, rest=eig_n, max_iters=max_size)
        V, diag, subdiag, *_ = init_val
        return V, diag, subdiag, idx + 1, norm[None]

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


def init_lanczos_from_vec(H, V, xnp, new_vec, rest, max_iters):
    dtype, device = H.dtype, H.device
    idx, norm = xnp.array(rest, dtype=xnp.int32, device=H.device), xnp.norm(new_vec)
    H1 = xnp.zeros(shape=(1, max_iters + 1, max_iters), dtype=dtype, device=device)
    Q1 = xnp.zeros(shape=(1, V.shape[0], max_iters + 1), dtype=dtype, device=device)

    H1 = xnp.update_array(H1, H[None][:, :rest, :rest], ..., slice(None, rest, None), slice(None, rest, None))
    H1 = xnp.update_array(H1, norm, ..., rest, rest - 1)
    Q1 = xnp.update_array(Q1, V[None][:, :, :rest], ..., slice(None, rest, None))
    Q1 = xnp.update_array(Q1, new_vec / xnp.clip(norm, 1e-10), ..., idx)
    return Q1, H1, idx, norm[None]


def lanczos(A: LinearOperator, start_vector: Array = None, max_iters=100, tol=1e-7, pbar=False, key=None):
    """
    Computes the Lanczos decomposition of a the operator A, A = Q T Q^*.

    Args:
        A (LinearOperator): A symmetric linear operator of size (n, n).
        start_vector (Array, optional): An initial vector to start Lanczos of size (n, ) or (b, n)
         Defaults to a random probe vector (n, ).
        max_iters (int, optional): The maximum number of iterations to run.
        tol (float, optional): Stopping criteria.
        key (PNRGKey, optional): PRNGKey for random number generation.

    Returns:
        tuple:
            - Q (Array): Orthogonal matrix of size (n, max_iters) or (b, n, max_iters) if start vector has a batch dim
            - T (Array): Tridiagonal matrix of size (max_iters, max_iters) or (b, max_iters, max_iters) if batch dim
            - info (dict): General information about the iterative procedure.

    Notes:
        On jax backend, T will be padded with 0s to size of max_iters since jax does not
         support dynamic sized outputs if traced.

    https://en.wikipedia.org/wiki/Lanczos_algorithm
    """
    xnp = A.xnp
    max_iters = min(max_iters, A.shape[0])
    if start_vector is None:
        key = xnp.PRNGKey(42) if key is None else key
        start_vector = xnp.randn(A.shape[0], dtype=A.dtype, device=A.device, key=key)

    if len(start_vector.shape) == 1:
        rhs = start_vector[:, None]
    else:
        rhs = start_vector

    init_val = init_lanczos(xnp, rhs, max_iters=max_iters, dtype=A.dtype)
    vec, beta, alpha, i, info = lanczos_fact(A, init_val, max_iters, tol)
    alpha, beta, Q, iters = alpha[..., 1:-1], beta, vec[..., 1:-1], i - 1
    if xnp.__name__.find("jax") < 0:
        alpha, beta, Q = alpha[..., :iters - 1], beta[..., :iters], Q[..., :iters]
    if len(start_vector.shape) == 1:
        alpha, beta = alpha[0], beta[0]
        T = Tridiagonal(alpha, beta, alpha)
        return Unitary(Dense(Q[0, :, :])), T, info
    else:
        T = xnp.vmap(Tridiagonal)(alpha, beta, alpha)
        Q = Unitary(xnp.vmap(Dense)(Q))
        return Q, T, info


def lanczos_fact(A: LinearOperator, init_val, max_iters=100, tol=1e-7, pbar=False):
    xnp = A.xnp

    def body_fun(state):
        V, diag, subdiag, i = state
        update = xnp.norm(V[..., i], axis=-1, keepdims=True)
        V = xnp.update_array(V, V[..., i] / update, ..., i)

        aux = xnp.permute(V, axes=[1, 0, 2])
        new_vec = (A @ aux[..., i]).T
        update = xnp.sum(xnp.conj(new_vec) * V[..., i], axis=-1)
        diag = xnp.update_array(diag, update, ..., i - 1)
        aux = diag[..., [i - 1]] * V[..., i] + subdiag[..., [i - 1]] * V[..., i - 1]
        new_vec -= aux
        new_vec = do_double_gram(V, new_vec, xnp)

        V = xnp.update_array(V, new_vec, ..., i + 1)
        subdiag = xnp.update_array(subdiag, xnp.norm(V[..., i + 1], axis=-1), ..., i)

        return V, diag, subdiag, i + 1

    def error(state):
        *_, subdiag, i = state
        err = xnp.array(1e-30, dtype=subdiag.real.dtype, device=xnp.get_device(subdiag))
        rel_err = subdiag[..., i - 1].real / xnp.maximum(subdiag[..., 1].real, err)
        # rel_err = alpha[..., i - 1].real
        return xnp.max(rel_err, axis=0) + (i <= 1) * 1.

    def cond_fun(state):
        *_, subdiag, i = state
        is_not_max = i <= max_iters
        is_large = (subdiag[..., i - 1].real > tol * subdiag[..., 1].real) | (i <= 1)
        flag = is_not_max & xnp.any(is_large)
        return flag

    while_fn, info = xnp.while_loop_winfo(error, tol, max_iters, pbar=pbar)
    V, diag, subdiag, i = while_fn(cond_fun, body_fun, init_val)
    return V, diag, subdiag, i, info


def init_lanczos(xnp, rhs, max_iters, dtype):
    device = xnp.get_device(rhs)
    i = xnp.array(1, dtype=xnp.int32, device=device)
    diag = xnp.zeros(shape=(rhs.shape[-1], max_iters), dtype=dtype, device=device)
    subdiag = xnp.zeros(shape=(rhs.shape[-1], max_iters + 1), dtype=dtype, device=device)
    V = xnp.zeros(shape=(rhs.shape[-1], rhs.shape[0], max_iters + 2), dtype=dtype, device=device)
    norm = xnp.norm(rhs, axis=-2, keepdims=True)
    rhs = rhs / norm
    V = xnp.update_array(V, xnp.copy(rhs.T), ..., 1)
    return V, diag, subdiag, i


def do_double_gram(vec, new_vec, xnp):
    new_vec = do_gram(vec, new_vec, xnp)
    new_vec = do_gram(vec, new_vec, xnp)
    return new_vec


def do_gram(vec, new_vec, xnp):
    aux = xnp.sum(xnp.conj(vec) * xnp.expand(new_vec, -1), axis=-2, keepdims=True)
    new_vec -= xnp.sum(vec * aux, axis=-1)
    return new_vec


def get_lu_from_tridiagonal(A: LinearOperator) -> Array:
    xnp = A.xnp
    eigenvals = xnp.zeros(shape=(A.shape[0], ), dtype=A.dtype, device=A.device)
    eigenvals = xnp.update_array(eigenvals, A.beta[0, 0], 0)

    def body_fun(i, state):
        pi = A.beta[i + 1, 0] - ((A.alpha[i, 0] * A.gamma[i, 0]) / state[i])
        state = xnp.update_array(state, pi, i + 1)
        return state

    lower = xnp.array(0, dtype=xnp.int32, device=A.device)
    upper = xnp.array(A.shape[0] - 1, dtype=xnp.int32, device=A.device)
    eigenvals = xnp.for_loop(lower, upper, body_fun, init_val=eigenvals)
    return eigenvals
