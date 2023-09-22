from cola import SelfAdjoint, Unitary
from cola.fns import lazify
from cola.ops import Array, LinearOperator, Dense, Tridiagonal
from cola.utils import export
import cola


def lanczos_max_eig(A: LinearOperator, rhs: Array, max_iters: int, tol: float = 1e-7):
    """
    Computes the maximum eigenvalue using lanczos

    A: LinearOperator (n, n) positive definite
    rhs: Array (n, b) multiple right hands or (n,) single vector (usually randomly sampled)
    max_iters: int maximum number of iters to run lanczos
    tol: float: tolerance criteria to stop lanczos
    """
    eigvals, *_ = lanczos_eigs(A=A, start_vector=rhs, max_iters=max_iters, tol=tol)
    return eigvals[-1]


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


# @export
# @iterative_autograd(lanczos_eig_bwd)
@export
def lanczos_eigs(A: LinearOperator, start_vector: Array = None, max_iters: int = 100, tol: float = 1e-7,
                 pbar: bool = False):
    """
    Computes the eigenvalues and eigenvectors using Lanczos.

    Args:
        A (LinearOperator): A symmetric linear operator of size (n, n).
        start_vector (Array, optional): An initial vector to start Lanczos of size (n, ).
         Defaults to a random probe.
        max_iters (int): The maximum number of iterations to run.
        tol (float, optional): Stopping criteria.
        pbar (bool, optional): Show a progress bar.

    Returns:
        tuple:
            - eigvals (Array): eigenvalues of shape (max_iters,).
            - eigvectors (LinearOperator): eigenvectors of shape (n, max_iters).
            - info (dict): General information about the iterative procedure.

    """
    xnp = A.xnp
    if start_vector is None:
        start_vector = xnp.randn(A.shape[0], dtype=A.dtype, device=A.device)
    Q, T, info = lanczos(A=A, start_vector=start_vector, max_iters=max_iters, tol=tol, pbar=pbar)
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


def lanczos(A: LinearOperator, start_vector: Array = None, max_iters=100, tol=1e-7, pbar=False):
    """
    Computes the Lanczos decomposition of a the operator A, A = Q T Q^*.

    Args:
        A (LinearOperator): A symmetric linear operator of size (n, n).
        start_vector (Array, optional): An initial vector to start Lanczos of size (n, ) or (b, n)
         Defaults to a random probe vector (n, ).
        max_iters (int, optional): The maximum number of iterations to run.
        tol (float, optional): Stopping criteria.

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
        start_vector = xnp.randn(A.shape[0], dtype=A.dtype, device=A.device)

    if len(start_vector.shape) == 1:
        rhs = start_vector[:, None]
    else:
        rhs = start_vector

    def body_fun(state):
        i, vec, beta, alpha = state
        update = xnp.norm(vec[..., i], axis=-1, keepdims=True)
        vec = xnp.update_array(vec, vec[..., i] / update, ..., i)

        aux = xnp.permute(vec, axes=[1, 0, 2])
        new_vec = (A @ aux[..., i]).T
        update = xnp.sum(xnp.conj(new_vec) * vec[..., i], axis=-1)
        beta = xnp.update_array(beta, update, ..., i - 1)
        aux = beta[..., [i - 1]] * vec[..., i] + alpha[..., [i - 1]] * vec[..., i - 1]
        new_vec -= aux
        new_vec = do_double_gram(vec, new_vec, xnp)

        vec = xnp.update_array(vec, new_vec, ..., i + 1)
        alpha = xnp.update_array(alpha, xnp.norm(vec[..., i + 1], axis=-1), ..., i)

        return i + 1, vec, beta, alpha

    def error(state):
        i, *_, alpha = state
        err = xnp.array(1e-30, dtype=alpha.real.dtype, device=xnp.get_device(alpha))
        rel_err = alpha[..., i - 1].real / xnp.maximum(alpha[..., 1].real, err)
        # rel_err = alpha[..., i - 1].real
        return xnp.max(rel_err, axis=0) + (i <= 1) * 1.

    def cond_fun(state):
        i, *_, alpha = state
        is_max = i <= max_iters
        is_tol = (alpha[..., i - 1].real > tol * alpha[..., 1].real) | (i <= 1)
        flag = is_max & xnp.any(is_tol)
        return flag

    init_val = initialize_lanczos_vec(xnp, rhs, max_iters=max_iters, dtype=A.dtype)
    while_fn, info = xnp.while_loop_winfo(error, tol, max_iters, pbar=pbar)
    i, vec, beta, alpha = while_fn(cond_fun, body_fun, init_val)
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


def initialize_lanczos_vec(xnp, rhs, max_iters, dtype):
    device = xnp.get_device(rhs)
    i = xnp.array(1, dtype=xnp.int32, device=device)
    beta = xnp.zeros(shape=(rhs.shape[-1], max_iters), dtype=dtype, device=device)
    alpha = xnp.zeros(shape=(rhs.shape[-1], max_iters + 1), dtype=dtype, device=device)
    vec = xnp.zeros(shape=(rhs.shape[-1], rhs.shape[0], max_iters + 2), dtype=dtype, device=device)
    rhs = rhs / xnp.norm(rhs, axis=-2, keepdims=True)
    vec = xnp.update_array(vec, xnp.copy(rhs.T), ..., 1)
    return i, vec, beta, alpha


def do_double_gram(vec, new_vec, xnp):
    new_vec = do_gram(vec, new_vec, xnp)
    new_vec = do_gram(vec, new_vec, xnp)
    return new_vec


def do_gram(vec, new_vec, xnp):
    aux = xnp.sum(xnp.conj(vec) * xnp.expand(new_vec, -1), axis=-2, keepdims=True)
    new_vec -= xnp.sum(vec * aux, axis=-1)
    return new_vec


def initialize_lanczos(xnp, vec, max_iters, dtype):
    device = xnp.get_device(vec)
    i = xnp.array(1, dtype=xnp.int32, device=device)
    beta = xnp.zeros(shape=(max_iters + 1, 1), dtype=dtype, device=device)
    alpha = xnp.zeros(shape=(max_iters + 1, 1), dtype=dtype, device=device)
    vec /= xnp.norm(vec)
    vec_prev = xnp.copy(vec)
    return i, vec, vec_prev, beta, alpha


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
