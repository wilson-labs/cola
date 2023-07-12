from cola.ops import LinearOperator
from cola.ops import Array
from cola.ops import get_library_fns
from cola.utils import export


def lanczos_max_eig(A: LinearOperator, rhs: Array, max_iters: int, tol: float = 1e-7):
    """
    Computes the maximum eigenvalue using lanczos

    A: LinearOperator (n, n) positive definite
    rhs: Array (n, b) multiple right hands or (n,) single vector (usually randomly sampled)
    max_iters: int maximum number of iters to run lanczos
    tol: float: tolerance criteria to stop lanczos
    """
    eigvals, _ = lanczos_eig(A=A, rhs=rhs, max_iters=max_iters, tol=tol)
    return eigvals[-1]


@export
def lanczos(A: LinearOperator, start_vector: Array = None, max_iters=100, tol=1e-7):
    """Computes the Lanczos decomposition of a matrix A.

    Args:
        A (LinearOperator): The linear operator representing the matrix A.
        start_vector (Array, optional): The initial vector to start the Lanczos process.
            If not provided, a random vector will be used. Defaults to None.
        max_iters (int, optional): The maximum number of iterations to run Lanczos. Defaults to 100.
        tol (float, optional): The tolerance criteria to stop Lanczos. Defaults to 1e-7.

    Returns:
        Q (Array): The orthogonal matrix Q in the Lanczos decomposition A = Q T Q^H.
        T (Array): The tridiagonal matrix T in the Lanczos decomposition A = Q T Q^H.
    """
    xnp = A.ops
    if start_vector is None:
        start_vector = xnp.random.randn(*A.shape[:-1])
    alpha, beta, iters, vec = lanczos_parts(A=A, rhs=start_vector, max_iters=max_iters, tol=tol)
    alpha, beta = alpha[..., :iters - 1], beta[..., :iters]
    Q = vec[0, :, 1:-1]
    T = construct_tridiagonal_batched(alpha, beta, alpha)
    return Q, T


def lanczos_eig(A: LinearOperator, rhs: Array, max_iters=100, tol=1e-7,
                pbar=False, info=False):
    """
    Computes the eigenvalues and eigenvectors using lanczos

    A: LinearOperator (n, n) positive definite
    rhs: Array (n, b) multiple right hands or (n,) single vector (usually randomly sampled)
    max_iters: int maximum number of iters to run lanczos
    tol: float: tolerance criteria to stop lanczos
    pbar: bool: flag to print progress bar
    """
    xnp = A.ops
    # alpha, beta, iters = get_lanczos_coeffs(A=A, rhs=rhs, max_iters=max_iters, tol=tol)
    # T = construct_tridiagonal(alpha[:iters - 1], beta[:iters], alpha[:iters - 1])
    out = lanczos_parts(A=A, rhs=rhs, max_iters=max_iters, tol=tol, pbar=pbar)
    alpha, beta, iters, vec, infodict = out
    alpha, beta = alpha[..., :iters - 1], beta[..., :iters]
    Q = vec[0, :, 1:-1]
    T = construct_tridiagonal_batched(alpha, beta, alpha)
    # import numpy as np
    # H = np.load("H.npy")
    # H = xnp.array(H, dtype=T.dtype)
    eigvals, eigvectors = xnp.eigh(T[0, :, :])
    V = Q @ eigvectors
    return eigvals, V


def lanczos_parts(A: LinearOperator, rhs: Array, max_iters: int, tol: float, pbar: bool):
    """
    Runs lanczos and saves the tridiagional matrix diagonal (beta) and bands (alpha) as
    well as the number of iterations (iter) and the ortonormal vectors found (vec)

    A: LinearOperator (n, n) positive definite
    rhs: Array (n, b) multiple right hands or (n,) single vector (usually randomly sampled)
    max_iters: int maximum number of iters to run lanczos
    tol: float: tolerance criteria to stop lanczos
    pbar: bool: flag to print progress bar

    https://en.wikipedia.org/wiki/Lanczos_algorithm
    """
    xnp = A.ops

    def body_fun(state):
        iter, vec, beta, alpha = state
        update = xnp.norm(vec[..., iter], axis=-1, keepdims=True)
        vec = xnp.update_array(vec, vec[..., iter] / update, ..., iter)

        # new_vec = A @ vec[..., iter]
        aux = xnp.permute(vec, axes=[1, 0, 2])
        new_vec = (A @ aux[..., iter]).T
        update = xnp.sum(xnp.conj(new_vec) * vec[..., iter], axis=-1)
        beta = xnp.update_array(beta, update, ..., iter - 1)
        aux = beta[..., [iter - 1]] * vec[..., iter] + alpha[..., [iter - 1]] * vec[..., iter - 1]
        new_vec -= aux
        new_vec = do_double_gram(vec, new_vec, xnp)

        vec = xnp.update_array(vec, new_vec, ..., iter + 1)
        alpha = xnp.update_array(alpha, xnp.norm(vec[..., iter + 1], axis=-1), ..., iter)

        return iter + 1, vec, beta, alpha

    def cond_fun(state):
        iter, *_, alpha = state
        is_max = iter <= max_iters
        is_tol = (alpha[..., iter - 1].real >= tol) | (iter <= 1)
        flag = is_max & xnp.any(is_tol)
        return flag

    init_val = initialize_lanczos_vec(xnp, rhs, max_iters=max_iters, dtype=A.dtype)
    # state = xnp.while_loop(cond_fun, body_fun, init_val)
    while_fn, info = xnp.while_loop_winfo(cond_fun, pbar=pbar, tol=tol)
    state = while_fn(cond_fun, body_fun, init_val)
    iter, vec, beta, alpha = state
    return alpha[..., 1:], beta, iter - 1, vec, info


def initialize_lanczos_vec(xnp, rhs, max_iters, dtype):
    iter = xnp.array(1, dtype=xnp.int32)
    beta = xnp.zeros(shape=(rhs.shape[-1], max_iters), dtype=dtype)
    alpha = xnp.zeros(shape=(rhs.shape[-1], max_iters + 1), dtype=dtype)
    vec = xnp.zeros(shape=(rhs.shape[-1], rhs.shape[0], max_iters + 2), dtype=dtype)
    rhs = rhs / xnp.norm(rhs, axis=-2, keepdims=True)
    vec = xnp.update_array(vec, xnp.copy(rhs.T), ..., 1)
    return iter, vec, beta, alpha


def do_double_gram(vec, new_vec, xnp):
    new_vec = do_gram(vec, new_vec, xnp)
    new_vec = do_gram(vec, new_vec, xnp)
    return new_vec


def do_gram(vec, new_vec, xnp):
    aux = xnp.sum(xnp.conj(vec) * xnp.expand(new_vec, -1), axis=-2, keepdims=True)
    new_vec -= xnp.sum(vec * aux, axis=-1)
    return new_vec


def get_lanczos_coeffs(A: LinearOperator, rhs: Array, max_iters: int, tol: float = 1e-7):
    xnp = A.ops

    def body_fun(state):
        iter, vec, vec_prev, beta, alpha = state

        new_vec = A @ vec
        update = xnp.sum(new_vec * vec, axis=-2)
        beta = xnp.update_array(beta, update, iter - 1)
        new_vec -= beta[iter - 1] * vec
        new_vec -= alpha[iter - 1] * vec_prev
        update = xnp.norm(new_vec, axis=-2)
        alpha = xnp.update_array(alpha, update, iter)
        new_vec /= update

        vec_prev = xnp.copy(vec)
        vec = xnp.copy(new_vec)
        return iter + 1, vec, vec_prev, beta, alpha

    def cond_fun(state):
        iter, *_, alpha = state
        is_max = iter <= max_iters
        is_tol = (alpha[iter - 1, 0] >= tol) | (iter <= 1)
        flag = (is_max) & is_tol
        return flag

    init_val = initialize_lanczos(xnp, rhs, max_iters=max_iters, dtype=A.dtype)
    state = xnp.while_loop(cond_fun, body_fun, init_val)
    iter, *_, beta, alpha = state
    return alpha[1:], beta[:-1], iter - 1


def initialize_lanczos(xnp, vec, max_iters, dtype):
    iter = xnp.array(1, dtype=xnp.int32)
    beta = xnp.zeros(shape=(max_iters + 1, 1), dtype=dtype)
    alpha = xnp.zeros(shape=(max_iters + 1, 1), dtype=dtype)
    vec /= xnp.norm(vec)
    vec_prev = xnp.copy(vec)
    return iter, vec, vec_prev, beta, alpha


def construct_tridiagonal(alpha: Array, beta: Array, gamma: Array) -> Array:
    T = construct_tridiagonal_batched(alpha.T, beta.T, gamma.T)
    return T[0, :, :]


def construct_tridiagonal_batched(alpha: Array, beta: Array, gamma: Array) -> Array:
    xnp = get_library_fns(beta.dtype)
    T = xnp.zeros(shape=(beta.shape[-2], beta.shape[-1], beta.shape[-1]), dtype=beta.dtype)
    diag_ind = xnp.array([idx for idx in range(beta.shape[-1])], dtype=xnp.int64)
    T = xnp.update_array(T, beta, ..., diag_ind, diag_ind)
    shifted_ind = xnp.array([idx + 1 for idx in range(gamma.shape[-1])], dtype=xnp.int64)
    T = xnp.update_array(T, gamma, ..., diag_ind[:-1], shifted_ind)
    T = xnp.update_array(T, alpha, ..., shifted_ind, diag_ind[:-1])
    return T


def get_lu_from_tridiagonal(A: LinearOperator) -> Array:
    xnp = A.ops
    eigenvals = xnp.zeros(shape=(A.shape[0], ), dtype=A.dtype)
    eigenvals = xnp.update_array(eigenvals, A.beta[0, 0], 0)

    def body_fun(iter, state):
        pi = A.beta[iter + 1, 0] - ((A.alpha[iter, 0] * A.gamma[iter, 0]) / state[iter])
        state = xnp.update_array(state, pi, iter + 1)
        return state

    lower, upper = xnp.array(0, dtype=xnp.int32), xnp.array(A.shape[0] - 1, dtype=xnp.int32)
    eigenvals = xnp.for_loop(lower, upper, body_fun, init_val=eigenvals)
    return eigenvals
