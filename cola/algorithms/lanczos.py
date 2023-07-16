from functools import partial
from cola.ops import LinearOperator
from cola.ops import Array
from cola.ops import get_library_fns
from cola.utils import export
from cola.utils.custom_autodiff import iterative_autograd
import cola


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


def lanczos_eig_bwd(res, grads, unflatten, *args, **kwargs):
    val_grads, eig_grads = grads
    op_args, (eig_vals, eig_vecs) = res
    A = unflatten(op_args)
    xnp = A.ops

    def fun(*theta, loc):
        Aop = unflatten(theta)
        return Aop @ eig_vecs[:, loc]

    def fun_eig(*theta, loc):
        Aop = unflatten(theta)
        op_diag = 1. / (eig_vals[loc] - eig_vals)
        op_diag = xnp.nan_to_num(op_diag, nan=0., posinf=0., neginf=0.)
        D = cola.ops.Diagonal(op_diag)
        weights = eig_vecs @ D @ xnp.conj(eig_vecs).T
        return weights @ Aop @ eig_vecs[:, loc]

    # TODO: get rid of for loops
    d_params_vals = []
    for idx in range(eig_vecs.shape[-1]):
        fn = partial(fun, loc=idx)
        dlam = xnp.vjp_derivs(fn, op_args, eig_vecs[:, idx])[0]
        required_shape = dlam.shape
        d_params_vals.append(dlam.reshape(-1))
    d_vals = xnp.stack(d_params_vals)
    d_vals = (val_grads @ d_vals).reshape(required_shape)

    # TODO: need to extract jacobian
    d_params_vecs = []
    for idx in range(eig_vecs.shape[-1]):
        fn = partial(fun, loc=idx)
        op_diag = 1. / (eig_vals[idx] - eig_vals)
        op_diag = xnp.nan_to_num(op_diag, nan=0., posinf=0., neginf=0.)
        D = cola.ops.Diagonal(op_diag)
        dl_jac = xnp.jacrev(fn)(*op_args)
        weights = eig_vecs @ D @ xnp.conj(eig_vecs).T
        out = weights @ dl_jac
        d_params_vecs.append(eig_grads[:, idx] @ out)
    d_vecs = xnp.stack(d_params_vecs)
    d_vecs = xnp.sum(d_vecs, axis=0)

    d_params = d_vals + d_vecs
    dA = unflatten([d_params])
    return (dA, )


@iterative_autograd(lanczos_eig_bwd)
def lanczos_eig(A: LinearOperator, rhs: Array, max_iters=100, tol=1e-7, pbar=False):
    """
    Computes the eigenvalues and eigenvectors using lanczos

    A: LinearOperator (n, n) positive definite
    rhs: Array (n, b) multiple right hands or (n,) single vector (usually randomly sampled)
    max_iters: int maximum number of iters to run lanczos
    tol: float: tolerance criteria to stop lanczos
    pbar: bool: flag to print progress bar
    """
    xnp = A.ops
    Q, T, info = lanczos(A=A, start_vector=rhs, max_iters=max_iters, tol=tol, pbar=pbar)
    eigvals, eigvectors = xnp.eigh(T)
    V = Q @ eigvectors
    # sort the eigenvalues and eigenvectors
    idx = xnp.argsort(eigvals, axis=-1)
    eigvals = eigvals[..., idx]
    V = V[..., idx]
    return eigvals, V


@export
def LanczosDecomposition(A: LinearOperator, start_vector=None, max_iters=100, tol=1e-7, pbar=False):
    """ Provides the Lanczos decomposition of a matrix A = Q T Q^H. LinearOperator form of lanczos,
        see lanczos for arguments."""
    Q, T, info = lanczos(A=A, start_vector=start_vector, max_iters=max_iters, tol=tol, pbar=pbar)
    A_approx = cola.UnitaryDecomposition(Q, T)
    A_approx.info = info
    return A_approx


@export
def lanczos(A: LinearOperator, start_vector: Array = None, max_iters=100, tol=1e-7, pbar=False):
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
        info (dict): A dictionary containing information about the Lanczos process.
    """
    xnp = A.ops
    if start_vector is None:
        start_vector = xnp.fixed_normal_samples((A.shape[0], 1))
    alpha, beta, vec, iters, info = lanczos_parts(A=A, rhs=start_vector, max_iters=max_iters,
                                                  tol=tol, pbar=pbar)
    alpha, beta, Q = alpha[..., :iters - 1], beta[..., :iters], vec[0]
    T = construct_tridiagonal_batched(alpha, beta, alpha)[0]
    return Q, T, info


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
    max_iters = min(max_iters, A.shape[0])

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
        return xnp.max(alpha[..., i - 1].real) + (i <= 1) * 1.

    def cond_fun(state):
        i, *_, alpha = state
        is_max = i <= max_iters
        is_tol = (alpha[..., i - 1].real >= tol) | (i <= 1)
        flag = is_max & xnp.any(is_tol)
        return flag

    init_val = initialize_lanczos_vec(xnp, rhs, max_iters=max_iters, dtype=A.dtype)
    while_fn, info = xnp.while_loop_winfo(error, pbar=pbar, tol=tol)
    state = while_fn(cond_fun, body_fun, init_val)
    i, vec, beta, alpha = state
    return alpha[..., 1:], beta, vec[..., 1:-1], i - 1, info


def initialize_lanczos_vec(xnp, rhs, max_iters, dtype):
    i = xnp.array(1, dtype=xnp.int32)
    beta = xnp.zeros(shape=(rhs.shape[-1], max_iters), dtype=dtype)
    alpha = xnp.zeros(shape=(rhs.shape[-1], max_iters + 1), dtype=dtype)
    vec = xnp.zeros(shape=(rhs.shape[-1], rhs.shape[0], max_iters + 2), dtype=dtype)
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


def get_lanczos_coeffs(A: LinearOperator, rhs: Array, max_iters: int, tol: float = 1e-7):
    xnp = A.ops

    def body_fun(state):
        i, vec, vec_prev, beta, alpha = state

        new_vec = A @ vec
        update = xnp.sum(new_vec * vec, axis=-2)
        beta = xnp.update_array(beta, update, i - 1)
        new_vec -= beta[i - 1] * vec
        new_vec -= alpha[i - 1] * vec_prev
        update = xnp.norm(new_vec, axis=-2)
        alpha = xnp.update_array(alpha, update, i)
        new_vec /= update

        vec_prev = xnp.copy(vec)
        vec = xnp.copy(new_vec)
        return i + 1, vec, vec_prev, beta, alpha

    def cond_fun(state):
        i, *_, alpha = state
        is_max = i <= max_iters
        is_tol = (alpha[i - 1, 0] >= tol) | (i <= 1)
        flag = (is_max) & is_tol
        return flag

    init_val = initialize_lanczos(xnp, rhs, max_iters=max_iters, dtype=A.dtype)
    state = xnp.while_loop(cond_fun, body_fun, init_val)
    i, *_, beta, alpha = state
    return alpha[1:], beta[:-1], i - 1


def initialize_lanczos(xnp, vec, max_iters, dtype):
    i = xnp.array(1, dtype=xnp.int32)
    beta = xnp.zeros(shape=(max_iters + 1, 1), dtype=dtype)
    alpha = xnp.zeros(shape=(max_iters + 1, 1), dtype=dtype)
    vec /= xnp.norm(vec)
    vec_prev = xnp.copy(vec)
    return i, vec, vec_prev, beta, alpha


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

    def body_fun(i, state):
        pi = A.beta[i + 1, 0] - ((A.alpha[i, 0] * A.gamma[i, 0]) / state[i])
        state = xnp.update_array(state, pi, i + 1)
        return state

    lower, upper = xnp.array(0, dtype=xnp.int32), xnp.array(A.shape[0] - 1, dtype=xnp.int32)
    eigenvals = xnp.for_loop(lower, upper, body_fun, init_val=eigenvals)
    return eigenvals
