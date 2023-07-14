from cola.ops import LinearOperator
from cola.ops import Array
from cola.ops import Householder, Product
from cola.utils.control_flow import for_loop
from cola.utils import export
import cola


def arnoldi_eig(A: LinearOperator, rhs: Array, max_iters: int, tol: float = 1e-7,
                use_householder=False, pbar=False):
    """Computes eigenvalues and eigenvectors using Arnoldi iteration.

    Args:
        A (LinearOperator): The linear operator representing the matrix A.
        rhs (Array): The right-hand side vector.
        max_iters (int): The maximum number of iterations.
        tol (float, optional): The tolerance criteria. Defaults to 1e-7.
        use_householder (bool, optional): Use Householder Arnoldi iteration. Defaults to False.

    Returns:
        Tuple: A tuple containing the eigenvalues and eigenvectors.
            - eigvals (Array): eigenvalues of shape (max_iters,).
            - eigvectors (Array): eigenvectors of shape (n, max_iters).
    """
    Q, H, info = arnoldi(A=A, start_vector=rhs, max_iters=max_iters, tol=tol,
                         use_householder=use_householder, pbar=pbar)
    xnp = A.ops
    eigvals, eigvectors = xnp.eig(H)
    return eigvals, xnp.cast(Q, dtype=eigvectors.dtype) @ eigvectors


@export
def arnoldi(A: LinearOperator, start_vector=None, max_iters=1000, tol: float = 1e-7,
            use_householder: bool = False, pbar: bool = False):
    """Computes the Arnoldi decomposition of the matrix A = QHQ^*.

    Args:
        A (LinearOperator): The linear operator representing the matrix A.
        start_vector (Array, optional): The vector to start the arnoldi iterations.
        max_iters (int): The maximum number of iterations.
        tol (float, optional): The tolerance criteria. Defaults to 1e-7.
        use_householder (bool, optional): Use Householder Arnoldi iteration. Defaults to False.
        pbar (bool, optional): show a progress bar. Defaults to False.

    Returns:
        Q (Array): The orthogonal matrix Q.
        H (Array): The upper hessenberg matrix H.
    """
    xnp = A.ops
    xnp = A.ops
    if start_vector is None:
        start_vector = xnp.fixed_normal_samples((A.shape[-1], 1))
    if use_householder:
        Q, H, infodict = run_householder_arnoldi(A=A, rhs=start_vector, max_iters=max_iters)
    else:
        Q, H, _, infodict = get_arnoldi_matrix(A=A, rhs=start_vector, max_iters=max_iters, tol=tol,
                                               pbar=pbar)
        # fn = xnp.jit(get_arnoldi_matrix, static_argnums=(0, 2, 3, 4))
        # Q, H, _, infodict = fn(A=A, rhs=start_vector, max_iters=max_iters, tol=tol, pbar=pbar)
        H, Q = H[:-1, :], Q[:, :-1]
    return Q, H, infodict


@export
def ArnoldiDecomposition(A: LinearOperator, start_vector=None, max_iters=100, tol=1e-7,
                         use_householder=False, pbar=False):
    """ Provides the Arnoldi decomposition of a matrix A = Q H Q^H. LinearOperator form of arnoldi,
        see arnoldi for arguments."""
    Q, H, info = arnoldi(A=A, start_vector=start_vector, max_iters=max_iters, tol=tol,
                         use_householder=use_householder, pbar=pbar)
    A_approx = cola.UnitaryDecomposition(Q, H)
    A_approx.info = info
    return A_approx


def get_householder_vec_simple(x, idx, xnp):
    indices = xnp.arange(x.shape[0])
    vec = xnp.where(indices >= idx, x=x, y=0.)
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
    infodict = {}
    return Q[:, 1:], H[:, 1:], infodict


def initialize_householder_arnoldi(xnp, rhs, max_iters, dtype):
    H = xnp.zeros(shape=(max_iters, max_iters + 1), dtype=dtype)
    Q = xnp.zeros(shape=(rhs.shape[-2], max_iters + 1), dtype=dtype)
    rhs = rhs / xnp.norm(rhs)
    Q = xnp.update_array(Q, xnp.copy(rhs[:, 0]), ..., 0)
    zj = Q[:, 0]
    return Q, H, zj


def get_arnoldi_matrix(A: LinearOperator, rhs: Array, max_iters: int, tol: float, pbar: bool):
    xnp = A.ops
    max_iters = min(max_iters, A.shape[0])

    def cond_fun(state):
        idx, *_, norm = state
        is_not_max = idx < max_iters
        is_large = norm >= tol
        return is_not_max & is_large

    def body_fun(state):
        idx, Q, H, _ = state
        new_vec = A @ Q[..., [idx]]
        inner_state = (new_vec, xnp.zeros(shape=(max_iters + 1, ), dtype=new_vec.dtype))

        def inner_loop(jdx, result):
            new_vec, h_vec = result
            angle = xnp.sum(xnp.conj(Q[..., [jdx]]) * new_vec)
            h_vec = xnp.update_array(h_vec, angle, jdx)
            new_vec = new_vec - h_vec[jdx] * Q[..., [jdx]]
            return (new_vec, h_vec)

        new_vec, h_vec = xnp.for_loop(0, idx + 1, inner_loop, inner_state)

        norm = xnp.norm(new_vec)
        new_vec /= norm
        h_vec = xnp.update_array(h_vec, norm, idx + 1)
        Q = xnp.update_array(Q, new_vec[..., 0], ..., idx + 1)
        H = xnp.update_array(H, h_vec, ..., idx)
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
    while_fn, info = xnp.while_loop_winfo(lambda s: s[-1], pbar=pbar, tol=tol)
    # while_fn, info = xnp.while_loop, {}
    state = while_fn(cond_fun, body_fun, init_val)
    state = last_iter_fun(state)
    idx, Q, H, _ = state
    return Q, H[:max_iters + 1, :max_iters], idx, info


def initialize_arnoldi(xnp, rhs, max_iters, dtype):
    idx = xnp.array(0, dtype=xnp.int32)
    H = xnp.zeros(shape=(max_iters + 1, max_iters + 1), dtype=dtype)
    Q = xnp.zeros(shape=(rhs.shape[-2], max_iters + 1), dtype=dtype)
    rhs = rhs / xnp.norm(rhs)
    Q = xnp.update_array(Q, xnp.copy(rhs[:, 0]), ..., 0)
    norm = 1.  # xnp.array(1., dtype=rhs.real.dtype)
    return idx, Q, H, norm
