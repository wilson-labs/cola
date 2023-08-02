from cola.ops import LinearOperator
from cola.ops import Array
from cola.ops import Householder, Product
from cola.utils.control_flow import for_loop
from cola.utils import export
import cola


@export
def arnoldi(A: LinearOperator, start_vector: Array = None, max_iters: int = 100, tol: float = 1e-7,
            use_householder: bool = False, pbar: bool = False):
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

    Returns:
        tuple:
            - eigvals (Array): eigenvalues of shape (max_iters,).
            - eigvectors (LinearOperator): eigenvectors of shape (n, max_iters).
            - info (dict): General information about the iterative procedure.
    """
    Q, H, info = arnoldi_decomp(A=A, start_vector=start_vector, max_iters=max_iters, tol=tol,
                                use_householder=use_householder, pbar=pbar)
    xnp = A.ops
    eigvals, eigvectors = xnp.eig(H)
    eigvectors = xnp.cast(Q, dtype=eigvectors.dtype) @ eigvectors
    eigvectors = xnp.cast(eigvectors, dtype=A.dtype)
    eigvals = xnp.cast(eigvals, dtype=A.dtype)
    return eigvals, eigvectors, info


@export
def arnoldi_decomp(A: LinearOperator, start_vector=None, max_iters=100, tol: float = 1e-7,
                   use_householder: bool = False, pbar: bool = False):
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

    Returns:
        tuple:
            - Q (Array): Unitary matrix of size (n, max_iters).
            - H (Array): The upper Hessenberg matrix of size (max_iters, max_iters).
            - info (dict): General information about the iterative procedure.
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
        Q, H = Q[:, :, 0], H[:, :, 0]
    return Q, H, infodict


def ArnoldiDecomposition(A: LinearOperator, start_vector=None, max_iters=100, tol=1e-7,
                         use_householder=False, pbar=False):
    """ Provides the Arnoldi decomposition of a matrix A = Q H Q^H. LinearOperator form of arnoldi,
        see arnoldi for arguments."""
    Q, H, info = arnoldi_decomp(A=A, start_vector=start_vector, max_iters=max_iters, tol=tol,
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
        *_, idx, norm = state
        is_not_max = idx < max_iters
        is_large = xnp.all(norm >= tol)
        return is_not_max & is_large

    def body_fun(state):
        Q, H, idx, _ = state
        new_vec = A @ Q[..., idx, :]
        h_vec = xnp.zeros(shape=(max_iters + 1, rhs.shape[-1]), dtype=new_vec.dtype)

        def inner_loop(jdx, result):
            new_vec, h_vec = result
            angle = xnp.sum(xnp.conj(Q[..., jdx, :]) * new_vec, axis=-2)
            h_vec = xnp.update_array(h_vec, angle, jdx)
            new_vec = new_vec - h_vec[jdx][None] * Q[..., jdx, :]
            return (new_vec, h_vec)

        new_vec, h_vec = xnp.for_loop(0, idx + 1, inner_loop, (new_vec, h_vec))

        norm = xnp.norm(new_vec, axis=-2)
        new_vec /= xnp.clip(norm, a_min=tol / 2.)
        h_vec = xnp.update_array(h_vec, norm, idx + 1)
        H = xnp.update_array(H, h_vec, ..., idx, slice(None, None, None))
        Q = xnp.update_array(Q, new_vec, ..., idx + 1, slice(None, None, None))
        return Q, H, idx + 1, norm

    init_val = initialize_arnoldi(xnp, rhs, max_iters=max_iters, dtype=A.dtype)
    while_fn, info = xnp.while_loop_winfo(lambda s: s[-1][0], pbar=pbar, tol=tol)
    state = while_fn(cond_fun, body_fun, init_val)
    Q, H, idx, _ = state
    return Q[:, :-1], H[:-1, :], idx, info


def initialize_arnoldi(xnp, rhs, max_iters, dtype):
    idx = xnp.array(0, dtype=xnp.int32)
    H = xnp.zeros(shape=(max_iters + 1, max_iters, rhs.shape[-1]), dtype=dtype)
    Q = xnp.zeros(shape=(rhs.shape[-2], max_iters + 1, rhs.shape[-1]), dtype=dtype)
    rhs = rhs / xnp.norm(rhs, axis=-2)
    Q = xnp.update_array(Q, xnp.copy(rhs), ..., 0, slice(None, None, None))
    norm = xnp.norm(rhs, axis=-2)
    return Q, H, idx, norm
