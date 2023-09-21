from cola.ops import LinearOperator
from cola.ops import Array
# from cola.algorithms.arnoldi import run_householder_arnoldi
from cola.algorithms.arnoldi import arnoldi
from cola.utils import export
from cola.utils.custom_autodiff import iterative_autograd


@export
def gmres(A: LinearOperator, rhs: Array, x0=None, max_iters=100, tol=1e-7, P=None, use_householder=False,
          use_triangular=False, pbar=False):
    """
    Solves Ax=b or AX=B using GMRES.

    Args:
        A (LinearOperator): A linear operator of size (n, n).
        rhs (Array): A single right hand side (n,) or multiple right hand sides (n, b).
        x0 (Array, optional): (n,) or (n, b) initial solution guess.
         Defaults to the zero vector.
        max_iters (int, optional): The maximum number of iterations to run.
        tol (float, optional): The tolerance for convergence.
        P (array, optional): Preconditioner. Defaults to the Identity.
        use_householder (bool, optional): Use Householder Arnoldi variatnt
        use_triangular (bool, optional): Use triangular QR factorization.
        pbar (bool, optional): show a progress bar.

    Returns:
        tuple:
            - soln (Array): solution to the linear system,  either (n,) or (n, b)
            - info (dict): general information about the iterative procedure.
    """
    xnp = A.xnp
    is_vector = len(rhs.shape) == 1
    if x0 is None:
        x0 = xnp.zeros_like(rhs)
    if is_vector:
        rhs = rhs[..., None]
        x0 = x0[..., None]
    soln, infodict = gmres_fwd(A=A, rhs=rhs, x0=x0, max_iters=max_iters, tol=tol, P=P, use_householder=use_householder,
                               use_triangular=use_triangular, pbar=pbar)
    if is_vector:
        soln = soln[:, 0]
    return soln, infodict


def gmres_bwd(res, grads, unflatten, *args, **kwargs):
    y_grads = grads[0]
    op_args, output = res
    soln = output[0]
    A = unflatten(op_args)
    xnp = A.xnp
    db, _ = gmres_fwd(A, y_grads, *args[1:], **kwargs)

    def fun(*theta):
        Aop = unflatten(theta)
        return Aop @ soln

    d_params = xnp.vjp_derivs(fun, op_args, -db)
    dA = unflatten(d_params)
    return (dA, db)


@iterative_autograd(gmres_bwd)
def gmres_fwd(A, rhs, x0, max_iters, tol, P, use_householder, use_triangular, pbar):
    xnp = A.xnp
    res = rhs - A @ x0  # (m,k)
    Q, H, infodict = arnoldi(A=A, start_vector=res, max_iters=max_iters, tol=tol, pbar=pbar,
                             use_householder=use_householder)

    beta = xnp.norm(res, axis=-2)
    e1 = xnp.zeros(shape=(H.shape[1], beta.shape[0]), dtype=rhs.dtype, device=A.device)
    e1 = xnp.update_array(e1, beta, 0)

    if use_triangular:
        # NOTE::: this will not work with multiple rhs Andres to fix
        R, Gs = get_hessenberg_triangular_qr(H[0, :, :], xnp=xnp)
        target = apply_givens_fwd(Gs, e1, xnp)
        y = xnp.solvetri(R, target, lower=False)
        pred = Q[0, :, :] @ y
    else:
        HT = xnp.conj(xnp.permute(H, axes=[0, 2, 1]))
        largest_vals = xnp.max(xnp.abs(H), -1)
        overall_max = xnp.max(largest_vals.reshape(largest_vals.shape[0], -1), -1)
        zero_thresh = 10 * tol * overall_max[:, None]
        padding = xnp.where(largest_vals < zero_thresh, xnp.ones_like(largest_vals), xnp.zeros_like(largest_vals))
        added_diag = xnp.vmap(xnp.diag)(padding)
        y = xnp.solve(HT @ H + added_diag, HT[..., 0]) * beta[:, None]
        zeros = xnp.zeros_like(y)
        y = xnp.where(largest_vals < zero_thresh, zeros, y)
        pred = xnp.permute(Q @ y[..., None], axes=[1, 0, 2])[:, :, 0]

    soln = x0 + pred
    return soln, infodict


def get_hessenberg_triangular_qr(H, xnp):
    device = xnp.get_device(H)
    R = xnp.copy(H)
    Gs = []
    for jdx in range(H.shape[0] - 1):
        cx, sx = get_givens_cos_sin(R[jdx, jdx], R[jdx + 1, jdx], xnp)
        G = xnp.array([[cx, sx], [-sx, cx]], dtype=H.dtype, device=device)
        Gs.append(G)
        update = G.T @ R[[jdx, jdx + 1], :]
        R = xnp.update_array(R, update, [jdx, jdx + 1])
    return R, Gs


def apply_givens_fwd(Gs, vec, xnp):
    for jdx in range(len(Gs)):
        update = Gs[jdx].T @ vec[[jdx, jdx + 1], :]
        vec = xnp.update_array(vec, update, [jdx, jdx + 1])
    return vec


def get_givens_cos_sin(a, b, xnp):
    if b == 0:
        c, s = 1, 0
    else:
        denom = xnp.sqrt(a**2. + b**2.)
        s = -b / denom
        c = a / denom
    return c, s
