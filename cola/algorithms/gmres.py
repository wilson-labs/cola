from cola.ops import LinearOperator
from cola.ops import Array
from cola.algorithms.arnoldi import run_householder_arnoldi
from cola.algorithms.arnoldi import get_arnoldi_matrix
from cola.utils import export


@export
def gmres(A: LinearOperator, rhs: Array, x0=None, max_iters=None, tol=1e-7, P=None,
          use_householder=False, use_triangular=False, pbar=False):
    """Solves a linear system Ax = rhs using the GMRES method.

    Args:
        A (LinearOperator): The linear operator representing the matrix A.
        rhs (Array): The right-hand side vector.
        x0 (Array, optional): The initial guess for the solution. Defaults to None.
        max_iters (int, optional): The maximum number of iterations. Defaults to None.
        tol (float, optional): The tolerance for convergence. Defaults to 1e-7.
        P (array, optional): Preconditioner. Defaults to None.
        use_householder (bool, optional): Use Householder Arnoldi iteration. Defaults to False.
        use_triangular (bool, optional): Use triangular QR factorization. Defaults to False.
        pbar (bool, optional): show a progress bar. Defaults to False.

    Returns:
        (tuple): (x, info) The solution vector x, satisfying Ax = rhs and the info dictionary.
    """
    xnp = A.ops
    is_vector = len(rhs.shape) == 1
    if x0 is None:
        x0 = xnp.zeros_like(rhs)
    if is_vector:
        rhs = rhs[..., None]
        x0 = x0[..., None]
    res = rhs - A @ x0
    if use_householder:
        Q, H, infodict = run_householder_arnoldi(A=A, rhs=res, max_iters=max_iters)
    else:
        Q, H, _, infodict = get_arnoldi_matrix(A=A, rhs=res, max_iters=max_iters, tol=tol,
                                               pbar=pbar)
    beta = xnp.norm(res, axis=-2)
    e1 = xnp.zeros(shape=(H.shape[0], beta.shape[0]), dtype=rhs.dtype)
    e1 = xnp.update_array(e1, beta, 0)

    if use_triangular:
        R, Gs = get_hessenberg_triangular_qr(H[:, :, 0], xnp=xnp)
        target = apply_givens_fwd(Gs, e1, xnp)
        y = xnp.solvetri(R, target, lower=False)
        pred = Q[:, :, 0] @ y
    else:
        nH = xnp.permute(H, axes=[2, 0, 1])
        nHT = xnp.permute(H, axes=[2, 1, 0])
        ne1 = xnp.permute(e1[None], axes=[2, 1, 0])
        nQ = xnp.permute(Q, axes=[2, 0, 1])

        y = xnp.solve(nHT @ nH, nHT @ ne1)
        pred = xnp.permute(nQ @ y, axes=[1, 0, 2])[:, :, 0]

    soln = x0 + pred
    if is_vector:
        soln = soln[:, 0]
    return soln, infodict


def get_hessenberg_triangular_qr(H, xnp):
    R = xnp.copy(H)
    Gs = []
    for jdx in range(H.shape[0] - 1):
        cx, sx = get_givens_cos_sin(R[jdx, jdx], R[jdx + 1, jdx], xnp)
        G = xnp.array([[cx, sx], [-sx, cx]], dtype=H.dtype)
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
