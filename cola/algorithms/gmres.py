from linops.operator_base import LinearOperator
from linops.operator_base import Array
from linops.algorithms.arnoldi import run_householder_arnoldi
from linops.algorithms.arnoldi import get_arnoldi_matrix


def run_gmres(A: LinearOperator, rhs: Array, x0=None, max_iters=None, tol=1e-7, P=None,
              use_householder=False, use_triangular=False, pbar=False, info=False):
    xnp = A.ops
    is_vector = len(rhs.shape) == 1
    if x0 is None:
        x0 = xnp.zeros_like(rhs)
    if is_vector:
        rhs = rhs[..., None]
        x0 = x0[..., None]
    res = rhs - A @ x0
    if use_householder:
        Q, H = run_householder_arnoldi(A=A, rhs=res, max_iters=max_iters)
    else:
        Q, H, _ = get_arnoldi_matrix(A=A, rhs=res, max_iters=max_iters, tol=tol)
        Q = Q[:, :-1]
    beta = xnp.norm(res, axis=-2)
    e1 = xnp.zeros(shape=(H.shape[0], 1), dtype=rhs.dtype)
    e1 = xnp.update_array(e1, beta, 0)

    if use_triangular:
        R, Gs = get_hessenberg_triangular_qr(H, xnp=xnp)
        target = apply_givens_fwd(Gs, e1, xnp)
        if use_householder:
            y = xnp.solvetri(R, target, lower=False)
        else:
            y = xnp.solvetri(R[:-1, :], target[:-1, :], lower=False)
    else:
        y = xnp.solve(H.T @ H, H.T @ e1)
    soln = x0 + Q @ y
    if is_vector:
        soln = soln[:, 0]
    return soln


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
