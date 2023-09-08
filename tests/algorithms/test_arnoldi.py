import numpy as np
from cola.ops import Householder
from cola.ops import Product
from cola.ops import Dense
from cola.fns import lazify
from cola.algorithms.arnoldi import get_arnoldi_matrix
from cola.algorithms.arnoldi import arnoldi_eigs
from cola.algorithms.arnoldi import run_householder_arnoldi
from cola.utils_test import get_xnp, parametrize, relative_error
from cola.utils_test import generate_spectrum, generate_pd_from_diag
from cola.utils_test import generate_lower_from_diag


@parametrize(['torch', 'jax']).excluding[:]
def test_arnoldi_vjp(backend):
    if backend == 'torch':
        import torch
        torch.manual_seed(seed=21)

    xnp = get_xnp(backend)
    dtype = xnp.float64
    matrix = [[6., 2., 3.], [2., 3., 1.], [3., 1., 4.]]
    diag = xnp.Parameter(xnp.array(matrix, dtype=dtype, device=None))
    diag_soln = xnp.Parameter(xnp.array(matrix, dtype=dtype, device=None))
    _, unflatten = Dense(diag).flatten()
    x0 = xnp.randn(diag.shape[0], 1, dtype=dtype, device=None, key=xnp.PRNGKey(21))

    def f(theta):
        Aop = unflatten([theta])
        eig_vals, *_ = arnoldi_eigs(Aop, x0, max_iters=10, tol=1e-6, pbar=False)
        loss = xnp.sum(xnp.abs(eig_vals)**2.)
        return loss

    def f_alt(theta):
        A = theta
        eig_vals, _ = xnp.eigh(A)
        loss = xnp.sum(xnp.abs(eig_vals)**2.)
        return loss

    out = f(diag)
    print(out)
    if backend == 'torch':
        out.backward()
        approx = diag.grad.clone()
    else:
        approx = xnp.grad(f)(diag)
    assert approx is not None

    out = f_alt(diag_soln)
    print(out)
    if backend == 'torch':
        out.backward()
        soln = diag_soln.grad.clone()
    else:
        soln = xnp.grad(f_alt)(diag)

    print(approx)
    print(soln)
    abs_error = xnp.norm(soln - approx)
    assert abs_error < 5e-5


@parametrize(['torch', 'jax'])
def test_arnoldi(backend):
    xnp = get_xnp(backend)
    dtype = xnp.complex64
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=4, dtype=np.float32)
    A = xnp.array(generate_lower_from_diag(diag, dtype=diag.dtype, seed=48), dtype=dtype,
                  device=None)
    zr = xnp.randn(A.shape[1], 1, dtype=xnp.float32, device=None, key=xnp.PRNGKey(123))
    rhs = xnp.cast(zr, dtype=dtype)
    eigvals, eigvecs, _ = arnoldi_eigs(lazify(A), rhs, max_iters=A.shape[-1])
    approx = xnp.sort(xnp.cast(eigvals, xnp.float32))
    soln = xnp.sort(xnp.array(diag, xnp.float32, device=None))

    rel_error = relative_error(soln, approx)
    assert rel_error < 1e-3

    approx = eigvecs @ xnp.diag(eigvals) @ xnp.inv(eigvecs.to_dense())
    rel_error = relative_error(A, approx)
    assert rel_error < 1e-3


@parametrize(['jax'])
def test_householder_arnoldi_decomp(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10, dtype=np.float32) - 0.5
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype, seed=21), dtype=dtype, device=None)
    rhs = xnp.randn(A.shape[1], 1, dtype=dtype, device=None, key=xnp.PRNGKey(48))
    # A_np, rhs_np = np.array(A, dtype=np.complex128), np.array(rhs[:, 0], dtype=np.complex128)
    A_np, rhs_np = np.array(A, dtype=np.float64), np.array(rhs[:, 0], dtype=np.float64)
    # Q_sol, H_sol = run_householder_arnoldi(A, rhs, A.shape[0], np.float64, xnp)
    Q_sol, H_sol = run_householder_arnoldi_np(A_np, rhs_np, A.shape[0], np.float64)

    # fn = run_householder_arnoldi
    fn = xnp.jit(run_householder_arnoldi, static_argnums=(0, 2))
    Q_approx, H_approx, _ = fn(lazify(A), rhs, max_iters=A.shape[0])

    for soln, approx in ((Q_sol, Q_approx), (H_sol, H_approx)):
        rel_error = relative_error(xnp.array(soln, dtype=dtype, device=None), approx)
        assert rel_error < 1e-5


@parametrize(['torch'])  # jax does not have complex128 # yeah it does ?
def test_get_arnoldi_matrix(backend):
    xnp = get_xnp(backend)
    dtype = xnp.complex128  # double precision on real and complex coordinates to achieve 1e-12 tol
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=20, dtype=np.float32) - 0.5
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype, seed=21), dtype=dtype, device=None)
    rhs = xnp.randn(A.shape[1], 1, dtype=dtype, device=None, key=xnp.PRNGKey(1256))
    rhs = xnp.concatenate((rhs, rhs), axis=-1)
    max_iter = A.shape[0]
    A_np, rhs_np = np.array(A, dtype=np.complex128), np.array(rhs[:, 0], dtype=np.complex128)
    Q_sol, H_sol = run_arnoldi(A_np, rhs_np, max_iter=max_iter, tol=1e-7, dtype=np.complex128)

    fn = xnp.jit(get_arnoldi_matrix, static_argnums=(0, 2, 3, 4))
    Q_approx, H_approx, *_ = fn(lazify(A), rhs, max_iter, tol=1e-12, pbar=False)

    rel_error = relative_error(Q_approx[:, :, 0], Q_approx[:, :, 1])
    rel_error += relative_error(H_approx[:, :, 0], H_approx[:, :, 1])
    assert rel_error < 1e-12

    Q_approx, H_approx = Q_approx[:, :, 0], H_approx[:, :, 0]
    for soln, approx in ((Q_sol[:, :-1], Q_approx), (H_sol[:-1, :], H_approx)):
        rel_error = relative_error(xnp.array(soln, dtype=dtype, device=None), approx)
        assert rel_error < 1e-12

    # Next test is only valid when full decomposition is ran
    rel_error = relative_error(A @ Q_approx, Q_approx @ H_approx)
    assert rel_error < 1e-12
    rel_error = relative_error(Q_approx.conj().T @ A @ Q_approx, H_approx)
    assert rel_error < 1e-12


@parametrize(['jax'])
def test_numpy_arnoldi(backend):
    xnp = get_xnp(backend)
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind': float_formatter})
    # dtype = np.complex64
    dtype = np.float64
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10, dtype=np.float32)
    A = np.array(generate_lower_from_diag(diag, dtype=diag.dtype, seed=48), dtype=dtype)
    rhs = np.random.normal(size=(A.shape[0], ))
    # rhs = np.random.normal(size=(A.shape[0], 2)).view(np.complex128)[:, 0]

    Q, H = run_householder_arnoldi_np(A, rhs, max_iter=A.shape[0], dtype=dtype)
    abs_error = np.linalg.norm(np.eye(A.shape[0]) - Q.T @ Q)
    assert abs_error < 1e-4
    abs_error = np.linalg.norm(Q.T @ A @ Q - H)
    assert abs_error < 1e-6

    Q, H = run_arnoldi(A, rhs, max_iter=A.shape[0] - 2, tol=1e-12, dtype=dtype)
    abs_error = np.linalg.norm(A @ Q[:, :-1] - Q @ H)
    assert abs_error < 1e-10
    abs_error = np.linalg.norm(Q[:, :-1].conj().T @ A @ Q[:, :-1] - H[:-1, :])
    assert abs_error < 1e-10


def run_householder_arnoldi_np(A, rhs, max_iter, dtype):
    H, Q, Ps, zj = initialize_householder_arnoldi(rhs, max_iter, dtype)
    for jdx in range(1, max_iter + 2):
        vec, beta = get_householder_vec_np(zj, jdx - 1)
        Ps[jdx].vec, Ps[jdx].beta = vec[:, None], beta
        H[:, jdx - 1] = np.array(Ps[jdx] @ zj)
        if jdx <= max_iter:
            Q[jdx - 1, jdx] = 1.
            Reflect = Product(*[Ps[idx] for idx in range(1, jdx + 1)])
            # Reflect = Product([Ps[idx] for idx in range(1, max_iter + 1)])
            Q[:, jdx] = Reflect @ Q[:, jdx]
            Reflect = Product(*[Ps[idx] for idx in range(jdx + 1, 0, -1)])
            # Reflect = Product([Ps[idx] for idx in range(max_iter + 1, 0, -1)])
            zj = Reflect @ A @ Q[:, jdx]
    return Q[:, 1:], H[:, 1:]


def initialize_householder_arnoldi(rhs, max_iter, dtype):
    H = np.zeros(shape=(max_iter, max_iter + 1), dtype=dtype)
    Q = np.zeros(shape=(max_iter, max_iter + 1), dtype=dtype)
    Q[:, 0] = rhs.copy()
    Q[:, 0] = Q[:, 0] / np.linalg.norm(Q[:, 0])
    Ps = [Householder(np.zeros((max_iter, 1), dtype=dtype)) for _ in range(max_iter + 2)]
    zj = Q[:, 0]
    return H, Q, Ps, zj


def get_householder_vec_np(x, idx):
    sigma_2 = np.linalg.norm(x[idx + 1:])**2.
    vec = np.zeros_like(x)
    vec[idx:] = x[idx:]
    if sigma_2 == 0 and x[idx] >= 0:
        beta = 0
    elif sigma_2 == 0 and x[idx] < 0:
        beta = -2
    else:
        x_norm_partial = np.sqrt(x[idx]**2 + sigma_2)
        if x[idx] <= 0:
            vec[idx] = x[idx] - x_norm_partial
        else:
            vec[idx] = -sigma_2 / (x[idx] + x_norm_partial)
        beta = 2 * vec[idx]**2 / (sigma_2 + vec[idx]**2)
        vec = vec / vec[idx]
        vec[idx:] = vec[idx:] / vec[idx]
    return vec, beta


def run_arnoldi(A, rhs, max_iter, tol, dtype):
    Q, H = initialize_arnoldi(rhs, max_iter=max_iter, dtype=dtype)
    idx, vec = 0, rhs.copy()
    norm = np.linalg.norm(vec)

    while (idx <= max_iter) & (norm > tol):
        Q[:, idx] = vec / norm
        vec = A @ Q[:, idx]
        for jdx in range(idx + 1):
            H[jdx, idx] = Q[:, jdx].conj().T @ vec
            vec -= H[jdx, idx] * Q[:, jdx]
        norm = np.linalg.norm(vec)
        H[idx + 1, idx] = norm
        idx += 1
    return Q, H[:max_iter + 1, :max_iter]


def initialize_arnoldi(rhs, max_iter, dtype):
    H = np.zeros(shape=(max_iter + 2, max_iter + 2), dtype=dtype)
    Q = np.zeros(shape=(rhs.shape[0], max_iter + 1), dtype=dtype)
    return Q, H
