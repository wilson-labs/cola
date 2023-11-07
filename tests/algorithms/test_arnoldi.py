import numpy as np
from cola.ops import Householder
from cola.ops import Product
from cola.ops import Dense
from cola.fns import lazify
from cola.linalg.decompositions.arnoldi import ira
from cola.linalg.decompositions.arnoldi import get_arnoldi_matrix
from cola.linalg.decompositions.arnoldi import arnoldi_eigs
from cola.linalg.decompositions.arnoldi import run_householder_arnoldi
from cola.linalg.decompositions.arnoldi import init_arnoldi
from cola.linalg.decompositions.arnoldi import init_arnoldi_from_vec
from cola.utils.test_utils import get_xnp, parametrize, relative_error
from cola.backends import all_backends
from cola.utils.test_utils import generate_spectrum, generate_pd_from_diag
from cola.utils.test_utils import generate_lower_from_diag


@parametrize(all_backends).excluding[:]
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


@parametrize(all_backends)
def test_ira(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10, dtype=np.float32)
    A = xnp.array(generate_lower_from_diag(diag, dtype=diag.dtype, seed=48), dtype=dtype, device=None)
    zr = xnp.randn(A.shape[1], dtype=xnp.float32, device=None, key=xnp.PRNGKey(123))
    rhs = xnp.cast(zr, dtype=dtype)
    V, H = ira(lazify(A), rhs, max_iters=7)


@parametrize(["torch"])
def test_arnoldi_factorization_restarted(backend):
    xnp = get_xnp(backend)
    dtype, np_dtype = xnp.float64, np.float64
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10, dtype=np_dtype)
    A = xnp.array(generate_lower_from_diag(diag, dtype=diag.dtype, seed=48), dtype=dtype, device=None)
    A = lazify(A)
    zr = xnp.randn((A.shape[1], 1), dtype=dtype, device=None, key=xnp.PRNGKey(123))
    rhs = xnp.cast(zr, dtype=dtype)
    init_val = init_arnoldi(xnp, rhs, max_iters=7, dtype=A.dtype)
    V, H, *_ = get_arnoldi_matrix(A, init_val, 3, tol=1e-12, pbar=False)

    V, H = V[0], H[0]
    e_vec = xnp.canonical(2, shape=(3, 1), dtype=dtype, device=None)
    new_vec = (H[3, 2] * V[:, [3]])
    alter = new_vec @ e_vec.T
    approx = A @ V[:, :3]
    soln = V[:, :3] @ H[:3, :3] + alter
    rel_error = relative_error(approx, soln)
    assert rel_error < 1e-12

    init_val = init_arnoldi_from_vec(H, V, xnp, new_vec[:, 0], rest=3)
    V, H, *_ = get_arnoldi_matrix(A, init_val, 7, tol=1e-12, pbar=False)
    V, H = V[0], H[0]
    e_vec = xnp.canonical(H.shape[1] - 1, shape=(H.shape[1], 1), dtype=dtype, device=None)
    new_vec = (H[-1, -1] * V[:, [-1]])
    alter = new_vec @ e_vec.T
    rel_error = relative_error(A @ V[:, :-1], V[:, :-1] @ H[:-1] + alter)
    assert rel_error < 1e-12

    A_np, rhs_np = np.array(A.to_dense(), dtype=np_dtype), np.array(rhs[:, 0], dtype=np_dtype)
    init_val = init_arnoldi_np(rhs_np, max_iter=7, dtype=np_dtype)
    Q_sol, H_sol = run_arnoldi(A_np, init_val, max_iter=7, tol=1e-7)

    for soln, approx in ((Q_sol, V), (H_sol, H)):
        rel_error = relative_error(xnp.array(soln, dtype=dtype, device=None), approx)
        assert rel_error < 1e-12


@parametrize(all_backends)
def test_arnoldi(backend):
    xnp = get_xnp(backend)
    dtype = xnp.complex64
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=4, dtype=np.float32)
    A = xnp.array(generate_lower_from_diag(diag, dtype=diag.dtype, seed=48), dtype=dtype, device=None)
    zr = xnp.randn(A.shape[1], dtype=xnp.float32, device=None, key=xnp.PRNGKey(123))
    rhs = xnp.cast(zr, dtype=dtype)
    eigvals, eigvecs, _ = arnoldi_eigs(lazify(A), rhs, max_iters=A.shape[-1])
    approx = xnp.sort(xnp.cast(eigvals, xnp.float32))
    soln = xnp.sort(xnp.array(diag, xnp.float32, device=None))

    rel_error = relative_error(soln, approx)
    assert rel_error < 1e-3

    approx = eigvecs @ xnp.diag(eigvals) @ xnp.inv(eigvecs.to_dense())
    rel_error = relative_error(A, approx)
    assert rel_error < 1e-3


@parametrize(['torch'])
def ignore_test_householder_arnoldi_decomp(backend):
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


@parametrize(["torch"])
def test_arnoldi_factorization(backend):
    xnp = get_xnp(backend)
    dtype = xnp.complex128  # double precision on real and complex coordinates to achieve 1e-12 tol
    np_dtype = np.complex128
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=20, dtype=np.float32) - 0.5
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype, seed=21), dtype=dtype, device=None)
    rhs = xnp.randn(A.shape[1], 1, dtype=dtype, device=None, key=xnp.PRNGKey(1256))
    rhs = xnp.concat((rhs, rhs), axis=-1)
    max_iter = A.shape[0]
    A_np, rhs_np = np.array(A, dtype=np_dtype), np.array(rhs[:, 0], dtype=np_dtype)
    init_val = init_arnoldi_np(rhs_np, max_iter=max_iter, dtype=np_dtype)
    Q_sol, H_sol = run_arnoldi(A_np, init_val, max_iter=max_iter, tol=1e-7)

    init_val = init_arnoldi(xnp, rhs, max_iters=max_iter, dtype=A.dtype)
    Q_approx, H_approx, *_ = get_arnoldi_matrix(lazify(A), init_val, max_iter, tol=1e-12, pbar=False)
    rel_error = relative_error(Q_approx[0], Q_approx[1])
    rel_error += relative_error(H_approx[0], H_approx[1])
    assert rel_error < 1e-12

    Q_approx, H_approx = Q_approx[0, :, :-1], H_approx[0, :-1]
    for soln, approx in ((Q_sol[:, :-1], Q_approx), (H_sol[:-1], H_approx)):
        rel_error = relative_error(xnp.array(soln, dtype=dtype, device=None), approx)
        assert rel_error < 1e-12

    rel_error = relative_error(Q_approx.conj().T @ A @ Q_approx, H_approx)
    assert rel_error < 1e-12

    max_iter = 10
    init_val = init_arnoldi(xnp, rhs, max_iters=max_iter, dtype=A.dtype)
    Q_approx, H_approx, *_ = get_arnoldi_matrix(lazify(A), init_val, max_iter, tol=1e-12, pbar=False)
    Q_approx, H_approx = Q_approx[0], H_approx[0]
    e_vec = xnp.canonical(max_iter - 1, shape=(max_iter, 1), dtype=dtype, device=None)
    alter = (H_approx[-1, -1] * Q_approx[:, [-1]]) @ e_vec.T
    Q_approx, H_approx = Q_approx[:, :-1], H_approx[:-1]
    rel_error = relative_error(A @ Q_approx, Q_approx @ H_approx + alter)
    assert rel_error < 1e-12


def test_numpy_iram():
    np.set_printoptions(formatter={"float": "{:0.2f}".format})
    np_dtype = np.float64
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10, dtype=np_dtype)
    A = np.array(generate_pd_from_diag(diag, dtype=diag.dtype, seed=48), dtype=np_dtype)
    rhs = np.random.normal(size=(A.shape[0], ))
    V, H = run_iram(A, rhs, eig_n=5, max_size=8, max_iter=100, tol=1e-12)
    abs_error = np.linalg.norm(A @ V - V @ H)
    print(f"\nAbs error: {abs_error:1.2e}")
    assert abs_error < 1e-12


def test_numpy_arnoldi():
    np.set_printoptions(formatter={"float": "{:0.2f}".format})
    np_dtype = np.float64
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10, dtype=np_dtype)
    A = np.array(generate_lower_from_diag(diag, dtype=diag.dtype, seed=48), dtype=np_dtype)
    rhs = np.random.normal(size=(A.shape[0], ))

    init_val = init_arnoldi_np(rhs, max_iter=A.shape[0] - 2, dtype=np_dtype)
    Q, H = run_arnoldi(A, init_val, max_iter=A.shape[0] - 2, tol=1e-12)

    part1, part2 = check_arnoldi_fact_np(Q, H, A, iter=A.shape[0] - 2)
    abs_error = np.linalg.norm(part1 - part2)
    print(f"Abs error: {abs_error:1.2e}")
    assert abs_error < 1e-12

    init_val = init_arnoldi_np(rhs, max_iter=A.shape[0], dtype=np_dtype)
    Q, H = run_arnoldi(A, init_val, max_iter=A.shape[0], tol=1e-12)
    abs_error = np.linalg.norm(A @ Q[:, :-1] - Q[:, :-1] @ H[:-1])
    print(f"Abs error: {abs_error:1.2e}")
    assert abs_error < 1e-10

    init_val = init_arnoldi_np(rhs, max_iter=8, dtype=np_dtype)
    Q_sol, H_sol = run_arnoldi(A, init_val, max_iter=8, tol=1e-12)
    init_val = init_arnoldi_np(rhs, max_iter=8, dtype=np_dtype)
    Q, H = run_arnoldi(A, init_val, max_iter=5, tol=1e-12)
    vec = H[5, 4] * Q[:, [5]]
    init_val = init_arnoldi_from_vec_np(Q, H, vec, max_iter=8, idx=5)
    Q, H = run_arnoldi(A, init_val, max_iter=8, tol=1e-12)
    for soln, approx in ((Q_sol, Q), (H_sol, H)):
        rel_error = relative_error(soln, approx)
        print(f"Rel error: {abs_error:1.2e}")
        assert rel_error < 1e-12


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


def run_iram(A, rhs, eig_n, max_size, max_iter, tol):
    init_val = init_arnoldi_np(rhs, max_size, A.dtype)
    norm, nq, counter = 2 * tol, max_size - eig_n, 0
    while (counter < max_iter) & (norm > tol):
        V, H = run_arnoldi(A, init_val, max_iter=max_size, tol=tol)
        # part1, part2 = check_arnoldi_fact_np(V, H, A, iter=max_size)
        eigvals, _ = np.linalg.eig(H[:-1])
        eigvals = np.sort(eigvals)
        vec = np.copy(H[-1, -1] * V[:, [-1]])
        H, Q = run_shift_np(H[:-1].copy(), eigvals[:nq])

        # V_new = V[:, :-1] @ Q
        # part1 = A @ V_new
        # part2 = V_new @ H
        # part2 += vec @ Q[[-1], :]
        # diff = np.linalg.norm(part1 - part2)
        # print(f"Abs error: {diff:1.2e}")

        beta = H[eig_n, eig_n - 1]
        sigma = Q[-1, eig_n - 1]
        new_vec = beta * V[:, [eig_n]] + sigma * vec
        V0 = V[:, :-1] @ Q[:, :eig_n]
        H0 = H[:eig_n, :eig_n]

        # part1 = A @ V0
        # e_vec = np.zeros(shape=(5, 1), dtype=H.dtype)
        # e_vec[4] = 1.0
        # extra = new_vec @ e_vec.T
        # part2 = V0 @ H0
        # part2 += extra
        # diff = np.linalg.norm(part1 - part2)
        # print(f"Abs error: {diff:1.2e}")

        init_val = init_arnoldi_from_vec_np(V0, H0, new_vec, max_iter=max_size, idx=eig_n)

        norm = np.linalg.norm(A @ V0 - V0 @ H0)
        counter += 1
    return V0, H0


def run_shift_np(A, shifts):
    Q, Id = np.eye(A.shape[0], dtype=A.dtype), np.eye(A.shape[0], dtype=A.dtype)
    for jdx in range(len(shifts)):
        Q1, _ = np.linalg.qr(A - shifts[jdx] * Id, mode="complete")
        A = Q1.T @ A @ Q1
        Q = Q @ Q1
    return A, Q


def run_arnoldi(A, init_val, max_iter, tol):
    Q, H, idx, norm = init_val

    while (idx < max_iter) & (norm > tol):
        vec = A @ Q[:, idx]
        for jdx in range(idx + 1):
            H[jdx, idx] = Q[:, jdx].conj().T @ vec
            vec -= H[jdx, idx] * Q[:, jdx]
        norm = np.linalg.norm(vec)
        H[idx + 1, idx] = norm
        Q[:, idx + 1] = vec / norm
        idx += 1
    return Q, H


def init_arnoldi_from_vec_np(Q0, H0, vec, max_iter, idx):
    dtype = H0.dtype
    H1 = np.zeros(shape=(max_iter + 1, max_iter), dtype=dtype)
    Q1 = np.zeros(shape=(vec.shape[0], max_iter + 1), dtype=dtype)
    H1[:idx, :idx] = H0[:idx, :idx].copy()
    norm = np.linalg.norm(vec)
    H1[idx, idx - 1] = norm
    Q1[:, :idx] = Q0[:, :idx].copy()
    Q1[:, [idx]] = vec / norm
    return Q1, H1, idx, norm


def init_arnoldi_np(rhs, max_iter, dtype):
    H = np.zeros(shape=(max_iter + 1, max_iter), dtype=dtype)
    Q = np.zeros(shape=(rhs.shape[0], max_iter + 1), dtype=dtype)
    idx, vec = 0, rhs.copy()
    norm = np.linalg.norm(vec)
    Q[:, idx] = vec / norm
    return Q, H, idx, norm


def check_arnoldi_fact_np(Q, H, A, iter):
    vec = H[iter, iter - 1] * Q[:, [iter]]
    e_vec = np.zeros(shape=(iter, 1), dtype=H.dtype)
    e_vec[iter - 1] = 1.0
    extra = vec @ e_vec.T
    part1 = A @ Q[:, :iter]
    part2 = Q[:, :iter] @ H[:iter, :iter]
    part2 += extra
    return part1, part2
