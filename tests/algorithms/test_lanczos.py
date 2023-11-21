import numpy as np
from cola.fns import lazify
from cola.ops import Dense
from cola.ops import Tridiagonal
from cola.linalg.decompositions.lanczos import irl
from cola.linalg.decompositions.lanczos import lanczos
from cola.linalg.decompositions.lanczos import lanczos_eigs
from cola.backends import all_backends, tracing_backends
from cola.utils.test_utils import get_xnp, parametrize, relative_error
from cola.utils.test_utils import generate_spectrum, generate_pd_from_diag
from cola.utils.test_utils import generate_diagonals
from cola.utils.test_utils import get_numpy_dtype

_tol = 1e-6


@parametrize(tracing_backends).excluding[:]  # disabled before we fix new lanczos outputs
def test_lanczos_vjp(backend):
    if backend == 'torch':
        import torch
        torch.manual_seed(seed=21)

    xnp = get_xnp(backend)
    dtype = xnp.float64
    # diag = xnp.Parameter(xnp.array([3., 4., 5.], dtype=dtype))
    # diag_soln = xnp.Parameter(xnp.array([3., 4., 5.], dtype=dtype))
    # _, unflatten = Diagonal(diag).flatten()
    # matrix = [[10., 2., 3.], [2., 14., 1.], [3., 1., 11.]]
    matrix = [[6., 2., 3.], [2., 3., 1.], [3., 1., 4.]]
    # matrix = [[3., 0., 0.], [0., 4., 0.], [0., 0., 5.]]
    diag = xnp.Parameter(xnp.array(matrix, dtype=dtype, device=None))
    diag_soln = xnp.Parameter(xnp.array(matrix, dtype=dtype, device=None))
    _, unflatten = Dense(diag).flatten()
    x0 = xnp.randn(diag.shape[0], 1, dtype=dtype, device=None)

    def f(theta):
        Aop = unflatten([theta])
        out = lanczos_eigs(Aop, x0, max_iters=10, tol=1e-6, pbar=False)
        eig_vals, eig_vecs, _ = out
        # loss = xnp.sum(eig_vals ** 2.) + xnp.sum(xnp.abs(eig_vecs), axis=[0, 1])
        loss = xnp.sum(eig_vals**2.)
        # loss = xnp.sum(eig_vecs ** 2., axis=[0, 1])
        # loss = xnp.sum(eig_vecs, axis=[0, 1])
        return loss

    def f_alt(theta):
        # A = xnp.diag(theta)
        A = theta
        eig_vals, eig_vecs = xnp.eigh(A)
        # loss = xnp.sum(eig_vals ** 2.) + xnp.sum(xnp.abs(eig_vecs), axis=[0, 1])
        loss = xnp.sum(eig_vals**2.)
        # loss = xnp.sum(eig_vecs ** 2., axis=[0, 1])
        # loss = xnp.sum(eig_vecs, axis=[0, 1])
        return loss

    out = f(diag)
    if backend == 'torch':
        out.backward()
        approx = diag.grad.clone()
    else:
        approx = xnp.grad(f)(diag)
    assert approx is not None

    out = f_alt(diag_soln)
    if backend == 'torch':
        out.backward()
        soln = diag_soln.grad.clone()
    else:
        soln = xnp.grad(f_alt)(diag)

    print(approx)
    print(soln)
    abs_error = xnp.norm(soln - approx)
    assert abs_error < _tol * 50


@parametrize(["torch"])
def _test_irl(backend):
    # TODO: working on it
    xnp = get_xnp(backend)
    dtype = xnp.float64
    np_dtype = get_numpy_dtype(dtype)
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10, dtype=np_dtype)
    A_np = generate_pd_from_diag(diag, dtype=np_dtype, seed=48)
    A = lazify(xnp.array(A_np, dtype=dtype, device=None))
    rhs = xnp.randn(A.shape[0], 1, dtype=dtype, device=None, key=xnp.PRNGKey(123))
    eig_n, max_size, max_iter, tol = 5, 8, 100, 1e-10
    V_sol, T_sol = run_irl(A_np, np.array(rhs[:, 0]), eig_n, max_size, max_iter, tol)
    approx, _ = np.linalg.eigh(T_sol)
    abs_error = np.linalg.norm(np.sort(diag[:eig_n]) - approx)
    print(f"\nAbs error: {abs_error:1.2e}")
    assert abs_error < 1e-10

    V, beta, alpha, idx, _ = irl(A, rhs, eig_n=eig_n, max_size=max_size, max_iters=max_iter, tol=tol)
    breakpoint()
    T = Tridiagonal(alpha[0, 1:idx], beta[0, :idx], alpha[0, 1:idx]).to_dense()
    V, T = V[0, :, :eig_n], T[:eig_n, :eig_n]

    abs_error = xnp.norm(A @ V - V @ T)
    print(f"\nAbs error: {abs_error:1.2e}")
    assert abs_error < 1e-10

    for soln, approx in ((V_sol, V), (T_sol, T)):
        rel_error = relative_error(soln, np.array(approx))
        print(f"Rel error: {rel_error:1.2e}")
        assert rel_error < 1e-10

    eigvals, _ = xnp.eigh(T)
    eigvals = xnp.sort(eigvals)
    rel_error = relative_error(np.sort(diag[:eig_n]), np.array(eigvals))
    print(f"Rel error: {rel_error:1.2e}")
    assert rel_error < 1e-11


@parametrize(all_backends)
def test_lanczos_complex(backend):
    xnp = get_xnp(backend)
    dtype = xnp.complex64
    np_dtype = get_numpy_dtype(dtype)
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10, dtype=np_dtype)
    A = xnp.array(generate_diagonals(diag, seed=21), dtype=dtype, device=None)
    A = A @ A.conj().T
    rhs = xnp.randn(A.shape[0], 1, dtype=dtype, device=None)
    Q_np, beta_np, alpha_np, T_np = case_numpy(A, rhs, xnp, np_dtype)

    B = lazify(A)
    max_iters, tol = A.shape[0], 1e-7
    Q, T, info = lanczos(B, rhs, max_iters=max_iters, tol=tol, pbar=False)
    alpha, beta = T.alpha[:, :, 0], T.beta[:, :, 0]
    Q, T = Q.to_dense(), xnp.vmap(T.__class__.to_dense)(T)
    Q, T = Q[0], T[0]

    comparisons = [
        (T_np, T),
        (Q_np, Q),
        (Q @ T, A @ Q),
        (alpha_np, alpha[0]),
        (beta_np, beta[0]),
    ]
    for soln, approx in comparisons:
        rel_error = relative_error(soln, approx)
        print(f"rel error: {rel_error:1.3e}")
        assert rel_error < 5e-5


@parametrize(all_backends)
def test_lanczos_random(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    np_dtype = get_numpy_dtype(dtype)
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10, dtype=np_dtype)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype, seed=21), dtype=dtype, device=None)
    rhs = xnp.ones(shape=(A.shape[0], 1), dtype=dtype, device=None)
    Q_np, beta_np, alpha_np, T_np = case_numpy(A, rhs, xnp, np_dtype)

    B, max_iters, tol = lazify(A), A.shape[0], 1e-7
    Q, T, info = lanczos(B, rhs, max_iters=max_iters, tol=tol, pbar=False)
    alpha, beta = T.alpha[:, :, 0], T.beta[:, :, 0]
    Q, T = Q.to_dense(), xnp.vmap(T.__class__.to_dense)(T)

    comparisons = [
        (T_np, T),
        (Q_np, Q),
        (Q @ T, A @ Q),
        (alpha_np, alpha[0]),
        (beta_np, beta[0]),
    ]
    for soln, approx in comparisons:
        rel_error = relative_error(soln, approx)
        assert rel_error < 5e-5


@parametrize(all_backends)
def test_lanczos_manual(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    cases = [case_early, case_3, case_2]
    for case in cases:
        out = case(xnp, dtype)
        A, rhs, beta_soln, alpha_soln, idx_soln = out

        max_iters, tol = A.shape[0], 1e-7
        B = lazify(A)
        Q, T, info = lanczos(B, rhs, max_iters=max_iters, tol=tol, pbar=False)
        alpha, beta = T.alpha[:, :, 0], T.beta[:, :, 0]
        Q, T = Q.to_dense(), xnp.vmap(T.__class__.to_dense)(T)
        idx = info["iterations"] - 1
        if backend == "jax":
            alpha, beta = alpha[:, :idx - 1], beta[:, :idx]

        assert idx == idx_soln
        rel_error = relative_error(beta_soln, beta.T)
        assert rel_error < _tol
        rel_error = relative_error(alpha_soln, alpha.T)
        assert rel_error < _tol


@parametrize(all_backends)
def test_lanczos_iter(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    np_dtype = get_numpy_dtype(dtype)
    max_eig = 6
    A = xnp.diag(xnp.array([4, 2, 1, max_eig], dtype=dtype, device=None))
    rhs = xnp.ones(shape=(A.shape[0], 7), dtype=dtype, device=None)
    Q_np, beta_np, alpha_np, T_np = case_numpy(A, rhs, xnp, np_dtype)

    max_iters, tol = A.shape[0], 1e-7
    B = lazify(A)
    Q, T, info = lanczos(B, rhs, max_iters=max_iters, tol=tol, pbar=False)
    Q = Q.to_dense()
    alpha, beta = T.alpha[:, :, 0], T.beta[:, :, 0]
    T = xnp.vmap(T.__class__.to_dense)(T)
    eigvals, _ = xnp.eigh(T)

    comparisons = [(T_np, T), (Q_np, Q), (Q @ T, A @ Q), (alpha_np, alpha), (beta_np, beta),
                   (xnp.array(max_eig, dtype=dtype, device=None), eigvals[0, -1])]
    for soln, check in comparisons:
        rel_error = relative_error(soln, check)
        assert rel_error < _tol


def test_numpy_lanczos():
    np.set_printoptions(formatter={"float": "{:0.2f}".format})
    np_dtype = np.float64
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10, dtype=np_dtype)
    A = np.array(generate_pd_from_diag(diag, dtype=diag.dtype, seed=48), dtype=np_dtype)
    rhs = np.random.normal(size=(A.shape[0], ))

    init_val = init_lanczos_np(rhs, max_iter=A.shape[0] - 2, dtype=np_dtype)
    Q, diag, subdiag, idx = run_lanczos(A, init_val, max_iter=A.shape[0] - 2, tol=1e-12)
    T = construct_tri(subdiag[1:idx], diag[:idx])
    V = Q[:, 1:idx + 1]

    part1 = V.T @ A @ V
    part2 = T
    abs_error = np.linalg.norm(part1 - part2)
    print(f"Abs error: {abs_error:1.2e}")
    assert abs_error < 1e-12

    part1 = A @ V - V @ T
    part2 = Q[:, -1] * subdiag[-1]
    abs_error = np.linalg.norm(part1[:, -1] - part2)
    print(f"Abs error: {abs_error:1.2e}")
    assert abs_error < 1e-12

    init_val = init_lanczos_np(rhs, max_iter=A.shape[0], dtype=np_dtype)
    Q, diag, subdiag, idx = run_lanczos(A, init_val, max_iter=A.shape[0], tol=1e-12)
    Q = Q[:, 1:idx + 1]
    T = construct_tri(subdiag[1:idx], diag[:idx])
    abs_error = np.linalg.norm(A @ Q - Q @ T)
    print(f"Abs error: {abs_error:1.2e}")
    assert abs_error < 1e-10

    init_val = init_lanczos_np(rhs, max_iter=8, dtype=np_dtype)
    Q_sol, diag, subdiag, idx = run_lanczos(A, init_val, max_iter=8, tol=1e-12)
    T_sol = construct_tri(subdiag[1:idx], diag[:idx])
    init_val = init_lanczos_np(rhs, max_iter=8, dtype=np_dtype)
    Q, diag, subdiag, idx = run_lanczos(A, init_val, max_iter=5, tol=1e-12)
    T = construct_tri(subdiag[1:idx], diag[:idx])
    vec = subdiag[idx] * Q[:, [idx + 1]]
    init_val = init_lanczos_from_vec_np(Q[:, 1:], T, vec, max_iter=8, idx=idx)
    Q, diag, subdiag, idx = run_lanczos(A, init_val, max_iter=8, tol=1e-12)
    T = construct_tri(subdiag[1:idx], diag[:idx])
    for soln, approx in ((Q_sol, Q), (T_sol, T)):
        rel_error = relative_error(soln, approx)
        print(f"Rel error: {abs_error:1.2e}")
        assert rel_error < 1e-12


def case_early(xnp, dtype):
    A = xnp.diag(xnp.array([4, 2, 1], dtype=dtype, device=None))
    # beta_soln = xnp.array([[4, 0., 0.]], dtype=dtype, device=None).T
    # alpha_soln = xnp.array([[0, 0.]], dtype=dtype, device=None).T
    beta_soln = xnp.array([[4]], dtype=dtype, device=None).T
    alpha_soln = xnp.array([], dtype=dtype, device=None).T
    rhs = xnp.array([[1.0, 0.0, 0.]], dtype=dtype, device=None).T
    idx_soln = 1
    return A, rhs, beta_soln, alpha_soln, idx_soln


def case_3(xnp, dtype):
    beta = [1., 3., 7.]
    alpha = [0.1, 1.0]
    A = [[beta[2], 0, alpha[1]], [0, beta[0], alpha[0]], [alpha[1], alpha[0], beta[1]]]
    beta_soln = xnp.array([beta], dtype=dtype, device=None).T
    alpha_soln = xnp.array([alpha], dtype=dtype, device=None).T
    A = xnp.array(A, dtype=dtype, device=None)
    rhs = xnp.array([[0.0, 1.0, 0.]], dtype=dtype, device=None).T
    idx_soln = A.shape[0]
    return A, rhs, beta_soln, alpha_soln, idx_soln


def case_2(xnp, dtype):
    beta = [1., 2., 4.]
    alpha = [0.1, 0.1]
    A = [[beta[0], alpha[0], 0.], [alpha[0], beta[1], alpha[0]], [0., alpha[0], beta[2]]]
    beta_soln = xnp.array([beta], dtype=dtype, device=None).T
    alpha_soln = xnp.array([alpha], dtype=dtype, device=None).T
    A = xnp.array(A, dtype=dtype, device=None)
    rhs = xnp.array([[1.0, 0., 0.]], dtype=dtype, device=None).T
    idx_soln = A.shape[0]
    return A, rhs, beta_soln, alpha_soln, idx_soln


def case_numpy(A, rhs, xnp, np_dtype=np.float64):
    A_np = np.array(A, dtype=np_dtype)
    rhs_np = np.array(rhs, np_dtype)
    results = []
    for bs in range(rhs_np.shape[-1]):
        init_val = init_lanczos_np(rhs_np[:, bs], max_iter=A_np.shape[0], dtype=np_dtype)
        V, diag, subdiag, idx = run_lanczos(A_np, init_val, max_iter=A_np.shape[0])
        T = construct_tri(subdiag[1:idx], diag[:idx])
        results.append((V[:, 1:idx + 1], diag[:idx], subdiag[1:idx], T))

    if rhs_np.shape[-1] > 1:
        out = stack_batches(results)
    else:
        out = results[0]
    out = [xnp.array(vec, rhs.dtype, device=None) for vec in out]
    Q, beta, alpha, T = out
    return Q, beta, alpha, T


def stack_batches(results):
    batch_size = len(results)
    tuple_size = len(results[0])
    out = []
    for jdx in range(tuple_size):
        stack = []
        for bs in range(batch_size):
            stack.append(results[bs][jdx])
        out.append(np.stack(stack, axis=0))
    return out


def run_irl(A, rhs, eig_n, max_size, max_iter, tol):
    init_val = init_lanczos_np(rhs, max_size, A.dtype)
    norm, nq, counter = 2 * tol, max_size - eig_n, 0
    while (counter < max_iter) & (norm > tol):
        V, diag, subdiag, idx = run_lanczos(A, init_val, max_iter=max_size, tol=tol)
        T = construct_tri(subdiag[1:idx], diag[:idx])
        eigvals, _ = np.linalg.eig(T)
        eigvals = np.sort(eigvals)
        vec = np.copy(subdiag[-1] * V[:, [-1]])
        T, Q = run_shift_np(T.copy(), eigvals[:nq])

        beta = T[eig_n, eig_n - 1]
        sigma = Q[-1, eig_n - 1]
        new_vec = beta * V[:, [eig_n]] + sigma * vec
        V0 = V[:, 1:-1] @ Q[:, :eig_n]
        T0 = T[:eig_n, :eig_n]
        norm = np.linalg.norm(A @ V0 - V0 @ T0)

        init_val = init_lanczos_from_vec_np(V0, T0, new_vec, max_iter=max_size, idx=eig_n)
        counter += 1
    return V0, T0


def init_lanczos_from_vec_np(Q0, T0, vec, max_iter, idx):
    dtype = Q0.dtype
    subdiag = np.zeros(shape=(max_iter + 1, ), dtype=dtype)
    diag = np.zeros(shape=(max_iter, ), dtype=dtype)
    Q1 = np.zeros(shape=(vec.shape[0], max_iter + 2), dtype=dtype)

    diag[:idx] = np.diag(T0.copy())
    subdiag[1:idx] = np.diag(T0, k=-1).copy()
    norm = np.linalg.norm(vec)
    subdiag[idx] = norm
    Q1[:, 1:idx + 1] = Q0[:, :idx].copy()
    Q1[:, [idx + 1]] = vec / norm
    return Q1, diag, subdiag, idx + 1, norm


def run_shift_np(A, shifts):
    Q, Id = np.eye(A.shape[0], dtype=A.dtype), np.eye(A.shape[0], dtype=A.dtype)
    for jdx in range(len(shifts)):
        Q1, _ = np.linalg.qr(A - shifts[jdx] * Id, mode="complete")
        A = Q1.T @ A @ Q1
        Q = Q @ Q1
    return A, Q


def run_lanczos(A, init_val, max_iter, tol=1e-8):
    V, diag, subdiag, idx, norm = init_val
    while (idx <= max_iter):
        V[:, idx + 1] = A @ V[:, idx]
        diag[idx - 1] = V[:, idx + 1].conj().T @ V[:, idx]
        V[:, idx + 1] -= diag[idx - 1] * V[:, idx]
        V[:, idx + 1] -= subdiag[idx - 1] * V[:, idx - 1]
        V = do_double_gram(V, ind=idx + 1)

        norm = np.linalg.norm(V[:, idx + 1])
        subdiag[idx] = norm
        idx += 1
        if norm < tol * np.sqrt(A.shape[0]):
            break
        else:
            V[:, idx] = V[:, idx] / norm

    return V, diag, subdiag, idx - 1


def init_lanczos_np(rhs, max_iter, dtype):
    subdiag = np.zeros(shape=(max_iter + 1, ), dtype=dtype)
    diag = np.zeros(shape=(max_iter, ), dtype=dtype)
    V = np.zeros(shape=rhs.shape + (max_iter + 2, ), dtype=dtype)
    idx = 1
    V[:, idx] = rhs.copy()
    norm = np.linalg.norm(V[:, idx])
    V[:, idx] = V[:, idx] / norm
    return V, diag, subdiag, idx, norm


def do_double_gram(vec, ind):
    vec[:, ind] = vec[:, ind] - vec[:, :ind] @ (vec[:, :ind].conj().T @ vec[:, ind])
    vec[:, ind] = vec[:, ind] - vec[:, :ind] @ (vec[:, :ind].conj().T @ vec[:, ind])
    return vec


def construct_tri(band, diag):
    dim = diag.shape[0]
    T = np.zeros((dim, dim), dtype=diag.dtype)
    for idx in range(dim):
        T[idx, idx] = diag[idx]
        if idx == 0:
            T[idx, idx + 1] = band[idx]
        elif idx == dim - 1:
            T[idx, idx - 1] = band[idx - 1]
        else:
            T[idx, idx + 1] = band[idx]
            T[idx, idx - 1] = band[idx - 1]
    return T
