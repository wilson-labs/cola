import numpy as np
import scipy
from jax.config import config
from scipy.io import mmread
import pytest
from cola.fns import lazify
from cola.ops import Dense
from cola.ops import Sparse
from cola.linalg.decompositions.lanczos import lanczos
from cola.linalg.decompositions.lanczos import lanczos_eigs
from cola.utils.test_utils import get_xnp, parametrize, relative_error
from cola.backends import all_backends, tracing_backends
from cola.utils.test_utils import generate_spectrum, generate_pd_from_diag
from cola.utils.test_utils import generate_diagonals, transform_to_csr

config.update('jax_enable_x64', True)

_tol = 1e-6


@pytest.mark.market
@parametrize(all_backends)
def test_matrix_market(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float64
    input_path_s = [
        "./tests/data/1138_bus.mtx",
    ]
    for input_path in input_path_s:
        print(input_path)
        matrix = mmread(input_path)
        data, col_ind, rowptr, shape = transform_to_csr(matrix.tocsc(), xnp=xnp, dtype=dtype)
        A = Sparse(data, col_ind, rowptr, shape)

        max_iters, tol = 5_000, 1e-8
        approx, *_ = lanczos_eigs(A, max_iters=max_iters, tol=tol)
        soln, *_ = scipy.linalg.eigh(matrix.toarray())
        soln = xnp.array(soln, dtype=dtype, device=None)

        rel_error = relative_error(approx[:100], soln[:100])
        print(f"Rel error: {rel_error:2.5e}")
        assert rel_error < 1e-8


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


@parametrize(all_backends)
def test_lanczos_complex(backend):
    xnp = get_xnp(backend)
    dtype = xnp.complex64
    np_dtype = np.complex64
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10, dtype=np_dtype)
    A = xnp.array(generate_diagonals(diag, seed=21), dtype=dtype, device=None)
    rhs = xnp.randn(A.shape[0], 1, dtype=dtype, device=None)
    alpha_np, beta_np, idx_np, Q_np, T_np = case_numpy(A, rhs, xnp, np_dtype)

    B = lazify(A)
    max_iters, tol = A.shape[0], 1e-7
    Q, T, info = lanczos(B, rhs, max_iters=max_iters, tol=tol, pbar=False)
    idx = info["iterations"] - 1
    alpha, beta = T.alpha[:, :, 0], T.beta[:, :, 0]
    Q, T = Q.to_dense(), xnp.vmap(T.__class__.to_dense)(T)

    assert idx == idx_np
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
def test_lanczos_random(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    np_dtype = np.float32
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10, dtype=np_dtype)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype, seed=21), dtype=dtype, device=None)
    rhs = xnp.ones(shape=(A.shape[0], 1), dtype=dtype, device=None)
    alpha_np, beta_np, idx_np, Q_np, T_np = case_numpy(A, rhs, xnp, np_dtype)

    B, max_iters, tol = lazify(A), A.shape[0], 1e-7
    Q, T, info = lanczos(B, rhs, max_iters=max_iters, tol=tol, pbar=False)
    idx = info["iterations"] - 1
    alpha, beta = T.alpha[:, :, 0], T.beta[:, :, 0]
    Q, T = Q.to_dense(), xnp.vmap(T.__class__.to_dense)(T)

    assert idx == idx_np
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
    max_eig = 6
    A = xnp.diag(xnp.array([4, 2, 1, max_eig], dtype=dtype, device=None))
    rhs = xnp.ones(shape=(A.shape[0], 7), dtype=dtype, device=None)
    alpha_np, beta_np, idx_np, Q_np, T_np = case_numpy(A, rhs, xnp)

    max_iters, tol = A.shape[0], 1e-7
    B = lazify(A)
    Q, T, info = lanczos(B, rhs, max_iters=max_iters, tol=tol, pbar=False)
    idx, Q = info["iterations"] - 1, Q.to_dense()
    alpha, beta = T.alpha[:, :, 0], T.beta[:, :, 0]
    T = xnp.vmap(T.__class__.to_dense)(T)
    eigvals, _ = xnp.eigh(T)

    assert idx == idx_np[0]
    comparisons = [(T_np, T), (Q_np, Q), (Q @ T, A @ Q), (alpha_np, alpha), (beta_np, beta),
                   (xnp.array(max_eig, dtype=dtype, device=None), eigvals[0, -1])]
    for soln, check in comparisons:
        rel_error = relative_error(soln, check)
        assert rel_error < _tol


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
        out = run_lanczos(A_np, rhs_np[:, bs], max_iter=A_np.shape[0], dtype=np_dtype)
        T = construct_tri(out[0], out[1])
        out += (T, )
        results.append(out)

    if rhs_np.shape[-1] > 1:
        out = stack_batches(results)
    out = [xnp.array(vec, rhs.dtype, device=None) for vec in out]
    return out


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


def run_lanczos(A, rhs, max_iter, dtype, tolerance=1e-8):
    max_iter = min(A.shape[0], max_iter)
    vec, alpha, beta = initialize_lanczos(rhs, max_iter=max_iter, dtype=dtype)
    for idx in range(1, max_iter + 1):
        vec[:, idx + 1] = A @ vec[:, idx]
        beta[idx - 1] = vec[:, idx + 1].conj().T @ vec[:, idx]
        vec[:, idx + 1] -= beta[idx - 1] * vec[:, idx]
        vec[:, idx + 1] -= alpha[idx - 1] * vec[:, idx - 1]
        vec = do_double_gram(vec, ind=idx + 1)

        alpha[idx] = np.linalg.norm(vec[:, idx + 1])
        if alpha[idx] < tolerance * np.sqrt(A.shape[0]):
            break
        else:
            vec[:, idx + 1] = vec[:, idx + 1] / alpha[idx]

    return alpha[1:idx], beta[:idx], idx, vec[:, 1:idx + 1]


def initialize_lanczos(rhs, max_iter, dtype):
    alpha = np.zeros(shape=(max_iter + 1, ), dtype=dtype)
    beta = np.zeros(shape=(max_iter, ), dtype=dtype)
    vec = np.zeros(shape=rhs.shape + (max_iter + 2, ), dtype=dtype)
    vec[:, 1] = rhs
    vec[:, 1] = vec[:, 1] / np.linalg.norm(vec[:, 1])
    return vec, alpha, beta


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
