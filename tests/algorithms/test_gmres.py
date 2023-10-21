import numpy as np
from scipy.io import mmread
import pytest
from cola.fns import lazify
from cola.ops import Identity
from cola.ops import Diagonal
from cola.ops import Sparse
from cola.linalg.inverse.inv import inv
from cola.linalg.inverse.gmres import GMRES
from cola.linalg.inverse.gmres import gmres
from cola.linalg.inverse.gmres import gmres_fwd
from cola.utils.test_utils import get_xnp, parametrize, relative_error
from cola.backends import all_backends, tracing_backends
from cola.utils.test_utils import generate_spectrum, generate_pd_from_diag
from cola.utils.test_utils import transform_to_csr


@pytest.mark.market
@parametrize(tracing_backends)
def test_matrix_market(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float64
    input_path_s = ["./tests/data/1138_bus.mtx"]
    for input_path in input_path_s:
        matrix = mmread(input_path)
        data, col_ind, rowptr, shape = transform_to_csr(matrix.tocsc(), xnp=xnp, dtype=dtype)
        A = Sparse(data, col_ind, rowptr, shape)
        rhs = xnp.ones(shape=(A.shape[0], 5), dtype=dtype, device=None)

        max_iters, tol = 1_000, 1e-8
        approx, _ = gmres(A, rhs, max_iters=max_iters, tol=tol)

        rel_error = relative_error(A @ approx, rhs)
        assert rel_error < 1e-3


@parametrize(tracing_backends)
def test_gmres_vjp(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    diag = xnp.Parameter(xnp.array([3., 4., 5.], dtype=dtype, device=None))
    diag_soln = xnp.Parameter(xnp.array([3., 4., 5.], dtype=dtype, device=None))
    A = xnp.diag(diag)
    rhs = xnp.ones(shape=(3, 1), dtype=dtype, device=None)
    max_iters, tol = 5, 1e-6
    use_householder, use_triangular, pbar = False, False, False
    x0, P = xnp.zeros_like(rhs), Identity(dtype=A.dtype, shape=A.shape)
    _, unflatten = Diagonal(diag).flatten()

    def f(theta):
        A = unflatten([theta])
        solve, _ = gmres_fwd(A, rhs, x0, max_iters, tol, P, use_householder, use_triangular, pbar)
        loss = xnp.sum(solve)
        return loss

    def f_alt(theta):
        X = xnp.diag(theta)
        solve = xnp.solve(X, rhs)
        loss = xnp.sum(solve)
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

    rel_error = relative_error(soln, approx)
    assert rel_error < 1e-6


@parametrize(all_backends)
def test_gmres_random(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=25, dtype=np.float32)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype), dtype=dtype, device=None)
    rhs = xnp.ones(shape=(A.shape[0], 3), dtype=dtype, device=None)
    soln = xnp.solve(A, rhs)

    max_iters, tol = A.shape[0] - 5, 1e-8
    A_inv = inv(lazify(A), GMRES(max_iters=max_iters, tol=tol))
    approx = A_inv @ rhs

    rel_error = relative_error(soln, approx)
    assert rel_error < 5e-4


@parametrize(all_backends)
def test_gmres_easy(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    A = xnp.diag(xnp.array([3., 4., 5.], dtype=dtype, device=None))
    rhs = [[1], [1], [1]]
    rhs = xnp.array(rhs, dtype=dtype, device=None)
    soln = [[1 / 3], [1 / 4], [1 / 5]]
    soln = xnp.array(soln, dtype=dtype, device=None)

    max_iters, tolerance = 3, 1e-8
    fn = gmres
    x0 = xnp.zeros_like(rhs)

    approx, _ = fn(lazify(A), rhs, x0, max_iters, tolerance)
    rel_error = relative_error(soln, approx)
    assert rel_error < 1e-6

    approx, _ = fn(lazify(A), rhs, x0, max_iters, tolerance, use_householder=False, use_triangular=True)
    rel_error = relative_error(soln, approx)
    assert rel_error < 1e-6

    # approx, _ = fn(lazify(A), rhs, x0, max_iters, tolerance, use_householder=True,
    #                use_triangular=False)
    # rel_error = relative_error(soln, approx)
    # assert rel_error < 5e-7

    # approx, _ = fn(lazify(A), rhs, x0, max_iters, tolerance, use_householder=True,
    #                use_triangular=True)
    # rel_error = relative_error(soln, approx)
    # assert rel_error < 5e-7
