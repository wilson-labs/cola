from scipy.io import mmread
import pytest
from cola.linalg.inverse.inv import inv
from cola.linalg.algorithm_base import Auto
from cola.ops import LinearOperator
from cola.ops import Sparse
from cola.utils.test_utils import get_xnp, parametrize, relative_error
from cola.utils.test_utils import transform_to_csr
from cola.linalg.inverse.cg import CG
from cola.linalg.inverse.gmres import GMRES
from cola.annotations import PSD
from cola.backends import all_backends, tracing_backends
from operator_market import op_names, get_test_operator


@pytest.mark.market
@parametrize(tracing_backends)
def test_gmres_matrix_market(backend):
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
        rhs = xnp.ones(shape=(A.shape[0], 5), dtype=dtype, device=None)

        A_inv = inv(A, GMRES(max_iters=1_000, tol=1e-8))
        rel_error = relative_error(A @ A_inv @ rhs, rhs)
        print(f"Rel error: {rel_error:2.5e}")
        assert rel_error < 1e-6


@pytest.mark.market
@parametrize(tracing_backends)
def test_matrix_market(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float64
    input_path_s = [
        "./tests/data/1138_bus.mtx",
        "./tests/data/Tre20k.mtx",
        "./tests/data/finan512.mtx",
        "./tests/data/cfd1.mtx",
    ]
    for input_path in input_path_s:
        print(input_path)
        matrix = mmread(input_path)
        data, col_ind, rowptr, shape = transform_to_csr(matrix.tocsc(), xnp=xnp, dtype=dtype)
        A = Sparse(data, col_ind, rowptr, shape)
        rhs = xnp.ones(shape=(A.shape[0], 5), dtype=dtype, device=None)

        A_inv = inv(PSD(A), CG(max_iters=5_000, tol=1e-8))
        rel_error = relative_error(A @ A_inv @ rhs, rhs)
        print(f"Rel error: {rel_error:2.5e}")
        assert rel_error < 1e-7


@parametrize(all_backends, ['float64'], op_names).excluding['torch', :, 'psd_kron']
def test_inverse(backend, precision, op_name):
    operator = get_test_operator(backend, precision, op_name)
    tol = 1e-5
    A, dtype, xnp = operator, operator.dtype, operator.xnp
    A2 = LinearOperator(A.dtype, A.shape, A._matmat)
    Ainv = inv(A, Auto(tol=tol))
    A3 = PSD(A2) if A.isa(PSD) else A2
    Ainv2 = inv(A3, Auto(tol=tol, method='dense'))
    Ainv3 = inv(A3, Auto(tol=tol, method='iterative'))
    B = xnp.randn(*(A.shape[-1], 10), dtype=dtype, device=None)
    B = xnp.array(B, dtype=dtype, device=None)
    X = Ainv @ B
    rel_error = relative_error(A @ X, B)
    assert rel_error < 3 * tol, f"Dispatch rules failed on {type(A)} with {rel_error}"
    rel_error = relative_error(X, Ainv2 @ B)
    assert rel_error < 3 * tol, f"Dense inversion failed on {type(A)} with {rel_error}"
    rel_error = relative_error(X, Ainv3 @ B)
    assert rel_error < 10 * tol, f"Krylov inversion failed on {type(A)} with {rel_error}"
