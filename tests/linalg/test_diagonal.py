from cola.fns import kron
from cola.ops import Dense, LinearOperator, Identity, Diagonal
from cola.linalg.trace.diagonal_estimation import exact_diag
from cola.linalg.trace.diagonal_estimation import hutchinson_diag_estimate as approx_diag
from cola.linalg.trace.diag_trace import trace
from cola.linalg.trace.diag_trace import diag
from cola.utils.utils_for_tests import get_xnp, parametrize, relative_error
from cola.linalg.trace.diagonal_estimation import Exact, Hutch
from cola.backends import all_backends


@parametrize(all_backends)
def test_exact_diag(backend):
    xnp = get_xnp(backend)
    A = Dense(xnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9.]], dtype=xnp.float32, device=None))
    for u in [-2, -1, 0, 1, 2]:
        d1 = exact_diag(A, u, bs=100)
        d2 = xnp.diag(A.to_dense(), u)
        assert relative_error(d1, d2) < 1e-5


@parametrize(all_backends)
def test_approx_diag(backend):
    xnp = get_xnp(backend)
    A = Dense(xnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9.]], dtype=xnp.float32, device=None))
    for u in [-2, -1, 0, 1, 2]:
        d1, _ = approx_diag(A, u, tol=5e-2)
        d2 = xnp.diag(A.to_dense(), u)
        assert relative_error(d1, d2) < 9e-1


@parametrize(all_backends)
def test_composite_diag(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    A = Dense(xnp.array([[-1, 2], [3, 2]], dtype=dtype, device=None))
    A = LinearOperator(A.dtype, A.shape, A._matmat)
    B = Dense(xnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9.]], dtype=dtype, device=None))
    C = Dense(-xnp.ones((3, 3), dtype=dtype, device=None))
    M = kron(A, B + C)
    d1 = diag(M, k=0, alg=Exact(bs=111))
    d2 = xnp.diag(M.to_dense())
    assert relative_error(d1, d2) < 1e-5


@parametrize(all_backends, ['exact', 'approx']).excluding[:, 'approx']
def test_large_trace(backend, method):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    key = xnp.PRNGKey(21)
    array = xnp.randn(*(210, 210), dtype=dtype, device=None, key=key)
    A = Dense(array)
    # A = LinearOperator(A.dtype, A.shape, A._matmat)
    if method == "exact":
        alg = Exact()
    elif method == "approx":
        alg = Hutch(tol=2e-2)
    d1 = trace(A, alg)
    d2 = xnp.diag(array).sum()
    assert relative_error(d1, d2) < (1e-1 if method == 'approx' else 1e-5)


@parametrize(all_backends)
def test_diagonal_diag(backend):
    size = 3
    xnp = get_xnp(backend)
    dtype = xnp.float32
    A = Diagonal(xnp.arange(size, dtype=dtype) + .1)
    B = Identity((size, size), dtype=dtype)
    for M in [A, B]:
        for u in range(-size + 1, size):
            d1 = diag(M, u)
            d2 = xnp.diag(M.to_dense(), u)
            assert d1.shape == d2.shape
            assert relative_error(d1, d2) < 1e-5
