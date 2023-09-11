from cola import kron
from cola.ops import Dense, LinearOperator, Identity, Diagonal
from cola.algorithms import exact_diag, approx_diag
from cola.linalg import diag, trace
from cola.utils_test import get_xnp, parametrize, relative_error


@parametrize(['torch', 'jax'])
def test_exact_diag(backend):
    xnp = get_xnp(backend)
    A = Dense(xnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9.]], dtype=xnp.float32, device=None))
    for u in [-2, -1, 0, 1, 2]:
        d1, _ = exact_diag(A, u)
        d2 = xnp.diag(A.to_dense(), u)
        assert relative_error(d1, d2) < 1e-5


@parametrize(['torch', 'jax'])
def test_approx_diag(backend):
    xnp = get_xnp(backend)
    A = Dense(xnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9.]], dtype=xnp.float32, device=None))
    for u in [-2, -1, 0, 1, 2]:
        d1, _ = approx_diag(A, u, tol=5e-2)
        d2 = xnp.diag(A.to_dense(), u)
        assert relative_error(d1, d2) < 9e-1


@parametrize(['torch', 'jax'])
def test_composite_diag(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    A = Dense(xnp.array([[-1, 2], [3, 2]], dtype=dtype, device=None))
    A = LinearOperator(A.dtype, A.shape, A._matmat)
    B = Dense(xnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9.]], dtype=dtype, device=None))
    C = Dense(-xnp.ones((3, 3), dtype=dtype, device=None))
    M = kron(A, B + C)
    d1 = diag(M)
    d2 = xnp.diag(M.to_dense())
    assert relative_error(d1, d2) < 1e-5


@parametrize(['torch', 'jax'], ['exact', 'approx']).excluding[:,'approx']
def test_large_trace(backend, method):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    array = xnp.fixed_normal_samples((210, 210), dtype=dtype, device=None)
    A = Dense(array)
    A = LinearOperator(A.dtype, A.shape, A._matmat)
    d1 = trace(A, method=method, tol=2e-2)
    d2 = xnp.diag(array).sum()
    assert relative_error(d1, d2) < (1e-1 if method == 'approx' else 1e-5)


@parametrize(['torch', 'jax'])
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
            assert d1.shape ==  d2.shape
            assert relative_error(d1, d2) < 1e-5