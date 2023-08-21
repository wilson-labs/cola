from cola import kron
from cola.ops import Dense, LinearOperator
from cola.algorithms import exact_diag, approx_diag
from cola.linalg import diag, trace
from jax.config import config
from cola.utils_test import parametrize, relative_error
from cola import jax_fns
from cola import torch_fns

config.update('jax_platform_name', 'cpu')


@parametrize([torch_fns, jax_fns])
def test_exact_diag(xnp):
    A = Dense(xnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9.]], dtype=xnp.float32))
    for u in [-2, -1, 0, 1, 2]:
        d1, _ = exact_diag(A, u)
        d2 = xnp.diag(A.to_dense(), u)
        assert relative_error(d1, d2) < 1e-5


@parametrize([torch_fns, jax_fns])
def test_approx_diag(xnp):
    A = Dense(xnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9.]], dtype=xnp.float32))
    for u in [-2, -1, 0, 1, 2]:
        d1, _ = approx_diag(A, u, tol=5e-2)
        d2 = xnp.diag(A.to_dense(), u)
        assert relative_error(d1, d2) < 9e-1


@parametrize([torch_fns, jax_fns])
def test_composite_diag(xnp):
    A = Dense(xnp.array([[-1, 2], [3, 2]], dtype=xnp.float32))
    A = LinearOperator(A.dtype, A.shape, A._matmat)
    B = Dense(xnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9.]], dtype=xnp.float32))
    C = Dense(-xnp.ones((3, 3), dtype=xnp.float32))
    M = kron(A, B + C)
    d1 = diag(M)
    d2 = xnp.diag(M.to_dense())
    assert relative_error(d1, d2) < 1e-5


@parametrize([torch_fns, jax_fns], ['exact', 'approx'])
def test_large_trace(xnp, method):
    array = xnp.fixed_normal_samples((210, 210))
    A = Dense(array)
    A = LinearOperator(A.dtype, A.shape, A._matmat)
    d1 = trace(A, method=method, tol=2e-2)
    d2 = xnp.diag(array).sum()
    assert relative_error(d1, d2) < (1e-1 if method == 'approx' else 1e-5)
