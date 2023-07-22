from operator_market import get_test_operators
from cola.linalg import inverse
from cola.ops import LinearOperator
from cola import jax_fns
from cola import torch_fns
from cola.utils_test import parametrize, relative_error

jax_test_ops = get_test_operators(jax_fns, jax_fns.float64)
torch_test_ops = get_test_operators(torch_fns, torch_fns.float64)


@parametrize(torch_test_ops + jax_test_ops)
def test_inverse(operator):
    A, dtype, xnp = operator, operator.dtype, operator.ops
    A2 = LinearOperator(A.dtype, A.shape, A._matmat)
    Ainv = inverse(A, tol=1e-4)
    Ainv2 = inverse(A2, tol=1e-4, method='dense')
    Ainv3 = inverse(A2, tol=1e-4, method='krylov')
    B = xnp.array(xnp.fixed_normal_samples((A.shape[-1], 10)), dtype=dtype)
    X = Ainv @ B
    rel_error = relative_error(A @ X, B)
    assert rel_error < 1e-4, f"Dispatch rules failed on {type(A)}"
    rel_error = relative_error(X, Ainv2 @ B)
    assert rel_error < 1e-4, f"Dense inversion failed on {type(A)}"
    rel_error = relative_error(X, Ainv3 @ B)
    assert rel_error, f"Krylov inversion failed on {type(A)}"
