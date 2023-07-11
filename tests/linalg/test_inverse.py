from operator_market import get_test_operators
from cola.linalg import inverse
from cola.ops import LinearOperator
from cola import jax_fns
from cola import torch_fns
from cola.utils_test import parametrize, relative_error

jax_test_ops = get_test_operators(jax_fns, jax_fns.float64)
torch_test_ops = get_test_operators(torch_fns, torch_fns.float64)

@parametrize(jax_test_ops+torch_test_ops)
def test_inverse(operator):
    A = operator
    A2 = LinearOperator(A.dtype, A.shape, A._matmat)
    Ainv = inverse(A,tol=1e-4)
    Ainv2 = inverse(A2,tol=1e-4,method='dense')
    Ainv3 = inverse(A2,tol=1e-4, method='krylov')
    B = A.ops.fixed_normal_samples((A.shape[-1], 10))
    X = Ainv @ B
    assert relative_error(A@X,B) < 1e-4, f"Dispatch rules failed on {type(A)}"
    assert relative_error(X, Ainv2 @ B) < 1e-4, f"Dense inversion failed on {type(A)}"
    assert relative_error(X, Ainv3 @ B) < 1e-4, f"Krylov inversion failed on {type(A)}"

