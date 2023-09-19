from cola.linalg import inv
from cola.ops import LinearOperator
from cola.utils.test_utils import parametrize, relative_error
import cola
from operator_market import op_names, get_test_operator
from cola.backends import all_backends


@parametrize(all_backends, ['float64'], op_names).excluding['torch', :, 'psd_kron']
def test_inverse(backend, precision, op_name):
    operator = get_test_operator(backend, precision, op_name)
    tol = 1e-5
    A, dtype, xnp = operator, operator.dtype, operator.xnp
    A2 = LinearOperator(A.dtype, A.shape, A._matmat)
    A2.xnp = operator.xnp
    Ainv = inv(A, tol=tol)
    Ainv.xnp = operator.xnp
    A3 = cola.PSD(A2) if A.isa(cola.PSD) else A2
    A3.xnp = operator.xnp
    Ainv2 = inv(A3, tol=tol, method='dense')
    Ainv3 = inv(A3, tol=tol, method='iterative')
    B = xnp.randn(*(A.shape[-1], 10), dtype=dtype, device=None)
    B = xnp.array(B, dtype=dtype, device=None)
    X = Ainv @ B
    rel_error = relative_error(A @ X, B)
    assert rel_error < 3 * tol, f"Dispatch rules failed on {type(A)} with {rel_error}"
    rel_error = relative_error(X, Ainv2 @ B)
    assert rel_error < 3 * tol, f"Dense inversion failed on {type(A)} with {rel_error}"
    rel_error = relative_error(X, Ainv3 @ B)
    assert rel_error < 10 * tol, f"Krylov inversion failed on {type(A)} with {rel_error}"
