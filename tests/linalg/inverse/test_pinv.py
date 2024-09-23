from nonsquare_operator_market import get_test_operator

from cola.backends import all_backends
from cola.linalg.algorithm_base import Auto
from cola.linalg.inverse.cg import CG
from cola.linalg.inverse.pinv import LSTSQ, pinv
from cola.ops import LinearOperator
from cola.utils.utils_for_tests import parametrize, relative_error


@parametrize(all_backends, ["float64"], ["psd_diagonal", "nonsquare_dense"])
def test_pinv(backend, precision, op_name):
    operator = get_test_operator(backend, precision, op_name)
    tol = 1e-5
    A, dtype, xnp = operator, operator.dtype, operator.xnp
    A2 = LinearOperator(A.dtype, A.shape, A._matmat)
    Ainv = pinv(A, Auto(tol=tol))
    Ainv2 = pinv(A2, LSTSQ())
    Ainv3 = pinv(A2, CG(tol=tol))
    B = xnp.randn(*(A.shape[0], 10), dtype=dtype, device=None)
    X = Ainv @ B
    rel_error = relative_error(A @ X, B)
    assert rel_error < 3 * tol, f"Dispatch rules failed on {type(A)} with {rel_error}"
    rel_error = relative_error(X, Ainv2 @ B)
    assert rel_error < 3 * tol, f"Dense inversion failed on {type(A)} with {rel_error}"
    rel_error = relative_error(X, Ainv3 @ B)
    assert rel_error < 10 * tol, f"Krylov inversion failed on {type(A)} with {rel_error}"
