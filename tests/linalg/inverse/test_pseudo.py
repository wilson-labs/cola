from cola.linalg.inverse.pseudo import pseudo
from cola.linalg.algorithm_base import Auto
from cola.ops import LinearOperator
from cola.utils.test_utils import parametrize, relative_error
from cola.backends import all_backends
from nonsquare_operator_market import get_test_operator


@parametrize(all_backends, ["float64"], ["psd_diagonal", "square_dense"])
def test_pseudo(backend, precision, op_name):
    operator = get_test_operator(backend, precision, op_name)
    tol = 1e-5
    A, dtype, xnp = operator, operator.dtype, operator.xnp
    A2 = LinearOperator(A.dtype, A.shape, A._matmat)
    Ainv = pseudo(A, Auto(tol=tol))
    Ainv2 = pseudo(A2, Auto(tol=tol))
    Ainv3 = pseudo(A2, Auto(tol=tol, method="iterative"))
    B = xnp.randn(*(A.shape[-1], 10), dtype=dtype, device=None)
    X = Ainv @ B
    rel_error = relative_error(A @ X, B)
    assert rel_error < 3 * tol, f"Dispatch rules failed on {type(A)} with {rel_error}"
    rel_error = relative_error(X, Ainv2 @ B)
    assert rel_error < 3 * tol, f"Dense inversion failed on {type(A)} with {rel_error}"
    rel_error = relative_error(X, Ainv3 @ B)
    assert rel_error < 10 * tol, f"Krylov inversion failed on {type(A)} with {rel_error}"
