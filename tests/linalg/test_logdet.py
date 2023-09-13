import numpy as np
from operator_market import op_names, get_test_operator
import cola
from cola.linalg import logdet
from cola.ops import LinearOperator
from cola.utils.test_utils import parametrize, relative_error

_exclude = (slice(None), slice(None), ['psd_identity', 'psd_scalarmul'])


@parametrize(['torch', 'jax'], ['float64'], op_names).excluding[_exclude]
def test_logdet(backend, precision, op_name):
    operator = get_test_operator(backend, precision, op_name)
    A, _, xnp = operator, operator.dtype, operator.xnp
    A2 = LinearOperator(A.dtype, A.shape, A._matmat)
    tol = 1e-4
    Adense = A.to_dense()
    l0 = xnp.slogdet(Adense)[1]
    l1 = logdet(A, tol=tol)
    e1 = relative_error(l0, l1)
    assert e1 < 3 * tol, f"Dispatch rules failed on {type(A)} with error {e1}"
    if np.prod(A.shape) < 1000:
        A3 = cola.PSD(A2) if A.isa(cola.PSD) else A2
        l2 = logdet(A3, tol=tol, method='dense')
        e2 = relative_error(l0, l2)
        assert e2 < 3 * tol, f"Dense logdet failed on {type(A)} with error {e2}"
    diag = xnp.diag(Adense)
    assert relative_error(xnp.diag(diag.mean() + 0. * diag), Adense) > 1e-5
    A3 = cola.PSD(A2) if A.isa(cola.PSD) else A2
    l3 = logdet(A3, tol=tol, method='iterative-stochastic', vtol=3e-2)
    e3 = relative_error(l0, l3)
    assert e3 < 3e-1, f"SLQ logdet failed on {type(A)} with error {e3}"
    l4 = logdet(A3, tol=tol, method='iterative-exact', vtol=tol)
    e4 = relative_error(l0, l4)
    assert e4 < 10 * tol, f"Tr(log(A)) logdet failed on {type(A)} with error {e4}"
