import numpy as np
from operator_market import get_test_operators
import cola
from cola.linalg import logdet
from cola.ops import LinearOperator
from cola import jax_fns
from cola import torch_fns
from cola.utils_test import parametrize, relative_error

jax_test_ops = get_test_operators(jax_fns, jax_fns.float64)
torch_test_ops = get_test_operators(torch_fns, torch_fns.float64)


@parametrize(torch_test_ops + jax_test_ops)
def test_logdet(operator):
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
    not_scalarmul = relative_error(xnp.diag(diag.mean() + 0. * diag), Adense) > 1e-5
    A3 = cola.PSD(A2) if A.isa(cola.PSD) else A2
    if not_scalarmul:
        l3 = logdet(A3, tol=tol, method='iterative-stochastic', vtol=3e-2)
        e3 = relative_error(l0, l3)
        assert e3 < 3e-1, f"SLQ logdet failed on {type(A)} with error {e3}"
        l4 = logdet(A3, tol=tol, method='iterative', vtol=tol)
        e4 = relative_error(l0, l4)
        assert e4 < 3 * tol, f"SLQ logdet failed on {type(A)} with error {e4}"
