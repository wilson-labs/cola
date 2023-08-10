from operator_market import get_test_operators
from cola.linalg import logdet
from cola.ops import LinearOperator
import cola
from cola import jax_fns
from cola import torch_fns
from cola.utils_test import parametrize, relative_error

jax_test_ops = get_test_operators(jax_fns, jax_fns.float64)
torch_test_ops = get_test_operators(torch_fns, torch_fns.float64)


@parametrize(torch_test_ops + jax_test_ops)
def test_logdet(operator):
    A, dtype, xnp = operator, operator.dtype, operator.xnp
    A2 = LinearOperator(A.dtype, A.shape, A._matmat)
    tol = 1e-4
    l0 = xnp.slogdet(A.to_dense())[1]
    l1 = logdet(A, tol=tol, method='dense')
    e1 = relative_error(l0, l1)
    assert e1 < tol, f"Dispatch rules failed on {type(A)} with error {e1}"
    A3 = cola.SelfAdjoint(A2) if  A.isa(cola.SelfAdjoint) else A2
    l2 = logdet(A3, tol=tol, method='dense')
    e2 = relative_error(l0, l2) 
    assert e2 < tol, f"Dense logdet failed on {type(A)} with error {e2}"
    if A.isa(cola.SelfAdjoint):
        l3 = logdet(A3, tol=tol, method='iterative', num_samples=100)
        e3 = relative_error(l0, l3)
        assert e3 < 1e-1, f"SLQ logdet failed on {type(A)} with error {e3}"