from operator_market import get_test_operators
from cola.linalg import sqrt, exp
from cola.ops import LinearOperator
import cola
from cola import jax_fns
from cola import torch_fns
from cola.utils_test import parametrize, relative_error
from scipy.linalg import sqrtm, expm
import numpy as np

torch_test_ops = get_test_operators(torch_fns, torch_fns.float32)
jax_test_ops = get_test_operators(jax_fns, jax_fns.float32)
ops = jax_test_ops + torch_test_ops


@parametrize(ops, [(exp, expm), (sqrt, sqrtm)])
def test_unary(operator, fns):
    fn, spfn = fns
    A, dtype, xnp = operator, operator.dtype, operator.xnp
    A2 = LinearOperator(A.dtype, A.shape, A._matmat)
    tol = 1e-4
    v = xnp.randn(A.shape[-1], dtype=dtype)
    Adense = A.to_dense()
    Anp = np.array(Adense)
    fv = spfn(Anp) @ np.array(v)
    fv1 = np.array(fn(A, tol=tol, method='auto') @ v)
    e1 = relative_error(fv, fv1)
    assert e1 < 3 * tol, f"Dispatch rules failed on {type(A)} with error {e1}"
    A3 = cola.SelfAdjoint(A2) if A.isa(cola.SelfAdjoint) else A2
    if np.prod(A.shape) < 1000:
        fv2 = np.array(fn(A3, tol=tol, method='dense') @ v)
        e2 = relative_error(fv, fv2)
        assert e2 < 3 * tol, f"Dense f(A) failed on {type(A)} with error {e2}"
    diag = xnp.diag(Adense)
    not_scalarmul = relative_error(xnp.diag(diag.mean() + 0. * diag), Adense) > 1e-5
    if not_scalarmul:
        fv3 = np.array(fn(A3, tol=tol, method='iterative') @ v)
        e3 = relative_error(fv, fv3)
        assert e3 < 3 * tol, f"SLQ logdet failed on {type(A)} with error {e3}"
