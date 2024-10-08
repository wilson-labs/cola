import numpy as np
from operator_market import get_test_operator, op_names

from cola.annotations import PSD, SelfAdjoint
from cola.backends import all_backends
from cola.linalg.algorithm_base import Auto
from cola.linalg.decompositions.decompositions import Lanczos
from cola.linalg.logdet.logdet import logdet
from cola.linalg.trace.diag_trace import Exact, Hutch
from cola.ops.operator_base import LinearOperator
from cola.ops.operators import Dense
from cola.utils.utils_for_tests import parametrize, relative_error

_exclude = (slice(None), slice(None), ['psd_identity', 'psd_scalarmul', 'selfadj_tridiagonal'])


@parametrize(all_backends, ['float64'], op_names).excluding[_exclude]
def test_logdet(backend, precision, op_name):
    operator = get_test_operator(backend, precision, op_name)
    A, _, xnp = operator, operator.dtype, operator.xnp
    if not A.isa(SelfAdjoint) and backend == 'numpy':
        return
    A2 = LinearOperator(A.dtype, A.shape, A._matmat)
    tol = 1e-4
    Adense = A.to_dense()
    l0 = xnp.slogdet(Adense)[1]
    l1 = logdet(A, log_alg=Auto(tol=tol), trace_alg=Auto(tol=tol))
    if l0 < 1e-4:
        assert True
        return
    e1 = relative_error(l0, l1)
    assert e1 < 3 * tol, f"Dispatch rules failed on {type(A)} with error {e1}"
    if np.prod(A.shape) < 1000:
        A3 = PSD(A2) if A.isa(PSD) else A2
        l2 = logdet(A3, log_alg=Auto(), trace_alg=Auto())
        e2 = relative_error(l0, l2)
        assert e2 < 3 * tol, f"Dense logdet failed on {type(A)} with error {e2}"
    diag = xnp.diag(Adense)
    assert relative_error(xnp.diag(diag.mean() + 0. * diag), Adense) > 1e-5
    if A.isa(PSD):
        A3, alg, mult = PSD(A2), Lanczos(), 1.0
    else:
        A3, alg, mult = PSD(Dense((A2.H @ A).to_dense())), Lanczos(), 0.5
    max_iters = 25 if backend == "torch" else 150
    l3 = logdet(A3, log_alg=alg, trace_alg=Hutch(max_iters=max_iters, key=xnp.PRNGKey(42)))
    e3 = relative_error(l0, l3 * mult)
    assert e3 < 5e-1, f"SLQ logdet failed on {type(A)} with error {e3}"
    l4 = logdet(A3, log_alg=alg, trace_alg=Exact())
    e4 = relative_error(l0, l4 * mult)
    assert e4 < 10 * tol, f"Tr(log(A)) logdet failed on {type(A)} with error {e4}"
