from cola.utils.test_utils import get_xnp, parametrize, relative_error
from cola.backends import all_backends
from linalg.operator_market import op_names, get_test_operator
from cola.linalg.inverse.inv import inv
from cola.linalg.logdet.logdet import logdet
from cola.linalg.decompositions.decompositions import cholesky, plu
from cola.linalg.algorithm_base import Auto


@parametrize(all_backends, [op for op in op_names if op.startswith('psd')])
def test_cholesky(backend, opname):
    xnp = get_xnp(backend)
    A = get_test_operator(backend, 'float32', opname)
    A_decomposed = cholesky(A)
    Ainv1 = xnp.inv(A_decomposed.to_dense())
    Ainv2 = inv(A_decomposed).to_dense()
    assert relative_error(Ainv1, Ainv2) < 1e-5, "inverse failed"
    logdet1 = xnp.slogdet(A_decomposed.to_dense())[1]
    logdet2 = logdet(A_decomposed, Auto(), Auto())
    assert relative_error(logdet1, logdet2) < 1e-5, "logdet failed"


@parametrize(all_backends, [op for op in op_names if op.startswith('square')])
def test_lu(backend, opname):
    xnp = get_xnp(backend)
    A = get_test_operator(backend, 'float32', opname)
    P, L, U = plu(A)
    A_decomposed = P @ L @ U
    Ainv1 = xnp.inv(A_decomposed.to_dense())
    Ainv2 = inv(A_decomposed).to_dense()
    assert relative_error(xnp.cast(Ainv1, Ainv2.dtype), Ainv2) < 1e-5, "inverse failed"
    logdet1 = xnp.slogdet(A_decomposed.to_dense())[1]
    logdet2 = logdet(A_decomposed, Auto(), Auto())
    if logdet1 > 1e-4 or logdet2 > 1e-4:
        assert relative_error(logdet1, logdet2) < 1e-5, "logdet failed"
