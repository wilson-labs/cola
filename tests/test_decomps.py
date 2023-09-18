import cola
from cola.utils.test_utils import get_xnp, parametrize, relative_error
from cola.backends import all_backends
from linalg.operator_market import op_names, get_test_operator


@parametrize(all_backends, [op for op in op_names if op.startswith('psd')])
def test_cholesky(backend, opname):
    xnp = get_xnp(backend)
    A = get_test_operator(backend, 'float32', opname)
    A_decomposed = cola.cholesky_decomposed(A)
    Ainv1 = xnp.inv(A_decomposed.to_dense())
    Ainv2 = cola.inv(A_decomposed).to_dense()
    assert relative_error(Ainv1, Ainv2) < 1e-5, "inverse failed"
    logdet1 = xnp.slogdet(A_decomposed.to_dense())[1]
    logdet2 = cola.logdet(A_decomposed)
    assert relative_error(logdet1, logdet2) < 1e-5, "logdet failed"


@parametrize(all_backends, [op for op in op_names if op.startswith('square')])
def test_lu(backend, opname):
    xnp = get_xnp(backend)
    A = get_test_operator(backend, 'float32', opname)
    A_decomposed = cola.lu_decomposed(A)
    Ainv1 = xnp.inv(A_decomposed.to_dense())
    Ainv2 = cola.inv(A_decomposed).to_dense()
    assert relative_error(xnp.cast(Ainv1, Ainv2.dtype), Ainv2) < 1e-5, "inverse failed"
    logdet1 = xnp.slogdet(A_decomposed.to_dense())[1]
    logdet2 = cola.logdet(A_decomposed)
    if logdet1 > 1e-4 or logdet2 > 1e-4:
        assert relative_error(logdet1, logdet2) < 1e-5, "logdet failed"
