from cola.ops.operators import BlockDiag, Diagonal, Kronecker
from cola.linalg.algorithm_base import Auto
from cola.linalg.inverse.pinv import pinv
from cola.utils.utils_for_tests import get_xnp, parametrize, relative_error
from cola.backends import all_backends


@parametrize(all_backends)
def test_pinv_block_diag(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float64
    A = Diagonal(xnp.array([1., 2., 0.], dtype=dtype, device=None))
    B = Diagonal(xnp.array([3., 4., 5.], dtype=dtype, device=None))
    BD = BlockDiag(A, B)

    # Dense pinv reference
    BD_dense = BD.to_dense()
    if backend == 'numpy':
        import numpy as np
        BD_pinv_dense = np.linalg.pinv(BD_dense)
    else:
        BD_pinv_dense = xnp.pinv(BD_dense)

    # Dispatch pinv
    BD_pinv = pinv(BD, alg=Auto())

    rel_error = relative_error(BD_pinv.to_dense(), BD_pinv_dense)
    assert rel_error < 1e-5


@parametrize(all_backends)
def test_pinv_kronecker(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float64
    A = Diagonal(xnp.array([1., 2.], dtype=dtype, device=None))
    B = Diagonal(xnp.array([3., 4., 0.], dtype=dtype, device=None))
    K = Kronecker(A, B)

    # Dense pinv reference
    K_dense = K.to_dense()
    if backend == 'numpy':
        import numpy as np
        K_pinv_dense = np.linalg.pinv(K_dense)
    else:
        K_pinv_dense = xnp.pinv(K_dense)

    # Dispatch pinv
    K_pinv = pinv(K, alg=Auto())

    rel_error = relative_error(K_pinv.to_dense(), K_pinv_dense)
    assert rel_error < 1e-5
