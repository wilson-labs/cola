from cola.ops.operators import BlockDiag, Diagonal, Kronecker, Permutation, ScalarMul
from cola.linalg.svd.svd import svd
from cola.utils.utils_for_tests import get_xnp, parametrize, relative_error
from cola.backends import all_backends
import numpy as np
import cola.backends.np_fns as np_fns

if not hasattr(np_fns, 'pinv'):
    np_fns.pinv = np.linalg.pinv


@parametrize(all_backends)
def test_svd_scalar_mul(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float64
    c = -3.0
    A = Diagonal(xnp.array([1., 2., 3.], dtype=dtype, device=None))
    S_mul = ScalarMul(c, shape=A.shape, dtype=dtype, device=None)

    U, S, V = svd(S_mul, k=A.shape[0])

    approx = U @ S @ V.H
    rel_error = relative_error(approx.to_dense(), S_mul.to_dense())
    assert rel_error < 1e-5

    # Check singular values
    s_vals = S.diag
    expected_s = xnp.array([abs(c)] * 3, dtype=dtype, device=None)
    assert relative_error(s_vals, expected_s) < 1e-5


@parametrize(all_backends)
def test_svd_permutation(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float64
    perm = xnp.array([1, 0, 2], dtype=xnp.int32, device=None)
    P = Permutation(perm, dtype=dtype)

    U, S, V = svd(P, k=P.shape[0])

    approx = U @ S @ V.H
    rel_error = relative_error(approx.to_dense(), P.to_dense())
    assert rel_error < 1e-5

    s_vals = S.diag
    expected_s = xnp.ones((3,), dtype=dtype, device=None)
    assert relative_error(s_vals, expected_s) < 1e-5


@parametrize(all_backends)
def test_svd_block_diag(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float64
    A = Diagonal(xnp.array([1., 2.], dtype=dtype, device=None))
    B = Diagonal(xnp.array([3., 4.], dtype=dtype, device=None))
    BD = BlockDiag(A, B)

    U, S, V = svd(BD, k=BD.shape[0])

    approx = U @ S @ V.H
    rel_error = relative_error(approx.to_dense(), BD.to_dense())
    assert rel_error < 1e-5


@parametrize(all_backends)
def test_svd_kronecker(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float64
    A = Diagonal(xnp.array([1., 2.], dtype=dtype, device=None))
    B = Diagonal(xnp.array([3., 4.], dtype=dtype, device=None))
    K = Kronecker(A, B)

    U, S, V = svd(K, k=K.shape[0])

    approx = U @ S @ V.H
    rel_error = relative_error(approx.to_dense(), K.to_dense())
    assert rel_error < 1e-5
