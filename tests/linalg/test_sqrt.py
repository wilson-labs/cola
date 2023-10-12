from cola.fns import kron
from cola.fns import lazify
# from cola.linalg import sqrt
from cola.linalg.unary.unary import apply_unary
from cola.linalg.algorithm_base import Auto
from cola.ops import Diagonal
from cola.annotations import SelfAdjoint
from cola.utils.test_utils import get_xnp, parametrize, relative_error
from cola.backends import all_backends
from cola.utils.test_utils import generate_spectrum, generate_pd_from_diag

_tol = 1e-6


@parametrize(all_backends)
def test_diagonal(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    diag = xnp.array([0.1, 0.2, 3., 4.], dtype=dtype, device=None)
    C = xnp.diag(diag**0.5)
    # B = sqrt(Diagonal(diag=diag))
    B = apply_unary(xnp.sqrt, Diagonal(diag=diag), Auto())

    rel_error = relative_error(C, B.to_dense())
    assert rel_error < _tol


@parametrize(all_backends)
def test_kronecker(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    diag = xnp.array([9., 4., 9., 4.], dtype=dtype, device=None)
    diag1 = Diagonal(xnp.array([3., 3.], dtype=dtype, device=None))
    diag2 = Diagonal(xnp.array([3., 4. / 3.], dtype=dtype, device=None))
    soln = xnp.diag(diag**0.5)
    K = kron(diag1, diag2)
    # approx = sqrt(K)
    approx = apply_unary(xnp.sqrt, K, Auto())

    rel_error = relative_error(soln, approx.to_dense())
    assert rel_error < _tol


@parametrize(all_backends)
def test_general(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    diag = generate_spectrum(coeff=0.75, scale=1.0, size=15)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype, seed=21), dtype=dtype, device=None)
    A = SelfAdjoint(lazify(A))
    soln = xnp.array(generate_pd_from_diag(diag**0.5, dtype=diag.dtype, seed=21), dtype=dtype, device=None)
    # approx = sqrt(A).to_dense()
    approx = apply_unary(xnp.sqrt, A, Auto()).to_dense()

    rel_error = relative_error(soln, approx)
    assert rel_error < _tol
