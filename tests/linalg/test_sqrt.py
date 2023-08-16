from cola import jax_fns
from cola import torch_fns
from cola.fns import kron
from cola.fns import lazify
from cola.linalg import sqrt
from cola.ops import Diagonal
from cola.annotations import SelfAdjoint
from cola.utils_test import parametrize, relative_error
from cola.utils_test import generate_spectrum, generate_pd_from_diag
from jax.config import config

config.update('jax_platform_name', 'cpu')
_tol = 1e-6


@parametrize([torch_fns, jax_fns])
def test_diagonal(xnp):
    dtype = xnp.float32
    diag = xnp.array([0.1, 0.2, 3., 4.], dtype=dtype)
    C = xnp.diag(diag**0.5)
    B = sqrt(Diagonal(diag=diag))

    rel_error = relative_error(C, B.to_dense())
    assert rel_error < _tol


@parametrize([torch_fns, jax_fns])
def test_kronecker(xnp):
    dtype = xnp.float32
    diag = xnp.array([9., 4., 9., 4.], dtype=dtype)
    diag1 = Diagonal(xnp.array([3., 3.], dtype=dtype))
    diag2 = Diagonal(xnp.array([3., 4. / 3.], dtype=dtype))
    soln = xnp.diag(diag**0.5)
    K = kron(diag1, diag2)
    approx = sqrt(K)

    rel_error = relative_error(soln, approx.to_dense())
    assert rel_error < _tol


@parametrize([torch_fns, jax_fns])
def test_general(xnp):
    dtype = xnp.float32
    diag = generate_spectrum(coeff=0.75, scale=1.0, size=15)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype, seed=21), dtype=dtype)
    A = SelfAdjoint(lazify(A))
    soln = xnp.array(generate_pd_from_diag(diag ** 0.5, dtype=diag.dtype, seed=21), dtype=dtype)
    approx = sqrt(A).to_dense()

    rel_error = relative_error(soln, approx)
    assert rel_error < _tol
