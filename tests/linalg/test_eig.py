from cola import jax_fns
from cola import torch_fns
from cola.fns import lazify
from cola.ops import Diagonal
from cola.annotations import SelfAdjoint
from cola.linalg.eigs import eig
from jax.config import config
from cola.utils_test import parametrize, relative_error
from cola.utils_test import generate_spectrum, generate_pd_from_diag

config.update('jax_platform_name', 'cpu')
# config.update("jax_enable_x64", True)
_tol = 1e-6


# @parametrize([torch_fns, jax_fns])
@parametrize([torch_fns])
def test_general(xnp):
    dtype = xnp.float32
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype, seed=21), dtype=dtype)
    soln_vals = xnp.sort(xnp.array(diag, dtype=dtype))
    A = lazify(A)

    eig_vals, eig_vecs = eig(A, eig_slice=slice(0, None, None), tol=1e-6)
    eig_vals, eig_vecs = xnp.cast(eig_vals, dtype), xnp.cast(eig_vecs, dtype)
    approx = eig_vecs @ xnp.diag(eig_vals) @ eig_vecs.T
    rel_error = relative_error(soln_vals, xnp.sort(eig_vals))
    assert rel_error < _tol
    rel_error = relative_error(A.to_dense(), approx)
    assert rel_error < _tol * 5
    eig_vals, eig_vecs = eig(A, eig_slice=slice(-2, None), tol=1e-6)
    eig_vals = xnp.cast(eig_vals, dtype)
    rel_error = relative_error(soln_vals[-2:], xnp.sort(eig_vals))
    assert rel_error < _tol
    assert eig_vecs.shape == (10, 2)

    A.annotations = set()
    eig_vals, eig_vecs = eig(A, tol=1e-6, method="arnoldi", max_iters=A.shape[-1])
    eig_vals, eig_vecs = xnp.cast(eig_vals, dtype), xnp.cast(eig_vecs, dtype)
    approx = eig_vecs @ xnp.diag(eig_vals) @ eig_vecs.T
    # approx = Q[:, :-1] @ eig_vecs @ xnp.diag(eig_vals) @ eig_vecs.T @ Q[:, :-1].T
    rel_error = relative_error(soln_vals, xnp.sort(eig_vals))
    assert rel_error < 5e-3
    rel_error = relative_error(A.to_dense(), approx)
    assert rel_error < 5e-2


@parametrize([torch_fns, jax_fns])
def test_adjoint(xnp):
    dtype = xnp.float32
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype, seed=21), dtype=dtype)
    soln_vals = xnp.sort(xnp.array(diag, dtype=dtype))
    A = SelfAdjoint(lazify(A))
    eig_vals, eig_vecs = eig(A)
    approx = eig_vecs @ xnp.diag(eig_vals) @ eig_vecs.T

    rel_error = relative_error(soln_vals, eig_vals)
    assert rel_error < _tol

    rel_error = relative_error(A.to_dense(), approx)
    assert rel_error < _tol

    eig_vals, eig_vecs = eig(A, method="lanczos", max_iters=A.shape[-1])
    print(eig_vals.shape, eig_vecs.shape)
    print(soln_vals.shape)
    approx = eig_vecs @ xnp.diag(eig_vals) @ eig_vecs.T

    rel_error = relative_error(soln_vals, eig_vals)
    assert rel_error < _tol

    rel_error = relative_error(A.to_dense(), approx)
    assert rel_error < _tol


@parametrize([torch_fns, jax_fns])
def test_diagonal(xnp):
    dtype = xnp.float32
    diag = xnp.array([0.1, 3., 0.2, 4.], dtype=dtype)
    soln_vecs = [[1., 0., 0., 0.], [0., 0., 1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]]
    soln_vecs = xnp.array(soln_vecs, dtype=dtype)
    eig_vals, eig_vecs = eig(Diagonal(diag=diag))

    rel_error = relative_error(xnp.sort(diag), eig_vals)
    assert rel_error < _tol
    assert relative_error(soln_vecs, eig_vecs) < _tol
