import numpy as np
from cola import jax_fns
from cola import torch_fns
from cola.fns import lazify
from cola.linalg.inverse import inverse
from cola.algorithms.gmres import gmres
from cola.utils_test import parametrize, relative_error
from cola.utils_test import generate_spectrum, generate_pd_from_diag
from jax.config import config

config.update('jax_platform_name', 'cpu')


@parametrize([torch_fns, jax_fns])
def test_gmres_random(xnp):
    dtype = xnp.float32
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=25, dtype=np.float32)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype), dtype=dtype)
    rhs = xnp.ones(shape=(A.shape[0], ), dtype=dtype)
    soln = xnp.solve(A, rhs)

    max_iters, tol = A.shape[0] - 5, 1e-8
    A_inv = inverse(lazify(A), max_iters=max_iters, tol=tol, method="gmres")
    approx = A_inv @ rhs

    rel_error = relative_error(soln, approx)
    assert rel_error < 5e-4


@parametrize([torch_fns, jax_fns])
def test_gmres_easy(xnp):
    dtype = xnp.float32
    A = xnp.diag(xnp.array([3., 4., 5.], dtype=dtype))
    rhs = [[1], [1], [1]]
    rhs = xnp.array(rhs, dtype=dtype)
    soln = [[1 / 3], [1 / 4], [1 / 5]]
    soln = xnp.array(soln, dtype=dtype)

    max_iters, tolerance = 3, 1e-8
    fn = gmres
    x0 = xnp.zeros_like(rhs)

    approx, _ = fn(lazify(A), rhs, x0, max_iters, tolerance)
    rel_error = relative_error(soln, approx)
    assert rel_error < 1e-7

    approx, _ = fn(lazify(A), rhs, x0, max_iters, tolerance, use_householder=False,
                   use_triangular=True)
    rel_error = relative_error(soln, approx)
    assert rel_error < 1e-7

    approx, _ = fn(lazify(A), rhs, x0, max_iters, tolerance, use_householder=True,
                   use_triangular=False)
    rel_error = relative_error(soln, approx)
    assert rel_error < 5e-7

    approx, _ = fn(lazify(A), rhs, x0, max_iters, tolerance, use_householder=True,
                   use_triangular=True)
    rel_error = relative_error(soln, approx)
    assert rel_error < 5e-7
