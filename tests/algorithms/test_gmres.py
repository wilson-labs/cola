import numpy as np
from cola import jax_fns
from cola import torch_fns
from cola.ops import Identity
from cola.ops import Diagonal
from cola.fns import lazify
from cola.linalg.inverse import inverse
from cola.algorithms.gmres import gmres
from cola.algorithms.gmres import gmres_fwd
from cola.utils_test import parametrize, relative_error
from cola.utils_test import generate_spectrum, generate_pd_from_diag
from jax.config import config

config.update('jax_platform_name', 'cpu')


@parametrize([torch_fns, jax_fns])
def test_gmres_vjp(xnp):
    dtype = xnp.float32
    diag = xnp.Parameter(xnp.array([3., 4., 5.], dtype=dtype))
    diag_soln = xnp.Parameter(xnp.array([3., 4., 5.], dtype=dtype))
    A = xnp.diag(diag)
    rhs = xnp.ones(shape=(3, 1), dtype=dtype)
    max_iters, tol = 5, 1e-6
    use_householder, use_triangular, pbar = False, False, False
    x0, P = xnp.zeros_like(rhs), Identity(dtype=A.dtype, shape=A.shape)
    _, unflatten = Diagonal(diag).flatten()

    def f(theta):
        A = unflatten([theta])
        solve, _ = gmres_fwd(A, rhs, x0, max_iters, tol, P, use_householder, use_triangular, pbar)
        loss = xnp.sum(solve)
        return loss

    def f_alt(theta):
        X = xnp.diag(theta)
        solve = xnp.solve(X, rhs)
        loss = xnp.sum(solve)
        return loss

    out = f(diag)
    if xnp.__name__.find("torch") >= 0:
        out.backward()
        approx = diag.grad.clone()
    else:
        approx = xnp.grad(f)(diag)
    assert approx is not None

    out = f_alt(diag_soln)
    if xnp.__name__.find("torch") >= 0:
        out.backward()
        soln = diag_soln.grad.clone()
    else:
        soln = xnp.grad(f_alt)(diag)

    rel_error = relative_error(soln, approx)
    assert rel_error < 1e-6


@parametrize([torch_fns, jax_fns])
def test_gmres_random(xnp):
    dtype = xnp.float32
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=25, dtype=np.float32)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype), dtype=dtype)
    rhs = xnp.ones(shape=(A.shape[0], 3), dtype=dtype)
    soln = xnp.solve(A, rhs)

    max_iters, tol = A.shape[0] - 5, 1e-8
    A_inv = inverse(lazify(A), max_iters=max_iters, tol=tol, method="iterative")
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

    # approx, _ = fn(lazify(A), rhs, x0, max_iters, tolerance, use_householder=True,
    #                use_triangular=False)
    # rel_error = relative_error(soln, approx)
    # assert rel_error < 5e-7

    # approx, _ = fn(lazify(A), rhs, x0, max_iters, tolerance, use_householder=True,
    #                use_triangular=True)
    # rel_error = relative_error(soln, approx)
    # assert rel_error < 5e-7
