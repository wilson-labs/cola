import numpy as np
from cola.ops import Diagonal
from cola.algorithms import stochastic_lanczos_quad
from cola.fns import lazify
from cola.utils_test import get_xnp, parametrize, relative_error
from cola.utils_test import generate_spectrum, generate_pd_from_diag


# @parametrize(['torch', 'jax'])
@parametrize(['torch'])
def test_slq_vjp(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    diag = xnp.Parameter(xnp.array([3., 4., 5.], dtype=dtype, device=None))
    diag_soln = xnp.Parameter(xnp.array([3., 4., 5.], dtype=dtype, device=None))
    _, unflatten = Diagonal(diag).flatten()

    def f(theta):
        A = unflatten([theta])
        loss = stochastic_lanczos_quad(A, xnp.log, vtol=1 / 10, max_iters=100, tol=1e-6, pbar=False)
        return loss

    def f_alt(theta):
        X = xnp.diag(theta)
        loss = xnp.logdet(X)
        return loss

    out = f(diag)
    if backend == 'torch':
        out.backward()
        approx = diag.grad.clone()
    else:
        approx = xnp.grad(f)(diag)
    assert approx is not None

    out = f_alt(diag_soln)
    if backend == 'torch':
        out.backward()
        soln = diag_soln.grad.clone()
    else:
        soln = xnp.grad(f_alt)(diag)

    rel_error = relative_error(soln, approx)
    assert rel_error < 1e-1


@parametrize(['torch', 'jax'])
def test_stochastic_lanczos_quad_random(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10, dtype=np.float32)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype), dtype=dtype, device=None)

    def fun(x):
        return xnp.log(x)

    soln = xnp.sum(fun(xnp.array(diag, dtype=dtype, device=None)))
    vtol, max_iters, tol = 1 / np.sqrt(70), A.shape[0], 1e-7
    approx = stochastic_lanczos_quad(lazify(A), fun, max_iters=max_iters, tol=tol, vtol=vtol)

    rel_error = relative_error(soln, approx)
    assert rel_error < 1e-1
