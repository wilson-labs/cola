from cola import jax_fns
import numpy as np
from cola import torch_fns
from cola.ops import Diagonal
from cola.algorithms.stochastic_lanczos_quad import stochastic_lanczos_quad
from cola.algorithms.stochastic_lanczos_quad import slq_fwd
from cola.fns import lazify
from cola.utils_test import parametrize, relative_error
from cola.utils_test import generate_spectrum, generate_pd_from_diag
from jax.config import config
config.update('jax_platform_name', 'cpu')


# @parametrize([torch_fns, jax_fns])
@parametrize([torch_fns])
def test_slq_vjp(xnp):
    dtype = xnp.float32
    diag = xnp.Parameter(xnp.array([3., 4., 5.], dtype=dtype))
    diag_soln = xnp.Parameter(xnp.array([3., 4., 5.], dtype=dtype))
    _, unflatten = Diagonal(diag).flatten()

    def f(theta):
        A = unflatten([theta])
        loss = slq_fwd(A, xnp.log, num_samples=1000, max_iters=100, tol=1e-6, pbar=False)
        return loss

    def f_alt(theta):
        X = xnp.diag(theta)
        loss = xnp.logdet(X)
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
    assert rel_error < 1e-1


@parametrize([torch_fns, jax_fns])
def test_stochastic_lanczos_quad_random(xnp):
    dtype = xnp.float32
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10, dtype=np.float32)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype), dtype=dtype)

    def fun(x):
        return xnp.log(x)

    soln = xnp.sum(fun(xnp.array(diag, dtype=dtype)))
    num_samples, max_iters, tol = 70, A.shape[0], 1e-7
    approx = stochastic_lanczos_quad(lazify(A), fun, num_samples, max_iters, tol)

    rel_error = relative_error(soln, approx)
    assert rel_error < 1e-1
