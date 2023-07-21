import cola
from cola import jax_fns
from cola import torch_fns
from cola.fns import lazify
from cola.utils_test import parametrize, relative_error
from cola.utils_test import generate_spectrum, generate_pd_from_diag


@parametrize([torch_fns, jax_fns])
def test_random_linear_system(xnp):
    dtype = xnp.float32
    diag = generate_spectrum(coeff=0.75, scale=1.0, size=25)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype), dtype=dtype)
    rhs = xnp.ones(shape=(A.shape[0], 5), dtype=dtype)
    soln = xnp.solve(A, rhs)

    approx = cola.solve(lazify(A), rhs)
    rel_error = relative_error(soln, approx)
    assert rel_error < 1e-6
