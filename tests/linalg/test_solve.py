import cola
from cola.fns import lazify
from cola.utils_test import get_xnp, parametrize, relative_error
from cola.utils_test import generate_spectrum, generate_pd_from_diag


@parametrize(['torch', 'jax'])
def test_random_linear_system(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    diag = generate_spectrum(coeff=0.75, scale=1.0, size=25)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype), dtype=dtype, device=None)
    rhs = xnp.ones(shape=(A.shape[0], 5), dtype=dtype, device=None)
    soln = xnp.solve(A, rhs)

    approx = cola.solve(cola.PSD(lazify(A)), rhs)
    rel_error = relative_error(soln, approx)
    assert rel_error < 1e-6
