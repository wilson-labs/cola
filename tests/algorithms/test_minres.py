from cola.backends import all_backends
from cola.fns import lazify
from cola.linalg.inverse.minres import minres
from cola.utils.utils_for_tests import (
    generate_pd_from_diag,
    generate_spectrum,
    get_xnp,
    parametrize,
    relative_error,
)
from cola.utils.utils_linalg import get_numpy_dtype


@parametrize(all_backends)
def test_minres_random(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    np_dtype = get_numpy_dtype(dtype)
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10, dtype=np_dtype)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype, seed=21), dtype=dtype, device=None)
    A = A @ A.T
    rhs = xnp.ones(shape=(A.shape[0], ), dtype=dtype, device=None)

    tol = 1e-10
    soln, info = minres(lazify(A), rhs=rhs, tol=tol, max_iters=None)

    rel_error = relative_error(A @ soln, rhs)
    print(f"{rel_error=:1.3e}")
    assert rel_error < 5e-5
    assert info["success"] == 0
