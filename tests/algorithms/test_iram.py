import numpy as np

from cola.backends import all_backends
from cola.fns import lazify
from cola.linalg.eig.iram import iram
from cola.utils.utils_for_tests import (
    generate_pd_from_diag,
    generate_spectrum,
    get_xnp,
    parametrize,
    relative_error,
)
from cola.utils.utils_linalg import get_numpy_dtype


@parametrize(all_backends)
def test_iram_random(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    np_dtype = get_numpy_dtype(dtype)
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10, dtype=np_dtype)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype, seed=21), dtype=dtype, device=None)
    rhs = xnp.ones(shape=(A.shape[0], ), dtype=dtype, device=None)

    max_iters, tol = A.shape[0], 1e-7
    eigvals, _, _ = iram(lazify(A), start_vector=rhs, eig_n=A.shape[0] - 2, max_iters=max_iters, tol=tol)
    eigvals = np.array(eigvals, dtype=np_dtype)
    eigvals = np.sort(eigvals)[::-1]

    rel_error = relative_error(eigvals, diag[:len(eigvals)])
    assert rel_error < 5e-5
