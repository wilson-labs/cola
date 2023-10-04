import numpy as np
from cola.fns import lazify
from cola.linalg.eig.lobpcg import lobpcg
from cola.utils.test_utils import get_xnp, parametrize, relative_error
from cola.backends import all_backends
from cola.utils.test_utils import generate_spectrum, generate_pd_from_diag


@parametrize(all_backends)
def test_lobpcg_random(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    np_dtype = np.float32
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10, dtype=np_dtype)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype, seed=21), dtype=dtype, device=None)

    max_iters, tol = A.shape[0], 1e-7
    eigvals, _, _ = lobpcg(lazify(A), max_iters=max_iters, tol=tol)

    diag = xnp.array(diag[:-1], dtype=dtype, device=None)
    idx = xnp.argsort(diag, axis=-1)
    rel_error = relative_error(eigvals, diag[idx])
    assert rel_error < 5e-5
