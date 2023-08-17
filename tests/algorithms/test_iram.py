import numpy as np
from cola import jax_fns
from cola import torch_fns
from cola.fns import lazify
from cola.algorithms.iram import iram
from cola.utils_test import parametrize, relative_error
from cola.utils_test import generate_spectrum, generate_pd_from_diag
from jax.config import config

config.update('jax_platform_name', 'cpu')


@parametrize([torch_fns, jax_fns])
def test_iram_random(xnp):
    dtype = xnp.float32
    np_dtype = np.float32
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10, dtype=np_dtype)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype, seed=21), dtype=dtype)
    rhs = xnp.ones(shape=(A.shape[0], 1), dtype=dtype)

    max_iters, tol = A.shape[0], 1e-7
    eigvals, _, _ = iram(lazify(A), start_vector=rhs, max_iters=max_iters, tol=tol)

    diag = xnp.array(diag[:-1], dtype=dtype)
    idx = xnp.argsort(diag, axis=-1)
    rel_error = relative_error(eigvals, diag[idx])
    assert rel_error < 5e-5
