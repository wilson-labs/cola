import numpy as np
import scipy
from scipy.io import mmread
import pytest
from cola.fns import lazify
from cola.ops import Sparse
from cola.linalg.eig.lobpcg import lobpcg
from cola.utils.test_utils import get_xnp, parametrize, relative_error
from cola.backends import all_backends
from cola.utils.test_utils import generate_spectrum, generate_pd_from_diag
from cola.utils.test_utils import transform_to_csr


@pytest.mark.market
@parametrize(['torch'])
def test_matrix_market(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float64
    input_path_s = [
        "./tests/data/1138_bus.mtx",
    ]
    for input_path in input_path_s:
        print(input_path)
        matrix = mmread(input_path)
        data, col_ind, rowptr, shape = transform_to_csr(matrix.tocsc(), xnp=xnp, dtype=dtype)
        A = Sparse(data, col_ind, rowptr, shape)

        max_iters, tol = A.shape[0], 1e-10
        approx, *_ = lobpcg(A, max_iters=max_iters, tol=tol)
        approx = xnp.sort(xnp.array(approx, dtype=dtype, device=None))
        soln, *_ = scipy.linalg.eigh(matrix.toarray())
        soln = xnp.array(soln, dtype=dtype, device=None)

        rel_error = relative_error(approx[-25:], soln[-25:])
        print(f"Rel error: {rel_error:2.5e}")
        assert rel_error < 5e-7


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
