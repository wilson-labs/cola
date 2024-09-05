from cola.backends import all_backends
from cola.fns import lazify
from cola.linalg.algorithm_base import Auto
from cola.linalg.decompositions.decompositions import Lanczos
from cola.linalg.eig.lobpcg import LOBPCG
from cola.linalg.svd.svd import svd
from cola.utils.test_utils import generate_pd_from_diag, generate_spectrum, get_xnp, parametrize, relative_error

_tol = 1e-6


@parametrize(all_backends)
def test_svd(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10)
    A = lazify(xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype, seed=21), dtype=dtype, device=None))
    soln_vals = xnp.sort(xnp.array(diag, dtype=dtype, device=None))

    U, Sigma, V = svd(A, A.shape[0], "LM", Auto(tol=1e-6))

    rel_error = relative_error(soln_vals, Sigma.diag)
    print(f"Rel error: {rel_error:2.5e}")
    assert rel_error < _tol * 5

    rel_error = relative_error((U @ Sigma @ V.H).to_dense(), A.to_dense())
    print(f"Rel error: {rel_error:2.5e}")
    assert rel_error < _tol * 5

    U, Sigma, V = svd(A, A.shape[0], "LM", Lanczos(tol=1e-6))

    rel_error = relative_error(soln_vals, Sigma.diag)
    print(f"Rel error: {rel_error:2.5e}")
    assert rel_error < _tol * 5

    rel_error = relative_error((U @ Sigma @ V.H).to_dense(), A.to_dense())
    print(f"Rel error: {rel_error:2.5e}")
    assert rel_error < _tol * 5

    U, Sigma, V = svd(A, A.shape[0], "LM", LOBPCG())

    rel_error = relative_error(soln_vals[-Sigma.shape[0]:], Sigma.diag)
    print(f"Rel error: {rel_error:2.5e}")
    assert rel_error < _tol * 5
