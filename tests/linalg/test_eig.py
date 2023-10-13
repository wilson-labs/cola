import numpy as np
from cola.fns import lazify
from cola.ops.operators import Diagonal
from cola.ops.operators import Identity
from cola.ops.operators import Triangular
from cola.annotations import SelfAdjoint
from cola.linalg.decompositions.decompositions import Lanczos
from cola.linalg.decompositions.decompositions import Arnoldi
from cola.linalg.algorithm_base import Auto
from cola.linalg.eig.eigs import eig
from cola.utils.test_utils import get_xnp, parametrize, relative_error
from cola.backends import all_backends
from cola.utils.test_utils import generate_spectrum, generate_pd_from_diag

_tol = 1e-6


@parametrize(all_backends)
def test_general(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype, seed=21), dtype=dtype, device=None)
    soln_vals = xnp.sort(xnp.array(diag, dtype=dtype, device=None))
    A = SelfAdjoint(lazify(A))

    eig_vals, eig_vecs = eig(A, A.shape[0], "LM", Auto(tol=1e-6))
    eig_vals, eig_vecs = xnp.cast(eig_vals, dtype), xnp.cast(eig_vecs.to_dense(), dtype)
    approx = eig_vecs @ xnp.diag(eig_vals) @ eig_vecs.T
    rel_error = relative_error(soln_vals, xnp.sort(eig_vals))
    assert rel_error < _tol
    rel_error = relative_error(A.to_dense(), approx)
    assert rel_error < _tol * 5
    eig_vals, eig_vecs = eig(A, 2, "SM", Auto(tol=1e-6))
    eig_vals = xnp.cast(eig_vals, dtype)
    rel_error = relative_error(soln_vals[:2], eig_vals)
    assert rel_error < _tol * 5
    assert eig_vecs.shape == (10, 2)

    A.annotations = set()
    eig_vals, eig_vecs = eig(A, A.shape[0], "LM", Arnoldi(tol=1e-6, max_iters=A.shape[-1]))
    eig_vals, eig_vecs = xnp.cast(eig_vals, dtype), xnp.cast(eig_vecs.to_dense(), dtype)
    approx = eig_vecs @ xnp.diag(eig_vals) @ eig_vecs.T
    rel_error = relative_error(soln_vals, xnp.sort(eig_vals))
    assert rel_error < 5e-3
    rel_error = relative_error(A.to_dense(), approx)
    assert rel_error < 5e-2

    eig_vals, eig_vecs = eig(SelfAdjoint(A), A.shape[0], "LM", Lanczos(tol=1e-6, max_iters=A.shape[-1]))
    eig_vals, eig_vecs = xnp.cast(eig_vals, dtype), xnp.cast(eig_vecs.to_dense(), dtype)
    approx = eig_vecs @ xnp.diag(eig_vals) @ eig_vecs.T
    rel_error = relative_error(soln_vals, xnp.sort(eig_vals))

    assert rel_error < 5e-3
    rel_error = relative_error(A.to_dense(), approx)
    assert rel_error < 5e-2


@parametrize(all_backends)
def test_Selfadjoint(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    diag = generate_spectrum(coeff=0.5, scale=1.0, size=10)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype, seed=21), dtype=dtype, device=None)
    soln_vals = xnp.sort(xnp.array(diag, dtype=dtype, device=None))
    A = SelfAdjoint(lazify(A))
    eig_vals, eig_vecs = eig(A, A.shape[0], "LM", Auto())
    approx = eig_vecs @ xnp.diag(eig_vals) @ eig_vecs.T

    rel_error = relative_error(soln_vals, eig_vals)
    assert rel_error < _tol

    rel_error = relative_error(A.to_dense(), approx)
    assert rel_error < _tol

    eig_vals, eig_vecs = eig(A, A.shape[0], "LM", Lanczos(max_iters=A.shape[-1]))
    approx = eig_vecs @ xnp.diag(eig_vals) @ eig_vecs.T

    rel_error = relative_error(soln_vals, eig_vals)
    assert rel_error < _tol

    rel_error = relative_error(A.to_dense(), approx)
    assert rel_error < _tol


@parametrize(all_backends)
def test_triangular(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    A = xnp.array([[1., 2., 3.], [0., 6., 5.], [0., 0., 4.]], dtype=dtype, device=None)
    soln_vecs = compute_lower_triangular_eigvecs(np.array(A))
    A = Triangular(A)
    soln_vals = xnp.array([1., 4., 6.], dtype=dtype, device=None)
    soln_vecs = xnp.array(soln_vecs, dtype=dtype, device=None)[:, [0, 2, 1]]
    eig_vals, eig_vecs = eig(A, k=A.shape[0])

    assert relative_error(soln_vals, eig_vals) < _tol
    assert relative_error(soln_vecs, eig_vecs.to_dense()) < _tol
    assert relative_error(A.to_dense() @ soln_vecs, soln_vals[None, :] * soln_vecs) < _tol


@parametrize(all_backends)
def test_identity(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    A = Identity(shape=(4, 4), dtype=dtype)
    soln_vals = xnp.array([1., 1., 1., 1.], dtype=dtype, device=None)
    eig_slice = slice(0, 3, None)
    soln_vecs = xnp.eye(4, 4, dtype=dtype, device=None)
    soln_vecs = soln_vecs[:, eig_slice]
    eig_vals, eig_vecs = eig(A, k=3, which="SM")

    assert relative_error(soln_vals[eig_slice], eig_vals) < _tol
    assert relative_error(soln_vecs[:, eig_slice], eig_vecs.to_dense()) < _tol


@parametrize(all_backends)
def test_diagonal(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    diag = xnp.array([0.1, 3., 0.2, 4.], dtype=dtype, device=None)
    soln_vecs = [[1., 0., 0., 0.], [0., 0., 1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]]
    soln_vecs = xnp.array(soln_vecs, dtype=dtype, device=None)
    D = Diagonal(diag)
    eig_vals, eig_vecs = eig(D, k=D.shape[0])
    eig_vecs = eig_vecs.to_dense()

    rel_error = relative_error(xnp.sort(diag), eig_vals)
    assert rel_error < _tol
    assert relative_error(soln_vecs, eig_vecs) < _tol


def compute_lower_triangular_eigvecs(L):
    eigvals = np.diag(L)
    eigvecs = np.eye(L.shape[0])
    for i in range(1, L.shape[0]):
        A = L[:i, :i] - eigvals[i] * np.eye(i)
        out = np.linalg.solve(A, -L[:i, i])
        eigvecs[:i, i] = out
    return eigvecs
