import cola as co
from cola.fns import lazify
from cola.ops import Tridiagonal
from cola.algorithms.lanczos import get_lu_from_tridiagonal
from cola.algorithms.lanczos import construct_tridiagonal
from cola.linalg.nullspace import nullspace
from cola.linalg.eigs import power_iteration
from cola.fns import kron
from cola.utils.test_utils import get_xnp, parametrize, relative_error
from cola.backends import all_backends
from cola.utils.test_utils import generate_spectrum, generate_pd_from_diag


_tol = 1e-7


@parametrize(all_backends)
def test_inv(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    diag = generate_spectrum(coeff=0.75, scale=1.0, size=25)
    A = xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype), dtype=dtype, device=None)
    rhs = xnp.ones(shape=(A.shape[0], 5), dtype=dtype, device=None)
    soln = xnp.solve(A, rhs)

    approx = co.inv(lazify(A)) @ rhs

    rel_error = relative_error(soln, approx)
    assert rel_error < _tol * 10


@parametrize(all_backends)
def test_power_iteration(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    A = xnp.diag(xnp.array([10., 9.75, 3., 0.1], dtype=dtype, device=None))
    B = lazify(A)
    soln = xnp.array(10., dtype=dtype, device=None)
    tol, max_iter = 1e-5, 500
    _, approx, _ = power_iteration(B, tol=tol, max_iter=max_iter, momentum=0.)
    rel_error = relative_error(soln, approx)
    assert rel_error < tol * 100


@parametrize(all_backends)
def test_get_lu_from_tridiagonal(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    alpha = [-1., -1., -1.]
    beta = [2., 2., 2., 1.]
    gamma = [-1., -1., -1.]
    alpha_j = xnp.array([alpha], dtype=dtype, device=None).T
    beta_j = xnp.array([beta], dtype=dtype, device=None).T
    gamma_j = xnp.array([gamma], dtype=dtype, device=None).T
    B = Tridiagonal(alpha=alpha_j, beta=beta_j, gamma=gamma_j)
    eigenvals = xnp.jit(get_lu_from_tridiagonal, static_argnums=(0, ))(B)
    actual = xnp.array([1 / 4, 4 / 3, 3 / 2, 2], dtype=dtype, device=None)
    sorted_eigenvals = xnp.sort(eigenvals)
    rel_error = relative_error(actual, sorted_eigenvals)
    assert rel_error < _tol


@parametrize(all_backends)
def test_construct_tridiagonal(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    alpha = [0.73, 1.5, 0.4]
    beta = [0.8, 0.29, -0.6, 0.9]
    gamma = [0.04, 0.59, 1.1]
    T = [[beta[0], gamma[0], 0, 0], [alpha[0], beta[1], gamma[1], 0],
         [0., alpha[1], beta[2], gamma[2]], [0., 0., alpha[2], beta[3]]]
    alpha = xnp.array([alpha], dtype=dtype, device=None).T
    beta = xnp.array([beta], dtype=dtype, device=None).T
    gamma = xnp.array([gamma], dtype=dtype, device=None).T
    T_actual = xnp.array(T, dtype=dtype, device=None)
    fn = xnp.jit(construct_tridiagonal)
    T_soln = fn(alpha, beta, gamma)
    rel_error = relative_error(T_actual, T_soln)
    assert T_actual.shape == T_soln.shape
    assert rel_error < _tol


@parametrize(all_backends)
def ignore_test_nullspace(backend):
    xnp = get_xnp(backend)
    # TODO: add test for double precision (pytorch fails with nan while jax succeeds)
    dtype = xnp.float32
    tol = 1e-4
    A = xnp.randn(12, 20, dtype=dtype)
    U, S, VT = xnp.svd(A, full_matrices=False)
    # S[3:5] = 0
    S = xnp.fill_at(S, slice(3, 5), 0.)
    A = U @ xnp.diag(S) @ VT
    B = xnp.randn(11, 4, dtype=dtype)
    U, S, VT = xnp.svd(B, full_matrices=False)
    S = xnp.fill_at(S, slice(2, 3), 0.)
    B = U @ xnp.diag(S) @ VT
    C = kron(A, B)
    assert C.dtype == dtype
    Q, _ = nullspace(C, pbar=True, info=True, tol=tol, method='iterative')
    Q2 = nullspace(C, pbar=True, tol=tol, method='svd')
    assert xnp.norm(Q @ Q.T - Q2 @ Q2.T) < tol, "Krylov and SVD nullspaces differ"
