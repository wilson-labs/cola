import pytest
import numpy as np
from cola import jax_fns
from cola import torch_fns
from cola.fns import kron, lazify
from cola.ops import Tridiagonal
from cola.ops import Diagonal
from cola.ops import Identity
from cola.ops import I_like
from cola.ops import KronSum
from cola.ops import Sum
from cola.ops import ScalarMul
from cola.ops import Product
from cola.ops import Sliced
from cola.ops import SelfAdjoint
from cola.ops import Symmetric
from cola.ops import Householder
from cola.ops import Sparse
from cola.ops import PSD
from cola.ops import LinearOperator
from cola.algorithms.arnoldi import get_householder_vec
from cola.utils_test import parametrize, relative_error
from jax.config import config

config.update('jax_platform_name', 'cpu')

_tol = 1e-6


@parametrize([torch_fns])
def test_sparse(xnp):
    dtype = xnp.float32
    A = [[0., 1., 0., 0., 0.], [0., 2., -1., 0., 0.], [0., 0., 0., 0., 0.], [6.6, 0., 0., 0., 1.4]]
    A = xnp.array(A, dtype=dtype)
    data = xnp.array([1., 2., -1., 6.6, 1.4], dtype=dtype)
    indices = xnp.array([1, 1, 2, 0, 4], dtype=xnp.int64)
    indptr = xnp.array([0, 1, 3, 3, 5], dtype=xnp.int64)
    shape = (4, 5)
    As = Sparse(data, indices, indptr, shape)
    x = xnp.array([0.29466099, 0.71853315, -0.06172857, -0.0432496, 0.44698924], dtype=dtype)
    rel_error = relative_error(A @ x, As @ x)
    assert rel_error < _tol


@parametrize([torch_fns, jax_fns])
def test_householder(xnp):
    dtype = xnp.float32
    x = xnp.array([0.29466099, 0.71853315, -0.06172857, -0.0432496, 0.44698924], dtype=dtype)
    w = x / xnp.norm(x)
    R = Householder(w[:, None])
    rel_error = relative_error(-x, R @ x)
    assert rel_error < _tol

    for idx in [0, 2]:
        vec, beta = get_householder_vec(x, idx=idx, xnp=xnp)
        R = Householder(vec[:, None], beta=beta)
        approx = xnp.canonical(idx, x.shape, dtype) * xnp.norm(x[idx:])
        approx = xnp.update_array(approx, x[:idx], slice(None, idx, None))
        rel_error = relative_error(approx, R @ x)
        assert rel_error < _tol


@parametrize([torch_fns, jax_fns])
def test_adjoint_and_symmetric(xnp):
    dtype = xnp.float32
    A = [[-0.38105885, -1.13140889, -1.66344203], [1.38351046, 0.50665351, -0.29059844],
         [0.18752258, 0.60077083, -0.95329955]]
    Aop = SelfAdjoint(lazify(xnp.array(A, dtype=dtype)))
    eye = I_like(Aop).to_dense()
    rel_error = relative_error(Aop.to_dense().T, eye @ Aop)
    assert rel_error < _tol

    Aop = Symmetric(lazify(xnp.array(A, dtype=dtype)))
    eye = I_like(Aop).to_dense()
    rel_error = relative_error(Aop.to_dense().T, eye @ Aop)
    assert rel_error < _tol

    Aop = PSD(lazify(xnp.array(A, dtype=dtype)))
    rel_error = relative_error(Aop.to_dense(), Aop.T.to_dense())
    assert rel_error < _tol


@parametrize([torch_fns, jax_fns])
def test_unflatten(xnp):
    dtype = xnp.float32
    diag1 = xnp.array([0.1, -0.2], dtype=dtype)
    Aop = Diagonal(diag1)
    diag2 = xnp.array([3., -7.], dtype=dtype)
    Bop = Diagonal(diag2)
    Cop = KronSum(Aop, Bop)
    flattened, un_fn = Cop.flatten()
    Cop_approx = un_fn(flattened)
    rel_error = relative_error(Cop.to_dense(), Cop_approx.to_dense())
    assert rel_error < _tol
    diag3 = xnp.array([1., 2., 3., 4], dtype=dtype)
    Dop = Diagonal(diag3)
    Fop = Sum(Cop, Dop)
    flattened, un_fn = Fop.flatten()
    Fop_approx = un_fn(flattened)
    rel_error = relative_error(Fop.to_dense(), Fop_approx.to_dense())
    assert rel_error < _tol


@parametrize([torch_fns, jax_fns])
def test_flatten(xnp):
    dtype = xnp.float32
    diag1 = xnp.array([0.1, -0.2], dtype=dtype)
    Aop = Diagonal(diag1)
    diag2 = xnp.array([3., -7.], dtype=dtype)
    Bop = Diagonal(diag2)
    Cop = KronSum(Aop, Bop)
    diag3 = xnp.array([1., 2., 3., 4], dtype=dtype)
    Dop = Diagonal(diag3)
    Fop = Sum(Cop, Dop)
    flattened, _ = Fop.flatten()
    soln = [diag1, diag2, diag3]
    assert flattened == soln


@parametrize([torch_fns, jax_fns])
def test_get_item(xnp):
    dtype = xnp.float32
    np.random.seed(seed=48)
    A = np.random.normal(size=(5, 4))
    A = xnp.array(A, dtype=dtype)
    B = lazify(A)
    approx = B[..., 0, 2]
    soln = A[..., 0, 2]
    rel_error = relative_error(soln, approx)
    assert rel_error < _tol

    approx = B[[0, 2], [3, -1]]
    soln = A[[0, 2], [3, -1]]
    rel_error = relative_error(soln, approx)
    assert rel_error < _tol


@parametrize([torch_fns, jax_fns])
def test_sliced(xnp):
    dtype = xnp.float32
    A = [[-1.0165, -1.9161, 0.8602, -0.4597], [1.1470, 0.9879, -1.0831, 1.2768],
         [1.0912, 0.6360, -0.8962, 0.7849], [-0.4359, 0.5612, 1.4560, 0.0682]]
    A = xnp.array(A, dtype=dtype)
    slices = (slice(1, 3, None), slice(0, 3, None))
    rhs = [[-0.3018, 0.0266, -2.4584], [0.3041, -1.2548, 0.0115], [-0.4590, -3.0145, 0.5659]]
    rhs = xnp.array(rhs, dtype=dtype)
    soln = A[1:3, 0:3] @ rhs
    B = Sliced(lazify(A), slices=slices)
    C = lazify(A)
    approx = B @ rhs

    print(str(B))
    rel_error = relative_error(soln, approx)
    assert rel_error < _tol
    rel_error = relative_error(A[1:3, 0:3], B.to_dense())
    assert rel_error < _tol
    rel_error = relative_error(A[1, 2], C[1, 2])
    assert rel_error < _tol
    rel_error = relative_error(A[1:3, 0:3], C[slices].to_dense())
    assert rel_error < _tol


@parametrize([torch_fns, jax_fns])
def test_product_op(xnp):
    dtype = xnp.float32
    A = [[-1.0165, -1.9161, 0.8602, -0.4597], [1.1470, 0.9879, -1.0831, 1.2768],
         [1.0912, 0.6360, -0.8962, 0.7849], [-0.4359, 0.5612, 1.4560, 0.0682]]
    A = xnp.array(A, dtype=dtype)
    B = [[-1.3213, 0.8053, -0.4334, 0.6765], [0.5056, 0.5160, 0.6120, -0.8747],
         [-1.8671, -0.1413, 1.6811, -0.5727], [-1.7080, 0.0548, -0.1522, -0.2481]]
    D = [[0.3105, 0.2883, 0.4809, -0.0063], [0.6082, 1.0688, -1.6521, -1.4216]]
    D = xnp.array(D, dtype=dtype)
    B = xnp.array(B, dtype=dtype)
    rhs = [[1., 3.], [4., 5.], [6., -8.], [-1., 0.2]]
    rhs = xnp.array(rhs, dtype=dtype)
    soln = A @ (B @ rhs)

    with pytest.raises(ValueError):
        C = Product(lazify(A), lazify(D))
    C = Product(lazify(A), lazify(B))
    approx = C @ rhs

    print(str(C))
    rel_error = relative_error(soln, approx)
    assert rel_error < _tol
    rel_error = relative_error((A @ B), C.to_dense())
    assert rel_error < _tol


@parametrize([torch_fns, jax_fns])
def test_sum_and_scalar_op(xnp):
    dtype = xnp.float32
    A = [[-1.0165, -1.9161, 0.8602, -0.4597], [1.1470, 0.9879, -1.0831, 1.2768],
         [1.0912, 0.6360, -0.8962, 0.7849], [-0.4359, 0.5612, 1.4560, 0.0682]]
    A = xnp.array(A, dtype=dtype)
    B = [[-1.3213, 0.8053, -0.4334, 0.6765], [0.5056, 0.5160, 0.6120, -0.8747],
         [-1.8671, -0.1413, 1.6811, -0.5727], [-1.7080, 0.0548, -0.1522, -0.2481]]
    B = xnp.array(B, dtype=dtype)
    rhs = [[1., 3.], [4., 5.], [6., -8.], [-1., 0.2]]
    rhs = xnp.array(rhs, dtype=dtype)
    soln = (A + B) @ rhs
    C = Sum(lazify(A), lazify(B))
    approx = C @ rhs
    D = ScalarMul(c=-5., shape=C.shape, dtype=dtype)

    print(str(C))
    print(str(D))
    rel_error = relative_error(soln, approx)
    assert rel_error < _tol
    rel_error = relative_error((-5.) * (A + B), D @ (C.to_dense()))
    assert rel_error < _tol


@parametrize([torch_fns, jax_fns])
def test_kron_sum(xnp):
    dtype = xnp.float32
    diag = xnp.array([0.1, -0.2], dtype=dtype)
    Aop = Diagonal(diag=diag)
    A = xnp.diag(diag)
    B = xnp.eye(2, 2, dtype=dtype)
    Id = xnp.eye(A.shape[0], A.shape[1], dtype=dtype)
    Bop = lazify(B)
    rhs = [[1., 3.], [4., 5.], [6., -8.], [-1., 0.2]]
    rhs = xnp.array(rhs, dtype=dtype)
    Cop = KronSum(Aop, Bop)
    approx = Cop @ rhs
    kron_dense = xnp.kron(A, Id) + xnp.kron(Id, B)
    soln = kron_dense @ rhs

    print(str(Cop))

    rel_error = relative_error(soln, approx)
    assert rel_error < _tol
    rel_error = relative_error(kron_dense, Cop.to_dense())
    assert rel_error < _tol


@parametrize([torch_fns, jax_fns])
def test_identity(xnp):
    dtype = xnp.float32
    Id = xnp.eye(3, 3, dtype=dtype)
    rhs = [[1., 3.], [4., 5.], [6., -8.]]
    rhs = xnp.array(rhs, dtype=dtype)
    soln = rhs
    B = Identity(dtype=dtype, shape=Id.shape)
    approx = B @ rhs
    C = I_like(Id)

    rel_error = relative_error(soln, approx)
    assert rel_error < _tol
    rel_error = relative_error(Id, B.to_dense())
    assert rel_error < _tol
    rel_error = relative_error(Id, C.to_dense())
    assert rel_error < _tol


@parametrize([torch_fns, jax_fns])
def test_diagonal(xnp):
    dtype = xnp.float32
    diag = xnp.array([0.1, 0.2, 3., 4.], dtype=dtype)
    A = xnp.diag(diag)
    rhs = xnp.ones(shape=(A.shape[0], 6), dtype=dtype)
    soln = A @ rhs
    B = Diagonal(diag=diag)
    approx = B @ rhs

    rel_error = relative_error(soln, approx)
    assert rel_error < _tol


@parametrize([torch_fns, jax_fns])
def test_diagonal_variants(xnp):
    dtype = xnp.float32
    value = xnp.array(0.1, dtype=dtype)
    diag = xnp.array([0.1, 0.1, 0.1, 0.1], dtype=dtype)
    A = xnp.diag(diag)
    rhs = xnp.ones(shape=(A.shape[0], 6), dtype=dtype)
    soln = A @ rhs
    B = ScalarMul(value, A.shape, dtype=dtype)
    approx = B @ rhs

    rel_error = relative_error(soln, approx)
    assert rel_error < _tol

    diag = xnp.array([2., 4., 2., 4.], dtype=dtype)
    diag1 = Diagonal(xnp.array([1., 1.], dtype=dtype))
    diag2 = Diagonal(xnp.array([2., 4.], dtype=dtype))
    soln = xnp.diag(diag)
    approx = kron(diag1, diag2)

    rel_error = relative_error(soln, approx.to_dense())
    assert rel_error < _tol


@parametrize([torch_fns, jax_fns])
def test_tridiagonal(xnp):
    alpha = [-0.5, 4., 5.]
    beta = [0.1, 0.2, -0.1, 0.4]
    gamma = [-1.0, 2., 3.]
    A = [[beta[0], gamma[0], 0, 0], [alpha[0], beta[1], gamma[1], 0],
         [0., alpha[1], beta[2], gamma[2]], [0., 0., alpha[2], beta[3]]]
    X = [[-0.23742934, -1.28097845], [-0.4654211, -0.42619589], [-1.78554193, -1.01728456],
         [-0.85312595, -0.52601772]]

    A_j = xnp.array(A)
    alpha_j = xnp.array(alpha)
    beta_j = xnp.array([beta]).T
    gamma_j = xnp.array([gamma]).T
    B = Tridiagonal(alpha=alpha_j, beta=beta_j, gamma=gamma_j)
    X_j = xnp.array(X)
    rel_error = relative_error(A_j @ X_j, B @ X_j)
    assert rel_error < 1e-6


@parametrize([torch_fns, jax_fns])
def test_adjoint_property(xnp):
    A = xnp.array([[1 + 1j, 2 - 2j, 7 - 3j], [3 + 1j, 4 - 1j, 5 + 2j]])
    B = LinearOperator(shape=A.shape, matmat=lambda x: A @ x, dtype=A.dtype)
    X = xnp.array([1. + 1j, 4. - 2j, 2.5 + 1j, -.1 - 1j, -3. + 1j, -7. - 3j]).reshape(2, 3)
    rel_error = xnp.norm(relative_error(xnp.conj(A).T @ X, B.H @ X))
    assert rel_error < _tol


@parametrize([torch_fns, jax_fns])
def test_transpose_property(xnp):
    A = xnp.array([[1., 2., 7], [3., 4., 5]])
    B = LinearOperator(shape=A.shape, matmat=lambda x: A @ x, dtype=A.dtype)

    X = xnp.array([1., 4., 2.5, -.1, -3., -7., -2., -5., 1.5]).reshape(3, 3)[:2, :]
    rel_error = relative_error(A.T @ X, B.T @ X)

    assert rel_error < _tol, f"JAX transpose relative error: {rel_error}"


@parametrize([torch_fns, jax_fns])
def test_vjp_transpose(xnp):
    A = xnp.array([[1., 2., 7], [3., 4., 5]])
    B = LinearOperator(shape=A.shape, matmat=lambda x: A @ x, dtype=A.dtype)
    X = xnp.array([1., 4., 2.5, -.1, -3., -7., -2., -5., 1.5]).reshape(3, 3)[:2, :2]
    rel_error = relative_error(X @ A, X @ B)
    assert rel_error < _tol, f"VJP transpose relative error: {rel_error}"


@parametrize([torch_fns, jax_fns])
def test_kronecker(xnp):
    A = xnp.array([[1., 2., 7], [3., 4., 5]])
    B = xnp.array([[-1, 4.], [3., 3.], [2., -1.], [2., 1.]])
    actual_soln = xnp.kron(A, B)
    approx = kron(A, B)
    approx_dense = approx.to_dense()
    assert approx_dense.shape == (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])
    rel_error = relative_error(approx_dense, actual_soln)
    assert rel_error < _tol, f"to_dense, relative error is {rel_error}"
    rel_error2 = relative_error(approx @ xnp.eye(approx.shape[-1]), actual_soln)
    assert rel_error2 < _tol, f"matmul, relative error is {rel_error2}"


@parametrize([torch_fns, jax_fns])
def test_matmul(xnp):
    A = xnp.array([1., 4., 2.5, -.1, -3., -7.]).reshape(3, 2)
    B = xnp.array([6., 2., 4., -20.]).reshape(2, 2)
    AB = A @ B
    AB2 = lazify(A) @ B
    AB3 = A @ lazify(B)
    AB4 = (lazify(A) @ lazify(B)).to_dense()
    assert relative_error(AB, AB2) < _tol, "Matmul fails"
    assert relative_error(AB, AB3) < _tol, "rMatmul fails"
    assert relative_error(AB, AB4) < _tol, "Matmul fails"


@parametrize([torch_fns, jax_fns])
def test_scalarmul(xnp):
    A = xnp.array([1, 4, 2.5, -.1, -3, -7]).reshape(3, 2)
    c = -2.2
    A2 = (c * lazify(A)).to_dense()
    A3 = (lazify(A) * c).to_dense()
    assert relative_error(c * A, A2) < _tol, f"Scalar mul fails with {A2}"
    assert relative_error(c * A, A3) < _tol, f"Scalar rmul fails with {A3}"


@parametrize([torch_fns, jax_fns])
def test_sum(xnp):
    A = xnp.array(
        [-12, 3., 54, 2, 31, 65, 7, 3, 0, 3, 4, 5, 67, 7, 23, 4234, 6234, 4, 6, 7, 3, 2, 1, 6])
    As = list(A.reshape(4, 2, 3))
    Asum = sum(As)
    Asum2 = sum(lazify(Ai) for Ai in As).to_dense()
    assert relative_error(Asum, Asum2) < _tol, "Sum fails"
