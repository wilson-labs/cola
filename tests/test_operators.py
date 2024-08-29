import numpy as np
import pytest
from linalg.operator_market import get_test_operator, op_names

from cola.backends import all_backends, tracing_backends
from cola.fns import kron, lazify
from cola.linalg.decompositions.arnoldi import get_householder_vec
from cola.ops import (
    Diagonal,
    Hessian,
    Householder,
    I_like,
    Identity,
    Jacobian,
    Kernel,
    KronSum,
    LinearOperator,
    Product,
    ScalarMul,
    Sliced,
    Sparse,
    Sum,
    Tridiagonal,
)
from cola.utils.test_utils import get_xnp, parametrize, relative_error

_tol = 1e-6


@parametrize(tracing_backends)
def test_Hessian(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    P = 27
    cons = xnp.array([idx for idx in range(P)], dtype=dtype, device=None)
    x = xnp.ones(shape=(P, ), device=None, dtype=dtype)

    def fn(z):
        out = cons * z**2.
        return xnp.sum(out)

    H = Hessian(fn, x)
    approx = H.to_dense()
    soln = 2 * xnp.diag(cons)
    rel_error = relative_error(approx, soln, xnp=xnp)
    assert rel_error < _tol


_exclude = (slice(None), slice(None), ['square_fft'])


@parametrize(tracing_backends, ['float32'], op_names).excluding[_exclude]
def test_ops_to(backend, precision, op_name):
    Op = get_test_operator(backend, precision, op_name)
    xnp = get_xnp(backend)
    dtype = xnp.float32
    device_cpu = xnp.get_default_device()
    dev_type = device_cpu.type if backend == "torch" else device_cpu.platform
    assert dev_type == "cpu"
    # FIXME: xnp.is_cuda_available failing for JAX
    device_gpu = xnp.device("cuda:0") if xnp.is_cuda_available() else device_cpu
    # device_gpu = xnp.device("cuda:0")

    assert Op.device == device_cpu
    ones = xnp.ones((Op.shape[0], ), dtype=dtype, device=device_cpu)
    aux = Op @ ones
    assert xnp.get_array_device(aux) == device_cpu

    Op = Op.to(device_gpu)
    assert Op.device == device_gpu
    ones = xnp.ones((Op.shape[0], ), dtype=dtype, device=device_gpu)
    aux = Op @ ones
    assert xnp.get_array_device(aux) == device_gpu


@parametrize(tracing_backends)
def test_find_device(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    Aop = Diagonal(xnp.array([0.1, -0.2], dtype=dtype, device=None))
    diag2 = xnp.array([3., -7.], dtype=dtype, device=None)
    Bop = Diagonal(diag=diag2)
    Cop = KronSum(Aop, Bop)
    Dop = Diagonal(xnp.array([1., 2., 3., 4], dtype=dtype, device=None))
    Fop = Sum(Cop, Dop)

    Ops = [Aop, Bop, Cop, Dop, Fop]
    for Op in Ops:
        print(Op.device)
        print(xnp.get_default_device())
        print(Op.device == xnp.get_default_device())
        assert Op.device == xnp.get_default_device()


@parametrize(['torch'])
def test_kernel(backend):
    N, D, B = 7, 3, 5
    xnp = get_xnp(backend)
    dtype = xnp.float64
    device = None
    x = xnp.randn(N, D, dtype=dtype, device=device)
    V = xnp.ones(shape=(N, B), dtype=dtype, device=device)

    def f(x1, x2):
        out = x1[:, None, :] - x2[None, :, :]
        out = xnp.exp(-xnp.norm(out, axis=-1))
        return out

    for (bs1, bs2) in [(2, 3), (N, 1), (1, N), (3, N), (1, 1)]:
        Ker = Kernel(x, x, fn=f, block_size1=bs1, block_size2=bs2)
        approx = Ker @ V
        soln = f(x, x) @ V
        diff = xnp.norm(approx - soln)
        print(f"Abs diff: {diff:1.5e}")
        assert diff < 1e-10


@parametrize(tracing_backends)
def test_sparse(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    A = [[0., 1., 0., 2.], [0., 0., 0., 3.], [4., 5., 6., 0.]]
    A = xnp.array(A, dtype=dtype, device=None)
    x1 = xnp.array([1., 0., -1., 1.], dtype=dtype, device=None)
    soln1 = xnp.array([2., 3., -2.], dtype=dtype, device=None)
    x2 = xnp.array([1., 2., 3.], dtype=dtype, device=None)
    soln2 = xnp.array([12., 16., 18., 8.], dtype=dtype, device=None)

    data = xnp.array([4., 5., 6., 1., 2., 3.], dtype=dtype, device=None)
    if backend == "torch":
        data.requires_grad = True
    row_indices = xnp.array([2., 2., 2., 0., 0., 1.], dtype=xnp.int64, device=None)
    col_indices = xnp.array([0., 1., 2., 1., 3., 3.], dtype=xnp.int64, device=None)
    Aop = Sparse(data, row_indices, col_indices, shape=(3, 4))

    rel_error = relative_error(Aop @ x1, soln1)
    assert rel_error < _tol

    rel_error = relative_error(x2 @ Aop, soln2)
    assert rel_error < _tol

    rel_error = relative_error(Aop.to_dense(), A)
    assert rel_error < _tol

    rel_error = relative_error(Aop.T.to_dense(), A.T)
    assert rel_error < _tol


@parametrize(['jax'])
def test_jacobian(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32

    def f1(x):
        return xnp.array([x[0]**2, x[1]**3, xnp.sin(x[2])], dtype=dtype, device=None)

    x = xnp.array([1, 2, 3], dtype=dtype, device=None)
    A = Jacobian(f1, x)
    assert xnp.norm(A @ x) is not None


@parametrize(all_backends)
def test_householder(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    x = xnp.array([0.29466099, 0.71853315, -0.06172857, -0.0432496, 0.44698924], dtype=dtype, device=None)
    w = x / xnp.norm(x)
    R = Householder(w[:, None])
    rel_error = relative_error(-x, R @ x)
    assert rel_error < _tol

    for idx in [0, 2]:
        vec, beta = get_householder_vec(x, idx=idx, xnp=xnp)
        R = Householder(vec[:, None], beta=beta)
        approx = xnp.canonical(idx, x.shape, dtype, device=None) * xnp.norm(x[idx:])
        approx = xnp.update_array(approx, x[:idx], slice(None, idx, None))
        rel_error = relative_error(approx, R @ x)
        assert rel_error < _tol


@parametrize(all_backends)
def test_unflatten(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    diag1 = xnp.array([0.1, -0.2], dtype=dtype, device=None)
    Aop = Diagonal(diag1)
    diag2 = xnp.array([3., -7.], dtype=dtype, device=None)
    Bop = Diagonal(diag2)
    Cop = KronSum(Aop, Bop)
    flattened, un_fn = Cop.flatten()
    Cop_approx = un_fn(flattened)
    rel_error = relative_error(Cop.to_dense(), Cop_approx.to_dense())
    assert rel_error < _tol
    diag3 = xnp.array([1., 2., 3., 4], dtype=dtype, device=None)
    Dop = Diagonal(diag3)
    Fop = Sum(Cop, Dop)
    flattened, un_fn = Fop.flatten()
    Fop_approx = un_fn(flattened)
    rel_error = relative_error(Fop.to_dense(), Fop_approx.to_dense())
    assert rel_error < _tol


@parametrize(all_backends)
def test_flatten(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    diag1 = xnp.array([0.1, -0.2], dtype=dtype, device=None)
    Aop = Diagonal(diag1)
    diag2 = xnp.array([3., -7.], dtype=dtype, device=None)
    Bop = Diagonal(diag2)
    Cop = KronSum(Aop, Bop)
    diag3 = xnp.array([1., 2., 3., 4], dtype=dtype, device=None)
    const = 3.
    Dop = Diagonal(diag3)
    Fop = Sum(Cop, Dop)
    Fop *= const

    flattened, _ = Fop.flatten()
    soln = [xnp.array(const, dtype=dtype, device=None), diag1, diag2, diag3]
    for par1, par2 in zip(flattened, soln):
        assert relative_error(par1, par2) < 1e-12


@parametrize(all_backends)
def test_get_item(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    np.random.seed(seed=48)
    A = np.random.normal(size=(5, 4))
    A = xnp.array(A, dtype=dtype, device=None)
    B = lazify(A)
    approx = B[0, 2]
    soln = A[0, 2]
    rel_error = relative_error(soln, approx)
    assert rel_error < _tol

    approx = B[[0, 2], [3, -1]]
    soln = A[[0, 2], [3, -1]]
    rel_error = relative_error(soln, approx)
    assert rel_error < _tol


@parametrize(all_backends)
def test_sliced(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    A = [[-1.0165, -1.9161, 0.8602, -0.4597], [1.1470, 0.9879, -1.0831, 1.2768], [1.0912, 0.6360, -0.8962, 0.7849],
         [-0.4359, 0.5612, 1.4560, 0.0682]]
    A = xnp.array(A, dtype=dtype, device=None)
    slices = (slice(1, 3, None), slice(0, 3, None))
    rhs = [[-0.3018, 0.0266, -2.4584], [0.3041, -1.2548, 0.0115], [-0.4590, -3.0145, 0.5659]]
    rhs = xnp.array(rhs, dtype=dtype, device=None)
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


@parametrize(all_backends)
def test_product_op(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    A = [[-1.0165, -1.9161, 0.8602, -0.4597], [1.1470, 0.9879, -1.0831, 1.2768], [1.0912, 0.6360, -0.8962, 0.7849],
         [-0.4359, 0.5612, 1.4560, 0.0682]]
    A = xnp.array(A, dtype=dtype, device=None)
    B = [[-1.3213, 0.8053, -0.4334, 0.6765], [0.5056, 0.5160, 0.6120, -0.8747], [-1.8671, -0.1413, 1.6811, -0.5727],
         [-1.7080, 0.0548, -0.1522, -0.2481]]
    D = [[0.3105, 0.2883, 0.4809, -0.0063], [0.6082, 1.0688, -1.6521, -1.4216]]
    D = xnp.array(D, dtype=dtype, device=None)
    B = xnp.array(B, dtype=dtype, device=None)
    rhs = [[1., 3.], [4., 5.], [6., -8.], [-1., 0.2]]
    rhs = xnp.array(rhs, dtype=dtype, device=None)
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


@parametrize(all_backends)
def test_sum_and_scalar_op(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    A = [[-1.0165, -1.9161, 0.8602, -0.4597], [1.1470, 0.9879, -1.0831, 1.2768], [1.0912, 0.6360, -0.8962, 0.7849],
         [-0.4359, 0.5612, 1.4560, 0.0682]]
    A = xnp.array(A, dtype=dtype, device=None)
    B = [[-1.3213, 0.8053, -0.4334, 0.6765], [0.5056, 0.5160, 0.6120, -0.8747], [-1.8671, -0.1413, 1.6811, -0.5727],
         [-1.7080, 0.0548, -0.1522, -0.2481]]
    B = xnp.array(B, dtype=dtype, device=None)
    rhs = [[1., 3.], [4., 5.], [6., -8.], [-1., 0.2]]
    rhs = xnp.array(rhs, dtype=dtype, device=None)
    soln = (A + B) @ rhs
    C = Sum(lazify(A), lazify(B))
    approx = C @ rhs
    D = ScalarMul(-5., shape=C.shape, dtype=dtype)

    print(str(C))
    print(str(D))
    rel_error = relative_error(soln, approx)
    assert rel_error < _tol
    rel_error = relative_error((-5.) * (A + B), D @ (C.to_dense()))
    assert rel_error < _tol


@parametrize(all_backends)
def test_kronsum(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    diag = xnp.array([0.1, -0.2], dtype=dtype, device=None)
    Aop = Diagonal(diag=diag)
    A = xnp.diag(diag)
    B = xnp.eye(2, 2, dtype=dtype, device=None)
    Id = xnp.eye(A.shape[0], A.shape[1], dtype=dtype, device=None)
    Bop = lazify(B)
    rhs = [[1., 3.], [4., 5.], [6., -8.], [-1., 0.2]]
    rhs = xnp.array(rhs, dtype=dtype, device=None)
    Cop = KronSum(Aop, Bop)
    approx = Cop @ rhs
    kron_dense = xnp.kron(A, Id) + xnp.kron(Id, B)
    soln = kron_dense @ rhs

    print(str(Cop))

    rel_error = relative_error(soln, approx)
    assert rel_error < _tol
    rel_error = relative_error(kron_dense, Cop.to_dense())
    assert rel_error < _tol


@parametrize(all_backends)
def test_identity(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    Id = xnp.eye(3, 3, dtype=dtype, device=None)
    rhs = [[1., 3.], [4., 5.], [6., -8.]]
    rhs = xnp.array(rhs, dtype=dtype, device=None)
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


@parametrize(all_backends)
def test_diagonal(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    diag = xnp.array([0.1, 0.2, 3., 4.], dtype=dtype, device=None)
    A = xnp.diag(diag)
    rhs = xnp.ones(shape=(A.shape[0], 6), dtype=dtype, device=None)
    soln = A @ rhs
    B = Diagonal(diag=diag)
    approx = B @ rhs

    rel_error = relative_error(soln, approx)
    assert rel_error < _tol


@parametrize(all_backends)
def test_diagonal_variants(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    value = xnp.array(0.1, dtype=dtype, device=None)
    diag = xnp.array([0.1, 0.1, 0.1, 0.1], dtype=dtype, device=None)
    A = xnp.diag(diag)
    rhs = xnp.ones(shape=(A.shape[0], 6), dtype=dtype, device=None)
    soln = A @ rhs
    B = ScalarMul(value, A.shape, dtype=dtype)
    approx = B @ rhs

    rel_error = relative_error(soln, approx)
    assert rel_error < _tol

    diag = xnp.array([2., 4., 2., 4.], dtype=dtype, device=None)
    diag1 = Diagonal(xnp.array([1., 1.], dtype=dtype, device=None))
    diag2 = Diagonal(xnp.array([2., 4.], dtype=dtype, device=None))
    soln = xnp.diag(diag)
    approx = kron(diag1, diag2)

    rel_error = relative_error(soln, approx.to_dense())
    assert rel_error < _tol


@parametrize(all_backends)
def test_tridiagonal(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    alpha = [-0.5, 4., 5.]
    beta = [0.1, 0.2, -0.1, 0.4]
    gamma = [-1.0, 2., 3.]
    A = [[beta[0], gamma[0], 0, 0], [alpha[0], beta[1], gamma[1], 0], [0., alpha[1], beta[2], gamma[2]],
         [0., 0., alpha[2], beta[3]]]
    X = [[-0.23742934, -1.28097845], [-0.4654211, -0.42619589], [-1.78554193, -1.01728456], [-0.85312595, -0.52601772]]

    A_j = xnp.array(A, dtype=dtype, device=None)
    alpha_j = xnp.array(alpha, dtype=dtype, device=None)
    beta_j = xnp.array([beta], dtype=dtype, device=None).T
    gamma_j = xnp.array([gamma], dtype=dtype, device=None).T
    B = Tridiagonal(alpha=alpha_j, beta=beta_j, gamma=gamma_j)
    X_j = xnp.array(X, dtype=dtype, device=None)
    rel_error = relative_error(A_j @ X_j, B @ X_j)
    assert rel_error < 1e-6


@parametrize(tracing_backends)
def test_adjoint_property(backend):
    xnp = get_xnp(backend)
    dtype = xnp.complex64
    A = xnp.array([[1 + 1j, 2 - 2j, 7 - 3j], [3 + 1j, 4 - 1j, 5 + 2j]], dtype=dtype, device=None)
    B = LinearOperator(shape=A.shape, matmat=lambda x: A @ x, dtype=A.dtype)
    X = xnp.array([1. + 1j, 4. - 2j, 2.5 + 1j, -.1 - 1j, -3. + 1j, -7. - 3j], dtype=dtype, device=None)
    X = X.reshape(2, 3)
    rel_error = relative_error(xnp.conj(A).T @ X, B.H @ X)
    assert rel_error < _tol


@parametrize(tracing_backends)
def test_transpose_property(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    A = xnp.array([[1., 2., 7], [3., 4., 5]], dtype=dtype, device=None)
    B = LinearOperator(shape=A.shape, matmat=lambda x: A @ x, dtype=A.dtype)

    X = xnp.array([1., 4., 2.5, -.1, -3., -7., -2., -5., 1.5], dtype=dtype, device=None)
    X = X.reshape(3, 3)[:2, :]
    rel_error = relative_error(A.T @ X, B.T @ X)

    assert rel_error < _tol, f"JAX transpose relative error: {rel_error}"


@parametrize(tracing_backends)
def test_vjp_transpose(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    A = xnp.array([[1., 2., 7], [3., 4., 5]], dtype=dtype, device=None)
    B = LinearOperator(shape=A.shape, matmat=lambda x: A @ x, dtype=A.dtype)
    X = xnp.array([1., 4., 2.5, -.1, -3., -7., -2., -5., 1.5], dtype=dtype, device=None)
    X = X.reshape(3, 3)[:2, :2]
    rel_error = relative_error(X @ A, X @ B)
    assert rel_error < _tol, f"VJP transpose relative error: {rel_error}"


@parametrize(all_backends)
def test_kronecker(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    A = xnp.array([[1., 2., 7], [3., 4., 5]], dtype=dtype, device=None)
    B = xnp.array([[-1, 4.], [3., 3.], [2., -1.], [2., 1.]], dtype=dtype, device=None)
    actual_soln = xnp.kron(A, B)
    approx = kron(A, B)
    approx_dense = approx.to_dense()
    assert approx_dense.shape == (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])
    rel_error = relative_error(approx_dense, actual_soln)
    assert rel_error < _tol, f"to_dense, relative error is {rel_error}"
    Id = xnp.eye(approx.shape[-1], approx.shape[-1], dtype=dtype, device=None)
    rel_error2 = relative_error(approx @ Id, actual_soln)
    assert rel_error2 < _tol, f"matmul, relative error is {rel_error2}"


@parametrize(all_backends)
def test_matmul(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    A = xnp.array([1., 4., 2.5, -.1, -3., -7.], dtype=dtype, device=None).reshape(3, 2)
    B = xnp.array([6., 2., 4., -20.], dtype=dtype, device=None).reshape(2, 2)
    AB = A @ B
    AB2 = lazify(A) @ B
    AB3 = A @ lazify(B)
    AB4 = (lazify(A) @ lazify(B)).to_dense()
    assert relative_error(AB, AB2) < _tol, "Matmul fails"
    assert relative_error(AB, AB3) < _tol, "rMatmul fails"
    assert relative_error(AB, AB4) < _tol, "Matmul fails"


@parametrize(all_backends)
def test_scalarmul(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    A = xnp.array([1, 4, 2.5, -.1, -3, -7], dtype=dtype, device=None)
    A = A.reshape(3, 2)
    c = -2.2
    A2 = (c * lazify(A)).to_dense()
    A3 = (lazify(A) * c).to_dense()
    assert relative_error(c * A, A2) < _tol, f"Scalar mul fails with {A2}"
    assert relative_error(c * A, A3) < _tol, f"Scalar rmul fails with {A3}"


@parametrize(all_backends)
def test_sum(backend):
    xnp = get_xnp(backend)
    dtype = xnp.float32
    A = [-12, 3., 54, 2, 31, 65, 7, 3, 0, 3, 4, 5, 67, 7, 23, 4234, 6234, 4, 6, 7, 3, 2, 1, 6]
    A = xnp.array(A, dtype=dtype, device=None)
    As = list(A.reshape(4, 2, 3))
    Asum = sum(As)
    Asum2 = sum(lazify(Ai) for Ai in As).to_dense()
    assert relative_error(Asum, Asum2) < _tol, "Sum fails"
