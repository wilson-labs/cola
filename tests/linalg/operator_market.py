from cola import jax_fns
from cola import torch_fns
from cola.fns import kron, lazify

from cola.ops import Tridiagonal, Diagonal, Identity, I_like
from cola.ops import KronSum, ScalarMul, Product, Sliced
from cola.ops import Sparse, LowerTriangular, Kronecker, Permutation
from cola.ops import Dense, BlockDiag, Jacobian, Hessian

from cola.annotations import SelfAdjoint
from cola.annotations import PSD
from cola.ops import LinearOperator

xnp = jax_fns


def get_test_operators(xnp, dtype):
    alpha = xnp.array([1, 2, 3],dtype=dtype)[:2]
    beta = xnp.array([4, 5, 6],dtype=dtype)
    gamma = xnp.array([7, 8, 9],dtype=dtype)[:2]
    tridiagonal = Tridiagonal(alpha, beta, gamma)


    shape = (3, 3)
    dtype = xnp.float64
    identity = Identity(shape, dtype)

    M1 = xnp.array([[1, 2], [3, 4]],dtype=dtype)
    M2 = xnp.array([[5, 6], [7, 8]],dtype=dtype)
    kronsum = KronSum(lazify(M1), lazify(M2))

    scalarmul = 2.*identity

    product = Product(M1, M2)

    sliced = Sliced(M1, (slice(0,1), slice(0,2)))

    data = xnp.array([1, 2, 3, 4, 5, 6],dtype=dtype)
    indices = xnp.array([0, 2, 1, 0, 2, 1])
    indptr = xnp.array([0, 2, 4, 6])
    shape = (3, 3)
    #sparse = Sparse(data, indices, indptr, shape)

    lowertriangular = LowerTriangular(M1)

    kronecker = Kronecker(M1, M2)

    permutation = Permutation(xnp.array([1, 0, 2,3,6,5,4]))

    dense = Dense(M1)

    M1 = xnp.array([[1, 2], [3, 4]],dtype=dtype)
    M2 = xnp.array([[5, 6], [7, 8]],dtype=dtype)
    blockdiag = BlockDiag(M1, M2, multiplicities=[2, 3])

    # Jacobian
    def f1(x):
        return xnp.array([x[0]**2, x[1]**3, xnp.sin(x[2])])
    x = xnp.array([1, 2, 3],dtype=dtype)
    jacobian = Jacobian(f1, x)

    # Hessian
    def f2(x):
        return (x[1]-.1)**3 + xnp.cos(x[2])+(x[0]+.2)**2
    x = xnp.array([1, 2, 3],dtype=dtype)
    hessian = Hessian(f2, x)


    # PSD
    psd_ops = [Diagonal(xnp.array([.1,.5,.22,8.],dtype=dtype)), identity, scalarmul]
    symmetric_ops = [hessian ,Tridiagonal(alpha,beta,alpha)]
    square_ops = [kronsum, tridiagonal, dense, \
        kronecker, blockdiag, product, lowertriangular, jacobian]

    return [PSD(op) for op in psd_ops]+[SelfAdjoint(op) for op in symmetric_ops]+square_ops
    # # Non square
    # # None of the operators in the provided documentation generate non-square matrices
    # non_square_ops = []