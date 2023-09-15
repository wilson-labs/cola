from cola.fns import lazify
from cola.ops import LinearOperator, Tridiagonal, Diagonal, Identity
from cola.ops import KronSum, Product
from cola.ops import Triangular, Kronecker, Permutation
from cola.ops import Dense, BlockDiag, Jacobian, Hessian, Sparse
from cola.annotations import SelfAdjoint
from cola.annotations import PSD
from cola.utils_test import get_xnp
from functools import reduce
import cola

op_names: set[str] = {
    'psd_big',  # skipped by default
    'psd_blockdiag',
    'psd_diagonal',
    'psd_identity',
    'psd_prod',
    'psd_scalarmul',
    'psd_kron',
    'selfadj_hessian',
    'selfadj_tridiagonal',
    # 'square_big',  # skipped by default
    'square_blockdiag',
    'square_dense',
    # 'square_jacobian',
    'square_kronecker',
    'square_kronsum',
    'square_lowertriangular',
    'square_permutation',
    'square_product',
    # 'square_sparse',
    'square_tridiagonal',
    'diagonal_plus_uv',
    'uv_plus_diagonal',
}


def get_test_operator(backend: str, precision: str, op_name: str, device: str = 'cpu') -> LinearOperator:
    xnp = get_xnp(backend)
    dtype = getattr(xnp, precision)
    if backend == 'torch':
        import torch
        device = torch.device(device)
    else:
        from jax import numpy as jnp
        if dtype == jnp.float64:
            from jax.config import config
            config.update("jax_enable_x64", True)

        device = None

    # Define the operator
    op = None
    match op_name.split('_', 1):
        case ('psd', 'diagonal'):
            op = Diagonal(xnp.array([.1, .5, .22, 8.], dtype=dtype, device=device))

        case ('psd', ('identity' | 'scalarmul') as sub_op_name):
            shape = (3, 3)
            op = Identity(shape, dtype=dtype)
            if sub_op_name == 'scalarmul':
                op = 2 * op

        case ('psd', ('big' | 'blockdiag' | 'prod') as sub_op_name):
            M1 = Dense(xnp.array([[6., 2], [2, 4]], dtype=dtype, device=device))
            M2 = Dense(xnp.array([[7, 6], [6, 8]], dtype=dtype, device=device))
            match sub_op_name:
                case 'big':
                    dtype2 = (xnp.array([1.], dtype=dtype, device=device) + 1j).dtype
                    Id = Identity((15, 15), dtype=dtype2)
                    big_psd = reduce(cola.kron, [M1, M2, M2, Id])
                    op = big_psd + 0.04 * cola.ops.I_like(big_psd)
                case 'blockdiag':
                    op = BlockDiag(M1, M2, multiplicities=[2, 3])
                case 'prod':
                    op = M1 @ M1.T
    
        case ('psd', 'kron'):
            M1 = Dense(xnp.array([[6., 2], [2, 4]], dtype=dtype, device=device))
            M2 = Dense(xnp.array([[7, 6], [6, 8]], dtype=dtype, device=device))
            op = Kronecker(M1, M2)

        case (('selfadj' | 'square') as op_prop, 'tridiagonal'):
            alpha = xnp.array([1, 2, 3], dtype=dtype, device=device)[:2]
            beta = xnp.array([4, 5, 6], dtype=dtype, device=device)
            gamma = xnp.array([7, 8, 9], dtype=dtype, device=device)[:2]
            match op_prop:
                case 'selfadj':
                    op = Tridiagonal(alpha, beta, alpha)
                case 'square':
                    op = Tridiagonal(alpha, beta, gamma)

        case ('selfadj', 'hessian'):

            def f2(x):
                return (x[1] - .1)**3 + xnp.cos(x[2]) + (x[0] + .2)**2

            x = xnp.array([1., 2., 3.], dtype=dtype, device=device)
            op = Hessian(f2, x)

        case ('square', 'jacobian'):

            def f1(x):
                return xnp.array([x[0]**2, x[1]**3, xnp.sin(x[2])], dtype=dtype)

            x = xnp.array([1, 2, 3], dtype=dtype, device=device)
            op = Jacobian(f1, x)

        case ('square', 'permutation'):
            op = Permutation(xnp.array([1, 0, 2, 3, 6, 5, 4], dtype=xnp.int32, device=device))

        case ('square', sub_op_name):
            M1 = xnp.array([[1, 0], [3, 4]], dtype=dtype, device=device)
            M2 = xnp.array([[5, 6], [7, 8]], dtype=dtype, device=device)
            match sub_op_name:
                case 'big':
                    dtype2 = (xnp.array([1.], device=device, dtype=dtype) + 1j).dtype
                    M1 = [[1, 0, 0], [3, 4 + .1j, 2j], [0, 0, .1]]
                    M1 = Dense(xnp.array(M1, dtype=dtype2, device=device))
                    M2 = [[5, 2, 0], [3., 8, 0], [0, 0, -.5]]
                    M2 = Dense(xnp.array(M2, dtype=dtype2, device=device))
                    Id = Identity((10, 10), dtype=dtype2)
                    big = reduce(cola.kron, [M1, M2, M1 @ M1, M2, Id])
                    op = big + 0.5 * cola.ops.I_like(big)
                case 'blockdiag':
                    op = BlockDiag(M1, M2, multiplicities=[2, 3])
                case 'dense':
                    op = Dense(M1)
                case 'kronecker':
                    op = Kronecker(lazify(M1), lazify(M2))
                case 'kronsum':
                    op = KronSum(lazify(M1), lazify(M2))
                case 'lowertriangular':
                    op = Triangular(M1)
                case 'product':
                    op = Product(lazify(M1), lazify(M2))

        # case ('square', 'sparse'):
        #     data = xnp.array([1, 2, 3, 4, 5, 6], dtype=dtype, device=device)
        #     indices = xnp.array([0, 2, 1, 0, 2, 1], dtype=dtype, device=device)
        #     indptr = xnp.array([0, 2, 4, 6], dtype=dtype, device=device)
        #     shape = (3, 3)
        #     sparse = Sparse(data, indices, indptr, shape)

        case ('diagonal', 'plus_uv'):
            D = Diagonal(xnp.array([.1, .5, .22, 8.], dtype=dtype, device=device))
            U = Dense(xnp.array([[6., 2], [2, 4], [1.2, 2], [5, 10]], dtype=dtype, device=device))
            V = Dense(xnp.array([[7, 6, 6, 8.7], [3., .2, 13, 4]], dtype=dtype, device=device))
            op = D + U @ V

        case ('uv', 'plus_diagonal'):
            D = Diagonal(xnp.array([.1, .5, .22, 8.], dtype=dtype, device=device))
            U = Dense(xnp.array([[6., 2], [2, 4], [1.2, 2], [5, 10]], dtype=dtype, device=device))
            V = Dense(xnp.array([[7, 6, 6, 8.7], [3., .2, 13, 4]], dtype=dtype, device=device))
            op = U @ V + D

    # Check to sure that we hit a case statement
    if op is None:
        raise ValueError(op_name)

    # Maybe wrap as a PSD or SelfAdjoint linear operator
    match op_name.split('_')[0]:
        case 'psd':
            op = PSD(op)
        case 'selfadj':
            op = SelfAdjoint(op)
        case _:
            pass

    return op


def get_test_operators(backend: str, precision: str, device: str = 'cpu') -> list[LinearOperator]:
    return [
        get_test_operator(backend=backend, precision=precision, device=device, op_name=op_name) for op_name in op_names
    ]


__all__ = ['get_test_operator', 'get_test_operators', 'op_names']
