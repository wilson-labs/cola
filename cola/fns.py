""" Functional interface. Different functions for combining operators and convenience functions.
Like with linalg, these functions have dispatch rules and should be used in favor of the
LinearOperator constructors when possible. """

from typing import Any, List, Union

from plum import dispatch

import cola
from cola.ops import (
    Adjoint,
    Array,
    BlockDiag,
    Dense,
    Diagonal,
    Identity,
    Kronecker,
    KronSum,
    LinearOperator,
    Product,
    ScalarMul,
    Sparse,
    Sum,
    Transpose,
    Triangular,
)
from cola.utils import export

Scalar = Array


@export
def lazify(A: Union[LinearOperator, Array]):
    """ Convert an array to a linear operator if it is not already one. """
    if isinstance(A, LinearOperator):
        return A
    else:
        return Dense(A)


@export
def densify(A: Union[LinearOperator, Array]):
    """ Convert a linear operator to a dense array if it is not already one. """
    if isinstance(A, LinearOperator):
        return A.to_dense()
    else:
        return A


@export
def no_dispatch(A: LinearOperator):
    """
    This function allows the user to circumvent dispatch functionality that might be triggered
    based on the type of the operator.
    For example, if we have `D = A @ B + a * C` where `A`, `B`, `C` are linear operators and
    `a` is a scalar then to avoid triggering sum, product and scalar product dispatch rules
    we can wrap `D` as `D = cola.no_dispatch(D)`.
    """
    Op = LinearOperator(dtype=A.dtype, shape=A.shape, matmat=A._matmat)
    return Op


@dispatch
def dot(A: LinearOperator, B: LinearOperator):
    return Product(A, B)


@dispatch
def dot(A: Product, B: LinearOperator):
    return Product(*(A.Ms + (B, )))


@dispatch
def dot(A: LinearOperator, B: Product):
    return Product(*((A, ) + B.Ms))


@dispatch
def dot(A: Product, B: Product):
    return Product(*(A.Ms + B.Ms))


@dispatch
def dot(A: Any, B: Identity):
    return A


@dispatch
def dot(A: Identity, B: Any):
    return B


@dispatch
def add(A: Any, B: Any):
    return add(lazify(A), lazify(B))


@dispatch
def add(A: LinearOperator, B: LinearOperator):
    return Sum(A, B)


@dispatch
def add(A: Sum, B: LinearOperator):
    return Sum(*(A.Ms + (B, )))


@dispatch
def add(A: LinearOperator, B: Sum):
    return Sum(*((A, ) + B.Ms))


@dispatch
def add(A: Sum, B: Sum):
    return Sum(*(A.Ms + B.Ms))


@dispatch
def mul(A: LinearOperator, c: Scalar):
    S = ScalarMul(c, (A.shape[-2], A.shape[-2]), A.dtype, A.device)
    return Product(*[S, A])


@dispatch
def mul(A: ScalarMul, c: Scalar):
    return ScalarMul(A.c * c, A.shape, A.dtype, A.device)


@dispatch
def mul(c: Scalar, A: ScalarMul):
    return ScalarMul(A.c * c, A.shape, A.dtype, A.device)


@dispatch
def mul(A: ScalarMul, B: ScalarMul):
    assert A.device == B.device, "there is a device mismatch"
    return ScalarMul(A.c * B.c, A.shape, A.dtype, A.device)


@dispatch
def transpose(A: LinearOperator):
    return Transpose(A)


@dispatch
def transpose(A: Transpose):
    return A.A


@dispatch
def transpose(A: Dense):
    return Dense(A.A.T)


@dispatch(cond=lambda A: A.isa(cola.SelfAdjoint))
def transpose(A: LinearOperator):
    # dangerous, TODO: fix when A is complex or unify transpose and adjoint
    return A


@dispatch
def transpose(A: Triangular):
    return Triangular(A.A.T, lower=not A.lower)


@dispatch
def transpose(A: Sparse):
    return Sparse(A.data, A.col_indices, A.row_indices, shape=(A.shape[1], A.shape[0]))


@dispatch
def adjoint(A: LinearOperator):
    return Adjoint(A)


@dispatch(cond=lambda A: A.isa(cola.SelfAdjoint))
def adjoint(A: LinearOperator):
    return A


@dispatch
def adjoint(A: Adjoint):
    return A.A


@dispatch
def adjoint(A: Dense):
    return Dense(A.A.T.conj())


@dispatch
def adjoint(A: Triangular):
    return Triangular(A.A.T.conj(), lower=not A.lower)


@dispatch
@export
def kron(A: Any, B: Any):
    """ Kronecker product of two linear operators. """
    return kron(lazify(A), lazify(B))


@dispatch
def kron(A: LinearOperator, B: LinearOperator):
    return Kronecker(*[A, B])


@dispatch
def kron(A: Diagonal, B: Diagonal):
    diag = (A.diag[:, None] * B.diag[None, :]).reshape(-1)
    return Diagonal(diag)


@dispatch
def kron(A: Kronecker, B: LinearOperator):
    return Kronecker(*(A.Ms + (B, )))


@dispatch
def kron(A: LinearOperator, B: Kronecker):
    return Kronecker(*((A, ) + B.Ms))


@dispatch
@export
def kronsum(A: Any, B: Any):
    return kronsum(lazify(A), lazify(B))


@dispatch
def kronsum(A: LinearOperator, B: LinearOperator):
    return KronSum(*[A, B])


@dispatch
def kronsum(A: KronSum, B: LinearOperator):
    return KronSum(*(A.Ms + (B, )))


@dispatch
def kronsum(A: LinearOperator, B: KronSum):
    return KronSum(*((A, ) + B.Ms))


@export
def block_diag(*ops: List[LinearOperator]):
    """ Construct a block diagonal operator from a list of ops. """
    return BlockDiag(*ops)


def concatenate(ops: List[LinearOperator], axis=0):
    raise NotImplementedError
