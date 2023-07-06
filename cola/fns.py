""" Functional interface. Different functions for combining operators and convenience functions.
Like with linalg, these functions have dispatch rules and should be used in favor of the 
LinearOperator constructors when possible. """

from typing import List, Union, Any
from plum import dispatch
from cola.ops import LinearOperator, Array
from cola.ops import Dense
from cola.ops import Kronecker, Product, KronSum, Sum
from cola.ops import ScalarMul, Transpose, Adjoint, SelfAdjoint
from cola.ops import BlockDiag, Identity, Diagonal, I_like
from cola.utils import export
Scalar = Array

@export
def lazify(A: Union[LinearOperator, Array]) -> LinearOperator:
    """ Convert an array to a linear operator if it is not already one. """
    if isinstance(A, LinearOperator):
        return A
    else:
        return Dense(A)

@export
def densify(A: Union[LinearOperator, Array]) -> Array:
    """ Convert a linear operator to a dense array if it is not already one. """
    if isinstance(A, LinearOperator):
        return A.to_dense()
    else:
        return A


@dispatch
def dot(A: LinearOperator, B: LinearOperator) -> Product:
    return Product(A, B)


# @dispatch
# def dot(A: SelfAdjoint, B: SelfAdjoint) -> SelfAdjoint:
#     return type(A)(Product([A, B]))


@dispatch
def dot(A: Product, B: LinearOperator) -> Product:
    return Product(*(A.Ms + (B, )))


@dispatch
def dot(A: LinearOperator, B: Product) -> Product:
    return Product(*((A, ) + B.Ms))


@dispatch
def add(A: Any, B: Any) -> Sum:
    return add(lazify(A), lazify(B))


@dispatch
def add(A: LinearOperator, B: LinearOperator) -> Sum:
    return Sum(A, B)


@dispatch
def add(A: Sum, B: LinearOperator) -> Sum:
    return Sum(*(A.Ms + (B, )))


@dispatch
def add(A: LinearOperator, B: Sum) -> Sum:
    return Sum(*((A, ) + B.Ms))


@dispatch
def mul(A: LinearOperator, c: Scalar) -> LinearOperator:
    return Product(*[ScalarMul(c, (A.shape[-2], A.shape[-2]), A.dtype), A])


@dispatch
def mul(A: ScalarMul, c: Scalar) -> ScalarMul:
    return ScalarMul(A.c * c, A.shape,A.dtype)


@dispatch
def mul(c: Scalar, A: ScalarMul) -> ScalarMul:
    return ScalarMul(A.c * c, A.shape,A.dtype)


@dispatch
def mul(A: ScalarMul, B: ScalarMul) -> ScalarMul:
    return ScalarMul(A.c * B.c, A.shape,A.dtype)


@dispatch
def transpose(A: LinearOperator):
    return Transpose(A)


# @dispatch
# def transpose(A: Transpose):
#     return A.A


@dispatch
def adjoint(A: LinearOperator):
    return Adjoint(A)


# @dispatch
# def adjoint(A: Adjoint): 
#     return A.A


@dispatch
@export
def kron(A: Any, B: Any) -> Kronecker:
    """ Kronecker product of two linear operators. """
    return kron(lazify(A), lazify(B))


@dispatch
def kron(A: LinearOperator, B: LinearOperator) -> Kronecker:
    return Kronecker(*[A, B])


@dispatch
def kron(A: Diagonal, B: Diagonal) -> Diagonal:
    diag = (A.diag[:, None] * B.diag[None, :]).reshape(-1)
    return Diagonal(diag)


@dispatch
def kron(A: Kronecker, B: LinearOperator) -> Kronecker:
    return Kronecker(*(A.Ms + (B, )))


@dispatch
def kron(A: LinearOperator, B: Kronecker) -> Kronecker:
    return Kronecker(*((A, ) + B.Ms))


# @dispatch
# def kron(A: SelfAdjoint, B: SelfAdjoint) -> Kronecker:
#     return SelfAdjoint(kron(A.A, B.A))
# will create some problems?


@dispatch
@export
def kronsum(A: Any, B: Any) -> KronSum:
    return kronsum(lazify(A), lazify(B))


@dispatch
def kronsum(A: LinearOperator, B: LinearOperator) -> KronSum:
    return KronSum(*[A, B])


@dispatch
def kronsum(A: KronSum, B: LinearOperator) -> KronSum:
    return KronSum(*(A.Ms + (B, )))


@dispatch
def kronsum(A: LinearOperator, B: KronSum) -> KronSum:
    return KronSum(*((A, ) + B.Ms))

@export
def block_diag(*ops: List[LinearOperator]) -> LinearOperator:
    """ Construct a block diagonal operator from a list of ops. """
    return BlockDiag(*ops)

def concatenate(ops: List[LinearOperator], axis=0) -> LinearOperator:
    raise NotImplementedError
