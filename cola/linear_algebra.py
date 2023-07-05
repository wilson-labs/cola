from typing import List, Union, Any
from plum import dispatch
from cola.ops import LinearOperator, Array
from cola.ops import Dense
from cola.ops import Kronecker, Product, KronSum, Sum
from cola.ops import ScalarMul, Transpose, Adjoint, SelfAdjoint
from cola.ops import BlockDiag, Identity, Diagonal, I_like

Scalar = Array


def lazify(A: Union[LinearOperator, Array]) -> LinearOperator:
    """ Convert an array to a linear operator if it is not already one. """
    if isinstance(A, LinearOperator):
        return A
    else:
        return Dense(A)


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
    return ScalarMul(A.c * c, A.shape)


@dispatch
def mul(c: Scalar, A: ScalarMul) -> ScalarMul:
    return ScalarMul(A.c * c, A.shape)


@dispatch
def mul(A: ScalarMul, B: ScalarMul) -> ScalarMul:
    return ScalarMul(A.c * B.c, A.shape)


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
def kron(A: Any, B: Any) -> Kronecker:
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


@dispatch
def diag(v: Array, k=0):
    assert k == 0, "Off diagonal diag not yet supported"
    assert len(v.shape) == 1, f"Unknown input {v.shape}"
    return Diagonal(v)


@dispatch
def diag(A: LinearOperator, k=0) -> Array:
    raise NotImplementedError


@dispatch
def diag(A: Dense, diagonal=0) -> Array:
    xnp = A.ops
    return xnp.diag(A.A, diagonal=diagonal)


@dispatch
def diag(A: Identity, k=0) -> Array:
    if k == 0:
        return A.ops.ones(A.shape[0], A.dtype)
    else:
        return A.ops.zeros(A.shape[0] - k, A.dtype)


@dispatch
def diag(A: Sum, k=0) -> Array:
    return Sum(diag(M) for M in A.Ms)


@dispatch
def diag(A: BlockDiag, k=0) -> Array:
    assert k == 0, "Havent filled this case yet, need to pad with 0s"
    return A.ops.concatenate([diag(M) for M in A.Ms])


@dispatch
def diag(A: ScalarMul, k=0) -> Array:
    return A.c * diag(I_like(A), k=k)


def trace(A: LinearOperator):
    assert A.shape[0] == A.shape[1], "Can't trace non square matrix"
    return A.ops.sum(diag(A))


def block_diag(*ops: List[LinearOperator]) -> LinearOperator:
    """ Construct a block diagonal operator from a list of ops. """
    return BlockDiag(*ops)


def concatenate(ops: List[LinearOperator], axis=0) -> LinearOperator:
    raise NotImplementedError
