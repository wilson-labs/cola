from functools import reduce
from typing import Set, Union
from collections.abc import Iterable
from plum import dispatch
from cola.ops import LinearOperator, Array
from cola.ops import Kronecker, Product, Sum
from cola.ops import Transpose, Adjoint
from cola.ops import BlockDiag, Identity, ScalarMul
from cola.ops import Hessian, Permutation, Sliced
from cola.utils import export

Scalar = Array


class WrapMeta(type):
    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__name__

    def __call__(self, obj: LinearOperator):
        new_obj = obj.xnp.tree_unflatten(*obj.xnp.tree_flatten(obj)[::-1])
        new_obj.annotations = obj.annotations | {self}
        # possible issues with pytrees and immutability?
        # TODO: recreate object with annotation.
        # How to do this? Note, annotations may not be in the top level __init__
        # and we wouldn't necessarily want to require that from our users
        # Would have to be in calling the LinearOperator base class constructor I think
        return new_obj


class Annotation(metaclass=WrapMeta):
    pass


class SelfAdjoint(Annotation):
    """ Annotation for Self-Adjoint (Hermitian) matrices. A^H=A
        Means symmetric for real matrices. A^T = A"""
    pass


Hermitian = SelfAdjoint


class PSD(SelfAdjoint):
    """ Annotation for Positive Semi-Definite matrices. A >= 0
        all eigenvalues are greater than or equal to zero,
        and the matrix should be self-adjoint. """
    pass


class Stiefel(Annotation):
    """ Annotation for Stiefel matrices (incomplete unitary).
        A^H A = I but A A^H != I. """
    pass


class Unitary(Stiefel):
    """ Annotation for Unitary matrices. A^H A = I
        and A A^H = I. """
    pass


[export(a) for a in [Stiefel, Unitary, SelfAdjoint, Hermitian, PSD]]


@dispatch
@export
def get_annotations(A: LinearOperator) -> Set[str]:
    """ Return the get_annotations of a linear operator.
        Called in the constructor of LinearOperator."""
    return set()


def intersect_annotations(ops: Iterable[LinearOperator]) -> Set[str]:
    return reduce(lambda x, y: x & y, (op.annotations for op in ops))


@dispatch
def get_annotations(A: Kronecker):
    return intersect_annotations(A.Ms)


inferred_self_adjoint_types = Union[Product[LinearOperator, Union[Transpose[LinearOperator],
                                                                  Adjoint[LinearOperator]]],
                                    Product[Union[Transpose[LinearOperator],
                                                  Adjoint[LinearOperator]], LinearOperator]]


def are_the_same(A1, A1T):
    if isinstance(A1T, Adjoint):
        return A1 is A1T.A
    elif isinstance(A1T, Transpose):
        return A1 is A1T.A
    elif isinstance(A1, Adjoint):
        return A1.A is A1T
    elif isinstance(A1, Transpose):
        return A1.A is A1T
    else:
        return False


@dispatch
def get_annotations(A: Product):
    if issubclass(type(A), inferred_self_adjoint_types) and are_the_same(A.Ms[0], A.Ms[1]):
        return (intersect_annotations(A.Ms) & {Unitary, Stiefel}) | {PSD}
    not_commuting = [M for M in A.Ms if not isinstance(M, ScalarMul)]
    if len(not_commuting) == 1:
        return not_commuting[0].annotations
    return intersect_annotations(A.Ms) & {Unitary, Stiefel}


# @dispatch(cond= lambda A: are_the_same(A.Ms[0], A.Ms[1]))
# def get_annotations(A: inferred_self_adjoint_types):
#     # TODO: doesn't work properly at the moment due to challenges with parametric types in plum
#     return intersect_annotations(A.Ms) | {SelfAdjoint}


@dispatch
def get_annotations(A: Sum):
    return intersect_annotations(A.Ms) - {Unitary, Stiefel}


@dispatch
def get_annotations(A: BlockDiag):
    return intersect_annotations(A.Ms)


# @dispatch
# def get_annotations(A: Diagonal):
#     if A.xnp.isreal(A.diag).all():
#         if all(A.diag >= 0):
#             return {PSD}
#         return {SelfAdjoint}
#     return set()


@dispatch
def get_annotations(A: Hessian):
    return {SelfAdjoint}


@dispatch
def get_annotations(A: Identity):
    return {Unitary, PSD}


@dispatch
def get_annotations(A: Permutation):
    return {Unitary}


# @dispatch
# def get_annotations(A: Dense):
#     xnp = A.xnp
#     sym = A.shape[0] == A.shape[-1] and xnp.allclose(A.A, xnp.conj(A.A.T))
#     return {SelfAdjoint} if sym else set()


@dispatch
def get_annotations(A: Sliced):
    symmetric = False
    if isinstance(A.slices[0], slice) and isinstance(A.slices[0], slice):
        if A.slices[0] == A.slices[1]:
            symmetric = True
    elif (A.slices[0] == A.slices[1]).all():
        symmetric = True
    if symmetric:
        return A.A.annotations - {Unitary, Stiefel}
    # TODO: perhaps add case of slicing a unitary matrix
    return set()


@dispatch
def get_annotations(A: Transpose):
    # possible problem with complex
    return A.A.annotations


@dispatch
def get_annotations(A: Adjoint):
    return A.A.annotations
