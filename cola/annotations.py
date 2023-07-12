from typing import List, Union, Any, Set
from collections.abc import Iterable
from plum import dispatch
from cola.ops import LinearOperator, Array
from cola.ops import Dense
from cola.ops import Kronecker, Product, KronSum, Sum
from cola.ops import ScalarMul, Transpose, Adjoint
from cola.ops import BlockDiag, Identity, Diagonal, I_like
from cola.ops import Diagonal, Hessian, Permutation, Sliced
from cola.utils import export
# import reduce
from functools import reduce

Scalar = Array

class WrapMeta(type):

    def __str__(self):
        return self.__name__
    
    def __repr__(self):
        return self.__name__

    def __call__(self, obj: LinearOperator):
        new_obj = obj.__class__(*obj._args, **obj._kwargs)
        new_obj.annotations = obj.annotations | {self}
        # possible issues with pytrees and immutability?
        return new_obj


class Annotation(metaclass=WrapMeta):
    pass

class Stiefel(Annotation):
    pass

class Unitary(Stiefel):
    pass

class SelfAdjoint(Annotation):
    pass

Symmetric = Hermitian = SelfAdjoint

class PSD(SelfAdjoint):
    pass

[export(a) for a in [Stiefel, Unitary, SelfAdjoint, Symmetric, Hermitian,PSD]]

@dispatch
@export
def get_annotations(A: LinearOperator) -> Set[str]:
    """ Return the get_annotations of a linear operator. """
    return set()

def intersect_annotations(ops: Iterable[LinearOperator]) -> Set[str]:
    return reduce(lambda x, y: x & y, (op.annotations for op in ops))

@dispatch
def get_annotations(A: Kronecker):
    return intersect_annotations(A.Ms)

@dispatch
def get_annotations(A: Product):
    return intersect_annotations(A.Ms)&{Unitary,Stiefel}

@dispatch
def get_annotations(A: Sum):
    return intersect_annotations(A.Ms)-{Unitary,Stiefel}

@dispatch
def get_annotations(A: BlockDiag):
    return intersect_annotations(A.Ms)

@dispatch
def get_annotations(A: Diagonal):
    if A.ops.isreal(A.diag).all(): 
        if all(A.diag>=0):
            return {PSD}
        return {SelfAdjoint}
    return set()

@dispatch
def get_annotations(A: Hessian):
    return {SelfAdjoint}

@dispatch
def get_annotations(A: Identity):
    return {Unitary, PSD}

@dispatch
def get_annotations(A: Permutation):
    return {Unitary}

@dispatch
def get_annotations(A: Dense):
    xnp = A.ops
    sym = A.shape[0]==A.shape[-1] and xnp.allclose(A.A, xnp.conj(A.A.T))
    return {SelfAdjoint} if sym else set()

@dispatch
def get_annotations(A: Sliced):
    if A.slices[0]==A.slices[1]:
        return get_annotations(A.M)-{Unitary,Stiefel}
    # TODO: perhaps add case of slicing a unitary matrix
    return set()

@dispatch
def get_annotations(A: Transpose):
    #possible problem with complex
    return A.A.annotations

@dispatch
def get_annotations(A: Adjoint):
    return A.A.annotations