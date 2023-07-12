from functools import reduce, partial
from cola.ops.operator_base import LinearOperator
import numpy as np
from cola.utils.dispatch import parametric
import cola
from typing import Callable

from cola.algorithms import lanczos, arnoldi

from cola.utils import export

@export
class UnitaryDecomposition(LinearOperator):
    """ Decomposition of form Q A Q^H. 
    
    Convenient for computing inverses, eigs, traces, determinants.
    Assumes Q is unitary (or more precisely a Stiefel matrix): Q.H@Q = I,
    but not necessarily Q@Q.H = I and Q need not be square.
    """
    def __init__(self, Q, A):
        super().__init__(A.dtype, A.shape)
        self.Q = cola.fns.lazify(Q)
        self.A = cola.fns.lazify(A)


from cola.linalg import inverse, eigs, trace, logdet, apply_unary

@inverse.dispatch
def inverse(A: UnitaryDecomposition, **kwargs):
    Q, A = A.Q, A.A
    return Q @ inverse(A,**kwargs) @ Q.H

@eigs.dispatch
def eigs(A: UnitaryDecomposition, **kwargs):
    Q, A = A.Q, A.A
    return eigs(A,**kwargs)

@trace.dispatch
def trace(A: UnitaryDecomposition, **kwargs):
    Q, A = A.Q, A.A
    return trace(A,**kwargs)

@apply_unary.dispatch
def apply_unary(fn: Callable, A: UnitaryDecomposition, **kwargs):
    Q, A = A.Q, A.A
    return Q@apply_unary(fn, A, **kwargs)@Q.H