from functools import reduce, partial
from cola.ops.operator_base import LinearOperator
import numpy as np
from cola.utils.dispatch import parametric
import cola
from typing import Callable

from cola.utils import export
from cola.linalg import inverse, eig, trace, logdet, apply_unary


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

@export
def LanczosDecomposition(A: LinearOperator, start_vector= None, max_iters=100, tol=1e-7, pbar=False):
    Q,T,*info = cola.algorithms.lanczos(A=A, start_vector=start_vector, max_iters=max_iters, tol=tol, pbar=pbar)
    return UnitaryDecomposition(Q,T)

@export
def ArnoldiDecomposition(A: LinearOperator, start_vector=None,
     max_iters=100, tol=1e-7, use_householder=False, pbar=False):
    Q,H,*info = cola.algorithms.arnoldi(A=A, start_vector=start_vector, max_iters=max_iters,
         tol=tol, use_householder=use_householder, pbar=pbar)
    return UnitaryDecomposition(Q,H)


@inverse.dispatch
def inverse(A: UnitaryDecomposition, **kwargs):
    Q, A = A.Q, A.A
    return Q @ inverse(A,**kwargs) @ Q.H

@eig.dispatch
def eig(A: UnitaryDecomposition, **kwargs):
    
    Q, A = A.Q, A.A
    print("called eig",Q.shape,A.shape)
    return eig(A,**kwargs)

@trace.dispatch
def trace(A: UnitaryDecomposition, **kwargs):
    Q, A = A.Q, A.A
    return trace(A,**kwargs)

@apply_unary.dispatch
def apply_unary(fn: Callable, A: UnitaryDecomposition, **kwargs):
    # Need to think carefully about the case where Q is not full rank
    Q, A = A.Q, A.A
    return Q@apply_unary(fn, A, **kwargs)@Q.H