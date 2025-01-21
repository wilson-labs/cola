from dataclasses import dataclass
from functools import reduce
from numbers import Number
from typing import Callable

import numpy as np
from plum import dispatch, parametric

from cola.annotations import PSD, SelfAdjoint
from cola.fns import lazify
from cola.linalg.algorithm_base import Algorithm, Auto
from cola.linalg.decompositions.arnoldi import arnoldi
from cola.linalg.decompositions.decompositions import LU, Arnoldi, Cholesky, Lanczos
from cola.linalg.decompositions.lanczos import lanczos
from cola.linalg.inverse.cg import CG
from cola.linalg.inverse.gmres import GMRES
from cola.linalg.inverse.inv import inv
from cola.ops import (
    Adjoint,
    BlockDiag,
    Diagonal,
    I_like,
    Identity,
    Kronecker,
    KronSum,
    LinearOperator,
    ScalarMul,
    Transpose,
)
from cola.utils import export


def product(As):
    return reduce(lambda x, y: x @ y, As)


@parametric
class LanczosUnary(LinearOperator):
    def __init__(self, A: LinearOperator, f: Callable, **kwargs):
        super().__init__(A.dtype, A.shape, annotations={SelfAdjoint})
        self.A = A
        self.f = f
        self.kwargs = kwargs
        self.info = {}

    def _matmat(self, V):
        xnp = self.xnp
        if "start_vector" in self.kwargs.keys():
            self.kwargs.pop("start_vector")
        Q, T, info = lanczos(self.A, V, **self.kwargs)  # outputs are batched
        self.info.update(info)
        eigvals, P = self.xnp.eigh(xnp.vmap(T.__class__.to_dense)(T))
        Q = xnp.vmap(Q.__class__.to_dense)(Q)
        norms = self.xnp.norm(V, axis=0)
        zero_thresh = 10 * xnp.finfo(self.dtype).eps * xnp.max(xnp.abs(eigvals), axis=1, keepdims=True)
        # truncate zero padded values (generating spurious eigenvalues)
        f_eigvals = xnp.where(xnp.abs(eigvals) > zero_thresh, self.f(eigvals), xnp.zeros_like(eigvals))
        out = self.A.xnp.conj(P)[:, 0, :] * norms[:, None]  # (bs,k)
        out = (Q @ P @ (f_eigvals * out)[..., None])[..., 0]
        return out.T


@parametric
class ArnoldiUnary(LinearOperator):
    def __init__(self, A: LinearOperator, f: Callable, **kwargs):
        super().__init__(A.dtype, A.shape)
        self.A = A
        self.f = f
        self.kwargs = kwargs
        self.info = {}

    def _matmat(self, V):  # (n,bs)
        xnp = self.xnp
        if "start_vector" in self.kwargs.keys():
            self.kwargs.pop("start_vector")
        Q, H, info = arnoldi(A=self.A, start_vector=V, **self.kwargs)
        Q, H = Q.to_dense()[:, :, :-1], H.to_dense()[:, :-1]
        # Q of shape (n, m, bs) H of shape (m, m, bs)
        self.info.update(info)
        eigvals, P = self.xnp.eig(H)
        norms = self.xnp.norm(V, axis=0)

        e0 = self.xnp.canonical(0, (P.shape[1], V.shape[-1]), dtype=P.dtype, device=self.device)
        Pinv0 = self.xnp.solve(P, e0.T[..., None]).squeeze(-1)  # (bs, m, m) vs (bs, m)
        out = Pinv0 * norms[:, None]  # (bs, m)
        Q = self.xnp.cast(Q, dtype=P.dtype)  # (bs, n, m)
        # (bs,n,m) @ (bs,m,m) @ (bs, m) -> (bs, n)
        zero_thresh = 10 * xnp.finfo(self.dtype).eps * xnp.max(xnp.abs(eigvals), axis=1, keepdims=True)
        f_eigvals = xnp.where(xnp.abs(eigvals) > zero_thresh, self.f(eigvals), xnp.zeros_like(eigvals))
        out = (Q @ P @ (f_eigvals * out)[..., None])[..., 0]
        return out.T


@export
@dispatch.abstract
def apply_unary(f: Callable, A: LinearOperator, alg: Algorithm = Auto()):
    """ Generic apply a unary function :math:`f` to a linear operator :math:`A`.
    That is, :math:`f(A)`, defined through the taylor expansion:
        :math:`f(A) = \\sum_{k=0}^\\infty \\frac{f^{(k)}(0)}{k!}A^k`.

    Args:
        f (Callable): The function to apply.
        A (LinearOperator): The linear operator to compute f(A) with.
        alg (Algorithm): The algorithm to use (Auto, Eig, Eigh, Lanczos, Arnoldi).

    Returns:
        LinearOperator: a lazy instantiation of :math:`f(A)`
    """


# ########### BASE CASES #############


@dispatch(precedence=-1)
def apply_unary(f: Callable, A: LinearOperator, alg: Auto):
    """ Auto:
        - if A is Hermitian and small, use Eigh
        - if A is Hermitian and large, use Lanczos
        - if A is not Hermitian and small, use Eig
        - if A is not Hermitian and large, use Arnoldi
    """
    psd, small = A.isa(PSD), np.prod(A.shape) <= 1e6
    if psd and small:
        alg = Eigh()
    elif not psd and small:
        alg = Eig()
    elif psd and not small:
        alg = Lanczos(**alg.__dict__)
    elif not psd and not small:
        alg = Arnoldi(**alg.__dict__)
    return apply_unary(f, A, alg)


@dispatch(precedence=-1)
def apply_unary(f: Callable, A: LinearOperator, alg: Lanczos):
    assert A.isa(SelfAdjoint), "Lanczos only valid for SelfAdjoint, wrap in cola.SelfAdjoint if desired"
    return LanczosUnary(A, f, **alg.__dict__)


@dispatch(precedence=-1)
def apply_unary(f: Callable, A: LinearOperator, alg: Arnoldi):
    return ArnoldiUnary(A, f, **alg.__dict__)


@export
@dataclass
class Eigh(Algorithm):
    """
    Uses a dense eigendecomposition for a real symmetric or complex Hermitian operator.
    """


@export
@dataclass
class Eig(Algorithm):
    """
    Uses a dense eigendecomposition for a general square operator.
    """


@dispatch(precedence=-1)
def apply_unary(f: Callable, A: LinearOperator, alg: Eigh):
    assert A.isa(SelfAdjoint), "Eigh only valid for SelfAdjoint, wrap in cola.SelfAdjoint if desired"
    Adense = A.to_dense()
    eigs, V = A.xnp.eigh(Adense)
    V = lazify(V)
    D = Diagonal(f(eigs))
    return V @ D @ V.H


@dispatch(precedence=-1)
def apply_unary(f: Callable, A: LinearOperator, alg: Eig):
    Adense = A.to_dense()
    eigs, V = A.xnp.eig(Adense)
    V = lazify(V)
    D = Diagonal(f(eigs))
    return V @ D @ inv(V)


# ############ ADDITIONAL DISPATCH RULES ############
@dispatch
def apply_unary(f: Callable, A: Diagonal, alg: Algorithm):
    return Diagonal(f(A.diag))


@dispatch
def apply_unary(f: Callable, A: BlockDiag, alg: Algorithm):
    fAs = [apply_unary(f, a, alg) for a in A.Ms]
    return BlockDiag(*fAs, multiplicities=A.multiplicities)


@dispatch
def apply_unary(f: Callable, A: Identity, alg: Algorithm):
    one = A.xnp.array(1., dtype=A.dtype, device=A.device)
    return f(one) * A


@dispatch
def apply_unary(f: Callable, A: ScalarMul, alg: Algorithm):
    return f(A.c) * I_like(A)


@dispatch
def apply_unary(f: Callable, A: Transpose, alg: Algorithm):
    return Transpose(apply_unary(f, A.A, alg))


@dispatch
def apply_unary(f: Callable, A: Adjoint, alg: Algorithm):
    return Adjoint(apply_unary(f, A.A, alg))


# @export
# @dispatch.abstract
# def exp(A: LinearOperator, alg: Algorithm = Auto()):
#     """ Computes the matrix exponential :math:`\\exp(A)` of the operator :math:`A`.

#     Args:
#         f (Callable): The function to apply.
#         A (LinearOperator): The linear operator to compute f(A) with.
#         alg (Algorithm): The algorithm to use (Auto, Eig, Eigh, Lanczos, Arnoldi).

#     Returns:
#         LinearOperator: a lazy instantiation of :math:`\\exp(A)`
#     """


@dispatch
@export
def exp(A: LinearOperator, alg: Algorithm = Auto()):
    """ Computes the matrix exponential :math:`\\exp(A)` of the operator :math:`A`.

    Args:
        f (Callable): The function to apply.
        A (LinearOperator): The linear operator to compute f(A) with.
        alg (Algorithm): The algorithm to use (Auto, Eig, Eigh, Lanczos, Arnoldi).

    Returns:
        LinearOperator: a lazy instantiation of :math:`\\exp(A)`
    """
    return apply_unary(A.xnp.exp, A, alg)


@dispatch
def exp(A: KronSum, alg: Algorithm):
    return Kronecker(*[exp(a, alg) for a in A.Ms])


@dispatch
@export
def log(A: LinearOperator, alg: Algorithm = Auto()):
    """ Computes the matrix logarithm :math:`log(A)` of positive
    definite operator :math:`A`.

    Args:
        A (LinearOperator): The linear operator to compute f(A) with.
        alg (Algorithm): The algorithm to use (Auto, Eig, Eigh, Lanczos, Arnoldi).

    Returns:
        LinearOperator: a lazy instantiation of log(A)
    """
    return apply_unary(A.xnp.log, A, alg)


@dispatch
@export
def pow(A: LinearOperator, alpha: Number, alg: Algorithm = Auto()):
    """ Computes the matrix power :math:`A^{\\alpha}` of an operator :math:`A`,
    where :math:`\\alpha` is the coefficient.

    Args:
        A (LinearOperator): The linear operator to compute f(A) with.
        alg (Algorithm): The algorithm to use (Auto, Eig, Eigh, Lanczos, Arnoldi).

    Returns:
        LinearOperator: a lazy instantiation of :math:`A^{\\alpha}`
    """
    # check if alpha is close to an integer
    if np.isclose(alpha, (k := int(np.round(alpha)))):
        if k == 0:
            return I_like(A)
        if k > 0 and k < 10:
            return product([A] * k)
        if k == -1:
            match alg:
                case Lanczos():
                    new_alg = CG(**alg.__dict__)
                case Arnoldi():
                    new_alg = GMRES(**alg.__dict__)
                case Eigh():
                    new_alg = Cholesky()
                case Eig():
                    new_alg = LU()
                case _:
                    new_alg = alg
            return inv(A, new_alg)
    return apply_unary(lambda x: x**alpha, A, alg)


@dispatch
def pow(A: Kronecker, alpha: Number, alg: Algorithm):
    return Kronecker(*[pow(a, alpha, alg) for a in A.Ms])


@dispatch
@export
def sqrt(A: LinearOperator, alg: Algorithm = Auto()):
    """ Computes the square root, :math:`A^{1/2}`
    of an operator :math:`A` using the principal branch.

    Args:
        A (LinearOperator): The linear operator to compute f(A) with.
        alg (Algorithm): The algorithm to use (Auto, Eig, Eigh, Lanczos, Arnoldi).

    Returns:
        LinearOperator: a lazy instantiation of :math:`A^{1/2}`
    """
    return pow(A, 0.5, alg)


@dispatch
@export
def isqrt(A: LinearOperator, alg: Algorithm = Auto()):
    """ Computes the matrix inverse :math:`A^{-1/2}` of an
    operator :math:`A` using the principal branch.

    Args:
        A (LinearOperator): The linear operator to compute f(A) with.
        alg (Algorithm): The algorithm to use (Auto, Eig, Eigh, Lanczos, Arnoldi).

    Returns:
        LinearOperator: a lazy instantiation of :math:`A^{-1/2}`
    """
    return pow(A, -0.5, alg)
