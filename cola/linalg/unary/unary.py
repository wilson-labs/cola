from plum import dispatch
import numpy as np
from numbers import Number
from typing import Callable
from functools import reduce
from cola.fns import lazify
from cola.linalg.trace.diag_trace import diag
from cola.ops import LinearOperator
from cola.utils import export
from cola.annotations import SelfAdjoint, PSD
from plum import parametric
from cola.linalg.decompositions.lanczos import lanczos
from cola.linalg.decompositions.arnoldi import get_arnoldi_matrix
from cola.ops import Diagonal, Identity, ScalarMul
from cola.ops import BlockDiag, Kronecker, KronSum, I_like, Transpose, Adjoint
from cola.linalg.inverse.inv import inv
from cola.linalg.algorithm_base import Algorithm, Auto
from cola.linalg.decompositions.decompositions import Arnoldi, Lanczos, LU, Cholesky
from cola.linalg.inverse.cg import CG
from cola.linalg.inverse.gmres import GMRES


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
        Q, H, _, info = get_arnoldi_matrix(A=self.A, rhs=V, **self.kwargs)
        # Q of shape (n, m, bs) H of shape (m,m,bs)
        self.info.update(info)
        eigvals, P = self.xnp.eig(H)
        norms = self.xnp.norm(V, axis=0)

        e0 = self.xnp.canonical(0, (P.shape[1], V.shape[-1]), dtype=P.dtype, device=self.device)
        Pinv0 = self.xnp.solve(P, e0.T)  # (bs, m, m) vs (bs, m)
        out = Pinv0 * norms[:, None]  # (bs, m)
        Q = self.xnp.cast(Q, dtype=P.dtype)  # (bs, n, m)
        # (bs,n,m) @ (bs,m,m) @ (bs, m) -> (bs, n)
        zero_thresh = 10 * xnp.finfo(self.dtype).eps * xnp.max(xnp.abs(eigvals), axis=1, keepdims=True)
        f_eigvals = xnp.where(xnp.abs(eigvals) > zero_thresh, self.f(eigvals), xnp.zeros_like(eigvals))
        out = (Q @ P @ (f_eigvals * out)[..., None])[..., 0]
        return out.T


@dispatch.abstract
@export
def apply_unary(f: Callable, A: LinearOperator, alg: Algorithm):
    """ Generic apply a unary function f to a linear operator A: f(A)

    Args:
        f (Callable): The function to apply.
        A (LinearOperator): The linear operator to compute f(A) with.
        alg (Algorithm): The algorithm to use (Auto, Eig, Eigh, Lanczos, Arnoldi).

    Returns:
        LinearOperator: the lazily implemented f(A)"""


@dispatch(precedence=-1)
def apply_unary(f: Callable, A: LinearOperator, alg: Auto):
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
    return LanczosUnary(A, f, **alg.__dict__)


@dispatch(precedence=-1)
def apply_unary(f: Callable, A: LinearOperator, alg: Arnoldi):
    return ArnoldiUnary(A, f, **alg.__dict__)


class Eigh(Algorithm):
    """ Docstring here"""


class Eig(Algorithm):
    """ Docstring here"""


@dispatch(precedence=-1)
def apply_unary(f: Callable, A: LinearOperator, alg: Eigh):
    Adense = A.to_dense()
    eigs, V = A.xnp.eigh(Adense)
    V = lazify(V)
    D = diag(f(eigs))
    return V @ D @ V.H


@dispatch(precedence=-1)
def apply_unary(f: Callable, A: LinearOperator, alg: Eig):
    Adense = A.to_dense()
    eigs, V = A.xnp.eig(Adense)
    V = lazify(V)
    D = diag(f(eigs))
    return V @ D @ inv(V)


@dispatch
def apply_unary(f: Callable, A: Diagonal, alg=Auto()):
    return Diagonal(f(A.diag))


@dispatch
def apply_unary(f: Callable, A: BlockDiag, alg=Auto()):
    fAs = [apply_unary(f, a, alg) for a in A.Ms]
    return BlockDiag(*fAs, multiplicities=A.multiplicities)


@dispatch
def apply_unary(f: Callable, A: Identity, alg=Auto()):
    one = A.xnp.array(1., dtype=A.dtype, device=A.device)
    return f(one) * A


@dispatch
def apply_unary(f: Callable, A: ScalarMul, alg=Auto()):
    return f(A.c) * I_like(A)


@dispatch
def apply_unary(f: Callable, A: Transpose, alg=Auto()):
    return Transpose(apply_unary(f, A.A, alg))


@dispatch
def apply_unary(f: Callable, A: Adjoint, alg=Auto()):
    return Adjoint(apply_unary(f, A.A, alg))


@dispatch
@export
def exp(A: LinearOperator, alg=Auto()):
    """ Computes the matrix exponential exp(A) of a matrix A.

    Args:
        f (Callable): The function to apply.
        A (LinearOperator): The linear operator to compute f(A) with.
        alg (Algorithm): The algorithm to use (Auto, Eig, Eigh, Lanczos, Arnoldi).

    Returns:
        LinearOperator: the lazily implemented exp(A)
    """
    return apply_unary(A.xnp.exp, A, alg)


@dispatch
def exp(A: KronSum, alg=Auto()):
    return Kronecker(*[exp(a, alg) for a in A.Ms])


@dispatch
@export
def log(A: LinearOperator, alg=Auto()):
    """ Computes the matrix logarithm log(A) of PSD matrix A

    Args:
        A (LinearOperator): The linear operator to compute f(A) with.
        alg (Algorithm): The algorithm to use (Auto, Eig, Eigh, Lanczos, Arnoldi).

    Returns:
        LinearOperator: the lazily implemented log(A)
    """
    return apply_unary(A.xnp.log, A, alg)


@dispatch
@export
def pow(A: LinearOperator, alpha: Number, alg=Auto()):
    """ Computes the matrix power A^alpha of a matrix A.

    Args:
        A (LinearOperator): The linear operator to compute f(A) with.
        alg (Algorithm): The algorithm to use (Auto, Eig, Eigh, Lanczos, Arnoldi).

    Returns:
        LinearOperator: the lazily implemented A^alpha
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
def pow(A: Kronecker, alpha: Number, alg=Auto()):
    return Kronecker(*[pow(a, alpha, alg) for a in A.Ms])


@dispatch
@export
def sqrt(A: LinearOperator, alg=Auto()):
    """ Computes the matrix sqrt A^{1/2} of a matrix A using the principal branch.

    Args:
        A (LinearOperator): The linear operator to compute f(A) with.
        alg (Algorithm): The algorithm to use (Auto, Eig, Eigh, Lanczos, Arnoldi).

    Returns: LinearOperator: the lazily implemented sqrt(A)
    """
    return pow(A, 0.5, alg)


@dispatch
@export
def isqrt(A: LinearOperator, alg=Auto()):
    """ Computes the matrix inverse sqrt A^{-1/2} of a matrix A using the principal branch.

    Args:
        A (LinearOperator): The linear operator to compute f(A) with.
        alg (Algorithm): The algorithm to use (Auto, Eig, Eigh, Lanczos, Arnoldi).

    Returns: LinearOperator: the lazily implemented A^{-1/2}
    """
    return pow(A, -0.5, alg)
