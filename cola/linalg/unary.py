from plum import dispatch
import numpy as np
from numbers import Number
from typing import Callable
from functools import reduce
import cola
from cola.ops import LinearOperator, Dense
from cola.utils import export
from cola.annotations import SelfAdjoint
from cola.utils.dispatch import parametric
from cola.algorithms.lanczos import lanczos_parts, construct_tridiagonal_batched
from cola.algorithms.arnoldi import get_arnoldi_matrix
from cola.ops import Diagonal, Identity, ScalarMul
from cola.ops import BlockDiag, Kronecker, KronSum, I_like, Transpose, Adjoint


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
        alpha, beta, Q, iters, info = lanczos_parts(self.A, V, **self.kwargs)
        self.info.update(info)
        alpha, beta, Q = alpha[:, :iters - 1], beta[:, :iters], Q[:, :, :iters]
        T = construct_tridiagonal_batched(alpha, beta, alpha)
        eigvals, P = self.xnp.eigh(T)
        norms = self.xnp.norm(V, axis=0)
        out = self.A.xnp.conj(P)[:, 0, :] * norms[:, None]  # (bs,k)
        return (Q @ P @ (self.f(eigvals) * out)[..., None])[..., 0]


@parametric
class ArnoldiUnary(LinearOperator):
    def __init__(self, A: LinearOperator, f: Callable, **kwargs):
        super().__init__(A.dtype, A.shape)
        self.A = A
        self.f = f
        self.kwargs = kwargs
        self.info = {}

    def _matmat(self, V):
        Q, H, _, info = get_arnoldi_matrix(A=self.A, rhs=V, **self.kwargs)
        Q = self.xnp.moveaxis(Q, 2, 0)
        H = self.xnp.moveaxis(H, 2, 0)
        # Q of shape (n, m, bs) H of shape (m,m,bs)
        self.info.update(info)
        eigvals, P = self.xnp.eig(H)
        norms = self.xnp.norm(V, axis=0)
        e0 = self.xnp.canonical(0, (P.shape[1], V.shape[-1]), dtype=P.dtype)
        Pinv0 = self.xnp.solve(P, e0.T)
        out = Pinv0 * norms[:, None]  # (bs,k)
        Q = self.xnp.cast(Q, dtype=P.dtype)
        return (Q @ P @ (self.f(eigvals) * out)[..., None])[..., 0]


@dispatch
@export
def apply_unary(f: Callable, A: LinearOperator, **kwargs):
    """ Generic apply a unary function f to a linear operator A: f(A)

    Args:
        f (Callable): The function to apply.
        A (LinearOperator): The linear operator to compute f(A) with.
        tol (float, optional): The tolerance criteria. Defaults to 1e-6.
        pbar (bool, optional): Whether to show a progress bar. Defaults to False.
        max_iters (int, optional): The maximum number of iterations. Defaults to 300.
        method (str, optional): Method to use, defaults to 'auto',
         options are 'auto', 'dense', 'iterative'.

    Returns:
        LinearOperator: the lazily implemented f(A)
    """
    kws = dict(method="auto", tol=1e-6, pbar=False, max_iters=300)
    kws.update(kwargs)
    method = kws.pop('method', 'auto')
    xnp = A.xnp
    if method == 'dense' or (method == 'auto' and (np.prod(A.shape) <= 1e6)):
        eigs, V = xnp.eig(A.to_dense())
        V = Dense(V)
        return V @ cola.diag(f(eigs)) @ cola.inverse(V)
    elif method == 'iterative' or (method == 'auto' and (np.prod(A.shape) > 1e6)):
        return ArnoldiUnary(A, f, **kws)
    else:
        raise ValueError(f"Unknown method {method} or CoLA didn't fit any selection criteria")


@dispatch(cond=lambda _, A, **kwargs: A.isa(SelfAdjoint))
def apply_unary(f: Callable, A: LinearOperator, **kwargs):
    kws = dict(method="auto", tol=1e-6, pbar=False, max_iters=300)
    kws.update(kwargs)
    method = kws.pop('method', 'auto')
    xnp = A.xnp
    if method == 'dense' or (method == 'auto' and (np.prod(A.shape) <= 1e6)):
        eigs, V = xnp.eigh(A.to_dense())
        V = Dense(V)
        return V @ cola.diag(f(eigs)) @ V.H
    elif method == 'iterative' or (method == 'auto' and (np.prod(A.shape) > 1e6)):
        return LanczosUnary(A, f, **kws)
    else:
        raise ValueError(f"Unknown method {method} or CoLA didn't fit any selection criteria")


@dispatch
def apply_unary(f: Callable, A: Diagonal, **kwargs):
    return Diagonal(f(A.diag))


@dispatch
def apply_unary(f: Callable, A: BlockDiag, **kwargs):
    fAs = [apply_unary(f, a, **kwargs) for a in A.Ms]
    return BlockDiag(*fAs, multiplicities=A.multiplicities)


@dispatch
def apply_unary(f: Callable, A: Identity, **kwargs):
    one = A.xnp.array(1., dtype=A.dtype, device=A.device)
    return f(one) * A


@dispatch
def apply_unary(f: Callable, A: ScalarMul, **kwargs):
    return f(A.c) * I_like(A)


@dispatch
def apply_unary(f: Callable, A: Transpose, **kwargs):
    return Transpose(apply_unary(f, A.A, **kwargs))


@dispatch
def apply_unary(f: Callable, A: Adjoint, **kwargs):
    return Adjoint(apply_unary(f, A.A, **kwargs))


@dispatch
@export
def exp(A: LinearOperator, **kwargs):
    """ Computes the matrix exponential exp(A) of a matrix A.

     Args:
        A (LinearOperator): The linear operator to compute the exp of.
        tol (float, optional): The tolerance criteria. Defaults to 1e-6.
        pbar (bool, optional): Whether to show a progress bar. Defaults to False.
        max_iters (int, optional): The maximum number of iterations. Defaults to 300.
        method (str, optional): Method to use, defaults to 'auto',
         options are 'auto', 'dense', 'iterative'.

    Returns:
        LinearOperator: the lazily implemented expm(A)
    """
    return apply_unary(A.xnp.exp, A, **kwargs)


@dispatch
def exp(A: KronSum, **kwargs):
    return Kronecker(*[exp(a, **kwargs) for a in A.Ms])


@dispatch
@export
def pow(A: LinearOperator, alpha: Number, **kwargs):
    """ Computes the matrix power A^alpha of a matrix A.

     Args:
        A (LinearOperator): The linear operator to compute the power of.
        alpha (float): The power to compute.
        tol (float, optional): The tolerance criteria. Defaults to 1e-6.
        pbar (bool, optional): Whether to show a progress bar. Defaults to False.
        max_iters (int, optional): The maximum number of iterations. Defaults to 300.
        method (str, optional): Method to use, defaults to 'auto',
         options are 'auto', 'dense', 'iterative'.

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
            return cola.inverse(A)

    return apply_unary(lambda x: x**alpha, A, **kwargs)


@dispatch
def pow(A: Kronecker, alpha: Number, **kwargs):
    return Kronecker(*[pow(a, alpha, **kwargs) for a in A.Ms])


@export
def sqrt(A: LinearOperator, **kwargs):
    """ Computes the matrix sqrt A^{1/2} of a matrix A using the principal branch.
    """
    return pow(A, 0.5, **kwargs)


@export
def isqrt(A: LinearOperator, **kwargs):
    """ Computes the matrix inverse sqrt A^{-1/2} of a matrix A using the principal branch.
    """
    return pow(A, -0.5, **kwargs)
