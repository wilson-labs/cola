from plum import dispatch
import numpy as np
from numbers import Number
from typing import Callable
from functools import reduce
import cola
from cola.ops import LinearOperator
from cola.utils import export
from cola.annotations import SelfAdjoint, PSD
from plum import parametric
from cola.algorithms.lanczos import lanczos  # , construct_tridiagonal
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
        xnp = self.xnp
        Q, T, info = lanczos(self.A, V, **self.kwargs)
        self.info.update(info)
        eigvals, P = self.xnp.eigh(T)
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
        LinearOperator: the lazily implemented f(A)"""
    kws = dict(method="auto", tol=1e-6, pbar=False, max_iters=300)
    kws.update(kwargs)
    method = kws.pop('method', 'auto')
    xnp = A.xnp
    if method == 'dense' or (method == 'auto' and (np.prod(A.shape) <= 1e6)):
        Adense = A.to_dense()
        eigs, V = xnp.eig(Adense)
        V = cola.lazify(V)
        D = cola.diag(f(eigs))
        return V @ D @ cola.inv(V)
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
        Adense = A.to_dense()
        eigs, V = xnp.eigh(Adense)
        V = cola.lazify(V)
        D = cola.diag(f(eigs))
        return V @ D @ V.H
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
        f (Callable): The function to apply.
        A (LinearOperator): The linear operator to compute f(A) with.
        tol (float, optional): The tolerance criteria. Defaults to 1e-6.
        pbar (bool, optional): Whether to show a progress bar. Defaults to False.
        max_iters (int, optional): The maximum number of iterations. Defaults to 300.
        method (str, optional): Method to use, defaults to 'auto',
            options are 'auto', 'dense', 'iterative'.

    Returns:
        LinearOperator: the lazily implemented exp(A)
    """
    return apply_unary(A.xnp.exp, A, **kwargs)


@dispatch
def exp(A: KronSum, **kwargs):
    return Kronecker(*[exp(a, **kwargs) for a in A.Ms])


@dispatch(cond=lambda A: A.isa(PSD))
@export
def log(A: LinearOperator, **kwargs):
    """ Computes the matrix logarithm log(A) of PSD matrix A

    Args:
        f (Callable): The function to apply.
        A (LinearOperator): The linear operator to compute f(A) with.
        tol (float, optional): The tolerance criteria. Defaults to 1e-6.
        pbar (bool, optional): Whether to show a progress bar. Defaults to False.
        max_iters (int, optional): The maximum number of iterations. Defaults to 300.
        method (str, optional): Method to use, defaults to 'auto',
            options are 'auto', 'dense', 'iterative'.

    Returns:
        LinearOperator: the lazily implemented log(A)
    """
    return apply_unary(A.xnp.log, A, **kwargs)


@dispatch
@export
def pow(A: LinearOperator, alpha: Number, **kwargs):
    """ Computes the matrix power A^alpha of a matrix A.

    Args:
        f (Callable): The function to apply.
        A (LinearOperator): The linear operator to compute f(A) with.
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
            return cola.inv(A)

    return apply_unary(lambda x: x**alpha, A, **kwargs)


@dispatch
def pow(A: Kronecker, alpha: Number, **kwargs):
    return Kronecker(*[pow(a, alpha, **kwargs) for a in A.Ms])


@export
def sqrt(A: LinearOperator, **kwargs):
    """ Computes the matrix sqrt A^{1/2} of a matrix A using the principal branch.

    Args:
        f (Callable): The function to apply.
        A (LinearOperator): The linear operator to compute f(A) with.
        tol (float, optional): The tolerance criteria. Defaults to 1e-6.
        pbar (bool, optional): Whether to show a progress bar. Defaults to False.
        max_iters (int, optional): The maximum number of iterations. Defaults to 300.
        method (str, optional): Method to use, defaults to 'auto',
            options are 'auto', 'dense', 'iterative'.

    Returns: LinearOperator: the lazily implemented sqrt(A)
    """
    return pow(A, 0.5, **kwargs)


@export
def isqrt(A: LinearOperator, **kwargs):
    """ Computes the matrix inverse sqrt A^{-1/2} of a matrix A using the principal branch.

    Args:
        f (Callable): The function to apply.
        A (LinearOperator): The linear operator to compute f(A) with.
        tol (float, optional): The tolerance criteria. Defaults to 1e-6.
        pbar (bool, optional): Whether to show a progress bar. Defaults to False.
        max_iters (int, optional): The maximum number of iterations. Defaults to 300.
        method (str, optional): Method to use, defaults to 'auto',
            options are 'auto', 'dense', 'iterative'.

    Returns: LinearOperator: the lazily implemented A^{-1/2}
    """
    return pow(A, -0.5, **kwargs)
