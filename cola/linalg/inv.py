import numpy as np
from plum import dispatch
from cola.ops import LinearOperator
from cola.ops import Diagonal, Permutation
from cola.ops import Identity, Dense
from cola.ops import ScalarMul, Sum
from cola.ops import BlockDiag, Triangular
from cola.ops import Kronecker, Product
from cola.algorithms.cg import cg
from cola.algorithms.gmres import gmres
from cola.algorithms.svrg import solve_svrg_symmetric
from cola.utils.dispatch import parametric
from cola.utils import export
from cola.annotations import PSD, Unitary
import cola


@parametric
class IterativeInv(LinearOperator):
    def __str__(self):
        return f"{str(self.A)}⁻¹"


class CGInv(IterativeInv):
    def __init__(self, A, **kwargs):
        super().__init__(A.dtype, A.shape, annotations={PSD})
        self.A = A
        self.kwargs = kwargs
        self.info = {}

    def _matmat(self, X):
        out, self.info = cg(self.A, X, **self.kwargs)
        return out


class GMResInv(IterativeInv):
    def __init__(self, A, **kwargs):
        super().__init__(A.dtype, A.shape)
        self.A = A
        self.kwargs = kwargs
        self.info = {}

    def _matmat(self, X):
        out, self.info = gmres(self.A, X, **self.kwargs)
        return out


class SymmetricSVRGInv(IterativeInv):
    def __init__(self, A, **kwargs):
        super().__init__(A.dtype, A.shape)
        self.A = A
        self.kwargs = kwargs
        self.info = {}

    def _matmat(self, X):
        out, self.info = solve_svrg_symmetric(self.A, X, **self.kwargs)
        return out


class GenericSVRGInv(IterativeInv):
    pass


@parametric
class TriangularInv(LinearOperator):
    def __init__(self, A: Triangular):
        super().__init__(A.dtype, A.shape)
        self.A = A.to_dense()
        self.lower = A.lower

    def _matmat(self, X):
        return self.xnp.solvetri(self.A, X, lower=self.lower)

    def _rmatmat(self, X):
        return self.xnp.solvetri(self.A.T, X.T, lower=not self.lower).T


@dispatch
@export
def inv(A: LinearOperator, **kwargs):
    """(lazily) computes the inverse of a linear operator, equivalent to solve.

    Args:
        A (LinearOperator): The linear operator to compute the inverse of.
        tol (float, optional): The tolerance criteria. Defaults to 1e-6.
        P (array, optional): Preconditioner matrix
        x0 (array, optional): The initial guess for the solution
        pbar (bool, optional): Whether to show a progress bar. Defaults to False.
        max_iters (int, optional): The maximum number of iterations. Defaults to 5000.
        method (str, optional): Method to use, defaults to 'auto',
         options are 'auto', 'dense', 'iterative', 'svrg'.

    Returns:
        LinearOperator: The inverse of the linear operator.

    Example:
        >>> A = MyLinearOperator()
        >>> x = cola.inverse(A) @ b

    """
    kws = dict(method="auto", tol=1e-6, P=None, x0=None, pbar=False, max_iters=5000)
    assert not kwargs.keys() - kws.keys(), f"Unknown kwargs {kwargs.keys()-kws.keys()}"
    kws.update(kwargs)
    method = kws.pop('method', 'auto')
    if method == 'dense' or (method == 'auto' and np.prod(A.shape) <= 1e6):
        return inv(cola.decompositions.lu_decomposed(A))
    elif method == 'iterative' or (method == 'auto' and np.prod(A.shape) > 1e6):
        return GMResInv(A, **kws)
    else:
        raise ValueError(f"Unknown method {method} or CoLA didn't fit any selection criteria")


@dispatch(cond=lambda A, **kwargs: A.isa(PSD))
def inv(A: LinearOperator, **kwargs):
    kws = dict(method="auto", tol=1e-6, P=None, x0=None, pbar=False, max_iters=5000)
    assert not kwargs.keys() - kws.keys(), f"Unknown kwargs {kwargs.keys()-kws.keys()}"
    kws.update(kwargs)
    method = kws.pop('method', 'auto')
    if method == 'dense' or (method == 'auto' and np.prod(A.shape) <= 1e6):
        return inv(cola.decompositions.cholesky_decomposed(A))
    if method == 'iterative' or (method == 'auto' and np.prod(A.shape) > 1e6):
        return CGInv(A, **kws)
    else:
        raise ValueError(f"Unknown method {method} or CoLA didn't fit any selection criteria")


@dispatch(cond=lambda A, **kwargs: A.isa(Unitary))
def inv(A: LinearOperator, **kwargs):
    return Unitary(A.H)


@dispatch
def inv(A: Identity, **kwargs):
    return A


@dispatch
def inv(A: ScalarMul, **kwargs):
    return ScalarMul(1 / A.c, shape=A.shape, dtype=A.dtype)


@dispatch
def inv(A: Permutation, **kwargs):
    return Permutation(A.xnp.argsort(A.perm), A.dtype)


@dispatch(cond=lambda A, **kwargs: all([M.shape[-2] == M.shape[-1] for M in A.Ms]))
def inv(A: Product, **kwargs):
    output = reversed([inv(M, **kwargs) for M in A.Ms])
    return Product(*output)


@dispatch
def inv(A: BlockDiag, **kwargs):
    return BlockDiag(*[inv(M, **kwargs) for M in A.Ms], multiplicities=A.multiplicities)


@dispatch
def inv(A: Kronecker, **kwargs):
    return Kronecker(*[inv(M, **kwargs) for M in A.Ms])


@dispatch
def inv(A: Diagonal, **kwargs):
    return Diagonal(1. / A.diag)


@dispatch
def inv(A: Triangular, **kwargs):
    return TriangularInv(A)


@dispatch
def inv(A: Sum[Product[Dense,Dense], Diagonal], **kwargs):
    U, V = A.Ms[0].Ms
    D_inv = inv(A.Ms[1], **kwargs)
    I = Identity(shape=(V.shape[0], U.shape[1]), dtype=V.dtype) 
    return D_inv - D_inv @ U @ inv(V @ D_inv @ U + I, **kwargs) @ V @ D_inv


@dispatch
def inv(A: Sum[Diagonal, Product[Dense,Dense]], **kwargs):
    return inv(Product(*A.Ms[1].Ms) + A.Ms[0], **kwargs)