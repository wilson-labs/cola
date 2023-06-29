from plum import dispatch
from linops.operator_base import LinearOperator
from linops.operators import Unitary
from linops.operators import Diagonal
from linops.operators import Identity
from linops.operators import ScalarMul
from linops.operators import Product
from linops.operators import BlockDiag
from linops.operators import Kronecker, SelfAdjoint, Sum
from linops.algorithms.cg import solve_cg
from linops.algorithms.gmres import run_gmres
from linops.algorithms.svrg import solve_svrg_symmetric
import numpy as np
from linops.utils.parametric import parametric


@parametric
class IterativeInverse(LinearOperator):
    def __str__(self):
        return f"{str(self.A)}⁻¹"


class CGInverse(IterativeInverse):
    def __init__(self, A, **kwargs):
        super().__init__(A.dtype, A.shape)
        self.A = A
        self.kwargs = kwargs
        self.info = {}

    def _matmat(self, X):
        # TODO: move this info handling to solve_cg?
        out, self.info = solve_cg(self.A, X, **self.kwargs)
        return out


class GMResInverse(IterativeInverse):
    def __init__(self, A, **kwargs):
        super().__init__(A.dtype, A.shape)
        self.A = A
        self.kwargs = kwargs
        self.info = {}

    def _matmat(self, X):
        out = run_gmres(self.A, X, **self.kwargs)
        return out


class SymmetricSVRGInverse(IterativeInverse):
    def __init__(self, A, **kwargs):
        super().__init__(A.dtype, A.shape)
        self.A = A
        self.kwargs = kwargs
        self.info = {}

    def _matmat(self, X):
        # TODO: move this info handling to solve_cg?
        out, self.info = solve_svrg_symmetric(self.A, X, **self.kwargs)
        return out


class GenericSVRGInverse(IterativeInverse):
    pass


@dispatch
def inverse(A: LinearOperator, **kwargs):
    kws = dict(tol=1e-6, P=None, x0=None, pbar=False, info=False, max_iters=5000)
    kws.update(kwargs)
    method = kws.pop('method', 'auto')
    if method == 'dense' or (method == 'auto' and np.prod(A.shape) <= 1e6):
        return A.ops.inv(A.to_dense())
    if issubclass(type(A), SelfAdjoint[Sum]) and (method == 'svrg' or (method == 'auto' and len(A.A.Ms) > 1e4)):
        return SymmetricSVRGInverse(A.A, **kws)
    if issubclass(type(A), Sum) and (method == 'svrg' or (method == 'auto' and len(A.Ms) > 1e4)):
        return GenericSVRGInverse(A, **kws)
    if issubclass(type(A), SelfAdjoint) and (method == 'cg' or (method == 'auto' and np.prod(A.shape) > 1e6)):
        return CGInverse(A, **kws)
    if method == 'gmres' or (method == 'auto' and np.prod(A.shape) > 1e6):
        return GMResInverse(A, **kws)
    else:
        raise ValueError(f"Unknown method {method} or CoLA didn't fit any selection criteria")


@dispatch
def inverse(A: Identity, **kwargs):
    return A


@dispatch
def inverse(A: ScalarMul, **kwargs) -> ScalarMul:
    return ScalarMul(1 / A.c, shape=A.shape, dtype=A.dtype)


@dispatch
def inverse(A: Product, **kwargs) -> Product:
    output = [inverse(M, **kwargs) for M in A.Ms].reverse()
    return Product(*output)


@dispatch
def inverse(A: BlockDiag, **kwargs) -> BlockDiag:
    return BlockDiag(*[inverse(M, **kwargs) for M in A.Ms], multiplicities=A.multiplicities)


@dispatch
def inverse(A: Kronecker, **kwargs) -> Kronecker:
    return Kronecker(*[inverse(M, **kwargs) for M in A.Ms])


@dispatch
def inverse(A: Diagonal, **kwargs) -> Diagonal:
    return Diagonal(1. / A.diag)


@dispatch
def inverse(A: Unitary, **kwargs) -> Unitary:
    return Unitary(A.H)
