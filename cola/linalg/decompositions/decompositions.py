""" Decompositions of linear operators, e.g. LU, Cholesky, Lanczos, Arnoldi, SVD"""
from plum import dispatch
from dataclasses import dataclass
from cola.ops import LinearOperator, Array
from cola.ops import Triangular, Permutation, Diagonal
from cola.ops import Identity, ScalarMul, Kronecker, BlockDiag
from cola.utils import export
from cola.linalg.algorithm_base import Algorithm
from cola.linalg.decompositions.lanczos import lanczos
from cola.linalg.decompositions.arnoldi import arnoldi
import cola.linalg


@export
class Cholesky(Algorithm):
    """ TODO: docstring"""
    def __call__(self, A: LinearOperator):
        return cholesky(A)


@export
class LU(Algorithm):
    """ TODO: docstring"""
    def __call__(self, A: LinearOperator):
        return plu(A)


@export
@dataclass
class Arnoldi(Algorithm):
    """ TODO: docstring"""
    start_vector: Array = None
    max_iters: int = 1_000
    tol: float = 1e-6
    pbar: bool = False

    def __call__(self, A: LinearOperator):
        return arnoldi(A, **self.__dict__)


@export
@dataclass
class Lanczos(Algorithm):
    """ TODO: docstring"""
    start_vector: Array = None
    max_iters: int = 1_000
    tol: float = 1e-6
    pbar: bool = False

    def __call__(self, A: LinearOperator):
        return lanczos(A, **self.__dict__)


@dispatch
@export
def cholesky(A: LinearOperator):
    """ Performs a cholesky decomposition A=LL* of a linear operator A.
        The returned operator L with triangular structure.

        (Implicitly assumes A is PSD)
    """
    return Triangular(A.xnp.cholesky(A.to_dense()), lower=True)


@dispatch
def cholesky(A: Identity):
    return A


@dispatch
def cholesky(A: Diagonal | ScalarMul):
    return cola.linalg.sqrt(A)


@dispatch
def cholesky(A: Kronecker):
    # see https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/kronthesisschaecke04.pdf
    return Kronecker(*[cholesky(Ai) for Ai in A.Ms])


@dispatch
def cholesky(A: BlockDiag):
    return BlockDiag(*[cholesky(Ai) for Ai in A.Ms], multiplicities=A.multiplicities)


@dispatch
@export
def plu(A: LinearOperator):
    """ Performs a cholesky decomposition A=PLU of a linear operator A.
        The returns P, L, U as linear operators, but represented using
        the triangular (and permutation) structure """
    p, L, U = A.xnp.lu(A.to_dense())
    P, L, U = Permutation(p), Triangular(L, lower=True), Triangular(U, lower=False)
    P, L, U = P.to(A.device), L.to(A.device), U.to(A.device)
    return P, L, U


@dispatch
def plu(A: Identity):
    return A, A, A


@dispatch
def plu(A: Diagonal | ScalarMul):
    S = cola.linalg.sqrt(A)
    return cola.ops.I_like(A), S, S


@dispatch
def plu(A: Kronecker):
    # see https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/kronthesisschaecke04.pdf
    P, L, U = zip(*[plu(Ai) for Ai in A.Ms])
    return Kronecker(*P), Kronecker(*L), Kronecker(*U)


@dispatch
def plu(A: BlockDiag):
    P, L, U = zip(*[plu(Ai) for Ai in A.Ms])
    BD = lambda *args: BlockDiag(*args, multiplicities=A.multiplicities)  # noqa
    return BD(*P), BD(*L), BD(*U)
