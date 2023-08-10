from plum import dispatch
from cola.ops import Array
from cola.ops import LinearOperator, Triangular, Permutation
from cola.ops import Diagonal, Kronecker, BlockDiag, Product
from cola.utils import export
from cola.annotations import SelfAdjoint
from cola.algorithms import stochastic_lanczos_quad
import cola
import numpy as np

@dispatch
@export
def logdet(A: LinearOperator, **kwargs) -> Array:
    kws = dict(method="auto", tol=1e-2, pbar=False, max_iters=5000)
    assert not kwargs.keys() - kws.keys(), f"Unknown kwargs {kwargs.keys()-kws.keys()}"
    kws.update(kwargs)
    method = kws.pop('method', 'auto')
    if method in ('dense','exact') or (method == 'auto' and (np.prod(A.shape) <= 1e6 or kws['tol']<3e-2)):
        return logdet(cola.decompositions.lu_decomposed(A), **kws)
    elif method in ('iterative', 'approx') or (method == 'auto' and (np.prod(A.shape) > 1e6 and kws['tol']>=3e-2)):
        raise NotImplementedError("Approximate logdet not implemented yet for non self adjoint matrices")
    else:
        raise ValueError(f"Unknown method {method} or CoLA didn't fit any selection criteria")

@dispatch(cond=lambda A: A.isa(SelfAdjoint))
def logdet(A: LinearOperator, **kwargs) -> Array:
    kws = dict(method="auto", tol=1e-2, pbar=False, max_iters=5000)
    #assert not kwargs.keys() - kws.keys(), f"Unknown kwargs {kwargs.keys()-kws.keys()}"
    kws.update(kwargs)
    method = kws.pop('method', 'auto')
    if method in ('dense','exact') or (method == 'auto' and (np.prod(A.shape) <= 1e6 or kws['tol']<3e-2)):
        return logdet(cola.decompositions.cholesky_decomposed(A), **kws)
    elif method in ('iterative', 'approx') or (method == 'auto' and (np.prod(A.shape) > 1e6 and kws['tol']>=3e-2)):
        return stochastic_lanczos_quad(A, A.xnp.log, **kws)
    else:
        raise ValueError(f"Unknown method {method} or CoLA didn't fit any selection criteria")

@dispatch(cond = lambda A, **kwargs: all([(Ai.shape[-2] == Ai.shape[-1]) for Ai in A.Ms]))
def logdet(A: Product, **kwargs) -> Array:
    return sum(logdet(Ai) for Ai in A.Ms)

@dispatch
def logdet(A: Diagonal, **kwargs) -> Array:
    xnp = A.xnp
    out = xnp.sum(xnp.log(A.diag))
    return out

@dispatch
def logdet(A: Kronecker, **kwargs) -> Array:
    # logdet(Pi A_i \otimes I) = sum_i logdet(A_i)
    return sum(logdet(Ai) for Ai in A.Ms)


@dispatch
def logdet(A: BlockDiag, **kwargs) -> Array:
    # logdet(\bigoplus A_i) = log \prod det(A_i) = sum_i logdet(A_i)
    return sum(logdet(Ai) for Ai in A.Ms)

@dispatch
def logdet(A: Triangular, **kwargs) -> Array:
    xnp = A.xnp
    return xnp.sum(xnp.log(xnp.diag(A.A)))

@dispatch
def logdet(A: Permutation, **kwargs) -> Array:
    # TODO: count the parity of the permutation and return an error if it is odd
    return 0.