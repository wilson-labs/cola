from plum import dispatch
from cola.ops import Array
from cola.ops import LinearOperator, Triangular, Permutation
from cola.ops import Diagonal, Kronecker, BlockDiag, Product
from cola.utils import export
from cola.annotations import SelfAdjoint, PSD
from cola.algorithms import stochastic_lanczos_quad
import cola
import numpy as np
from functools import reduce

def product(xs):
    return reduce(lambda x, y: x*y, xs, 1)

@export
def logdet(A: LinearOperator, **kwargs) -> Array:
    """ Computes logdet of a linear operator.
    Args:
        A (LinearOperator): The linear operator to compute the logdet of.
        tol (float, optional): The tolerance criteria. Defaults to 1e-6.
        pbar (bool, optional): Whether to show a progress bar. Defaults to False.
        max_iters (int, optional): The maximum number of iterations. Defaults to 300.
        method (str, optional): Method to use, defaults to 'auto',
         options are 'auto', 'dense', 'iterative'.
    
    Returns:
        Array: logdet
    """
    s, l = slogdet(A, **kwargs)
    return l

@dispatch
@export
def slogdet(A: LinearOperator, **kwargs) -> Array:
    """ Computes sign and logdet of a linear operator. such that det(A) = sign(A) exp(logdet(A))
    
     Args:
        A (LinearOperator): The linear operator to compute the logdet of.
        tol (float, optional): The tolerance criteria. Defaults to 1e-6.
        pbar (bool, optional): Whether to show a progress bar. Defaults to False.
        max_iters (int, optional): The maximum number of iterations. Defaults to 300.
        method (str, optional): Method to use, defaults to 'auto',
         options are 'auto', 'dense', 'iterative'.
    
    Returns:
        Tuple[Array,Array]: sign, logdet
    """
    kws = dict(method="auto", tol=1e-2, pbar=False, max_iters=300)
    assert not kwargs.keys() - kws.keys(), f"Unknown kwargs {kwargs.keys()-kws.keys()}"
    kws.update(kwargs)
    method = kws.pop('method', 'auto')
    if method in ('dense','exact') or (method == 'auto' and (np.prod(A.shape) <= 1e6 or kws['tol']<3e-2)):
        return slogdet(cola.decompositions.lu_decomposed(A), **kws)
    elif method in ('iterative', 'approx') or (method == 'auto' and (np.prod(A.shape) > 1e6 and kws['tol']>=3e-2)):
        raise NotImplementedError("Approximate logdet not implemented yet for non self adjoint matrices")
    else:
        raise ValueError(f"Unknown method {method} or CoLA didn't fit any selection criteria")

@dispatch(cond=lambda A: A.isa(PSD))
def slogdet(A: LinearOperator, **kwargs) -> Array:
    kws = dict(method="auto", tol=1e-2, pbar=False, max_iters=300)
    #assert not kwargs.keys() - kws.keys(), f"Unknown kwargs {kwargs.keys()-kws.keys()}"
    kws.update(kwargs)
    method = kws.pop('method', 'auto')
    if method in ('dense','exact') or (method == 'auto' and (np.prod(A.shape) <= 1e6 or kws['tol']<3e-2)):
        return slogdet(cola.decompositions.cholesky_decomposed(A), **kws)
    elif method in ('iterative', 'approx') or (method == 'auto' and (np.prod(A.shape) > 1e6 and kws['tol']>=3e-2)):
        return 1., stochastic_lanczos_quad(A, A.xnp.log, **kws)
    else:
        raise ValueError(f"Unknown method {method} or CoLA didn't fit any selection criteria")

@dispatch(cond = lambda A, **kwargs: all([(Ai.shape[-2] == Ai.shape[-1]) for Ai in A.Ms]))
def slogdet(A: Product, **kwargs) -> Array:
    signs, logdets = zip(*[slogdet(Ai) for Ai in A.Ms])
    # print(signs, logdets)
    # print([A.xnp.slogdet(Ai.to_dense()) for Ai in A.Ms])
    # print(A.xnp.slogdet(A.to_dense()))
    return product(signs), sum(logdets)

@dispatch
def slogdet(A: Diagonal, **kwargs) -> Array:
    xnp = A.xnp
    mag = xnp.abs(A.diag)
    phase = A.diag/mag
    return xnp.prod(phase), xnp.sum(xnp.log(mag))

@dispatch
def slogdet(A: Kronecker, **kwargs) -> Array:
    # logdet(Pi A_i \otimes I) = sum_i logdet(A_i)
    signs, logdets = zip(*[slogdet(Ai) for Ai in A.Ms])
    sizes = [Ai.shape[-1] for Ai in A.Ms]
    prod = product(sizes)
    scaled_logdets = [logdets[i] * prod/sizes[i] for i in range(len(sizes))]
    scaled_signs = [signs[i]**(prod/sizes[i]) for i in range(len(sizes))]
    return product(scaled_signs), sum(scaled_logdets)


@dispatch
def slogdet(A: BlockDiag, **kwargs) -> Array:
    # logdet(\bigoplus A_i) = log \prod det(A_i) = sum_i logdet(A_i)
    signs, logdets = zip(*[slogdet(Ai) for Ai in A.Ms])
    scaled_logdets = sum(l*n for l, n in zip(logdets, A.multiplicities))
    scaled_signs = product(s**n for s, n in zip(signs, A.multiplicities))
    return scaled_signs, scaled_logdets

@dispatch
def slogdet(A: Triangular, **kwargs) -> Array:
    xnp = A.xnp
    diag = xnp.diag(A.A)
    mag = xnp.abs(diag)
    phase = diag/mag
    return xnp.prod(phase), xnp.sum(xnp.log(mag))

@dispatch
def slogdet(A: Permutation, **kwargs) -> Array:
    # TODO: count the parity of the permutation and return an error if it is odd
    xnp = A.xnp
    zero = xnp.array(0., dtype=A.dtype, device=A.device)
    return 1.+zero, zero