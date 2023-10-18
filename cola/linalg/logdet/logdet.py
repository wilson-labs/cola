import numpy as np
from functools import reduce
from plum import dispatch
from cola.annotations import PSD
from cola.ops.operators import LinearOperator, Triangular, Permutation, Identity, ScalarMul
from cola.ops.operators import Diagonal, Kronecker, BlockDiag, Product
from cola.utils import export
from cola.linalg.algorithm_base import Algorithm, Auto
from cola.linalg.decompositions.decompositions import Cholesky, LU, Arnoldi, Lanczos
from cola.linalg.decompositions.decompositions import plu, cholesky
from cola.linalg.trace.diag_trace import trace
from cola.linalg.unary.unary import log


def product(xs):
    return reduce(lambda x, y: x * y, xs, 1)


@export
def logdet(A: LinearOperator, log_alg: Algorithm = Auto(), trace_alg: Algorithm = Auto()):
    r""" Computes log determinant of a linear operator.

    For large inputs (or with method='iterative'),
    uses either :math:`O(\tfrac{1}{\delta^2}\log(1/\epsilon))` time stochastic algorithm (SLQ)
    where :math:`\epsilon=` tol is the bias and :math:`\delta=` vtol is the standard deviation of
    the estimate, or a deterministic :math:`O(n\log(1/\epsilon))` time algorithm if
    :math:`\delta < 1/\sqrt{10n}`.

    Args:
        A (LinearOperator): The linear operator to compute the logdet of.
        log_alg (Algorithm, optional): The algorithm to use for the log.
            Specify LU() or Cholesky() for a dense approach.
        trace_alg (Algorithm, optional): The algorithm to use for the trace computation.

    Returns:
        Array: logdet
    """
    _, ld = slogdet(A, log_alg=log_alg, trace_alg=trace_alg)
    return ld


@export
@dispatch.abstract
def slogdet(A: LinearOperator, log_alg: Algorithm = Auto(), trace_alg: Algorithm = Auto()):
    r""" Computes sign and logdet of a linear operator. such that det(A) = sign(A) exp(logdet(A))

    Args:
        A (LinearOperator): The linear operator to compute the logdet of.
        log_alg (Algorithm, optional): The algorithm to use for the log.
            Specify LU() or Cholesky() for a dense approach.
        trace_alg (Algorithm, optional): The algorithm to use for the trace computation.

    Returns:
        Tuple[Array, Array]: sign, logdet: logdet
    # """
    # Args:
    #     A (LinearOperator): The linear operator to compute the logdet of.
    #     tol (float, optional): Tolerance for the bias of the solution. Defaults to 1e-6.
    #     vtol (float, optional): Tolerance for the variance (std) of the solution,
    #      returns a stochastic estimate if large that saves considerable computation. Default: 1e-6
    #     pbar (bool, optional): Whether to show a progress bar. Defaults to False.
    #     max_iters (int, optional): The maximum number of iterations. Defaults to 300.
    #     method (str, optional): Method to use, defaults to 'auto',
    #      options are 'auto', 'dense', 'iterative', 'iterative-exact', 'iterative-stochastic'

    # Returns:
    #     Tuple[Array, Array]: sign, logdet
    # """


# ########### BASE CASES #############
@dispatch(precedence=-1)
def slogdet(A: LinearOperator, log_alg: Auto, trace_alg: Algorithm):
    is_PSD = A.isa(PSD)
    small = np.prod(A.shape) <= 1e6
    if is_PSD and small:
        log_alg = Cholesky()
    elif not is_PSD and small:
        log_alg = LU()
    elif is_PSD and not small:
        log_alg = Lanczos(**log_alg.__dict__)
    elif not is_PSD and not small:
        log_alg = Arnoldi(**log_alg.__dict__)
    return slogdet(A, log_alg=log_alg, trace_alg=trace_alg)


@dispatch(precedence=-1)
def slogdet(A: LinearOperator, log_alg: Cholesky, trace_alg: Algorithm):
    assert A.isa(PSD), "Cholesky only valid for PSD matrices, wrap in cola.PSD if necessary"
    L = cholesky(A)
    sign, logdet = slogdet(L, log_alg, trace_alg)
    return sign * A.xnp.conj(sign), 2 * logdet


@dispatch(precedence=-1)
def slogdet(A: LinearOperator, log_alg: LU, trace_alg: Algorithm):
    P, L, U = plu(A)
    return slogdet(P @ L @ U, log_alg, trace_alg)


@dispatch(precedence=-1)
def slogdet(A: LinearOperator, log_alg: Lanczos | Arnoldi, trace_alg: Algorithm):
    logA = log(A, log_alg)
    trlogA = trace(logA, trace_alg)
    mag = A.xnp.abs(trlogA)
    phase = trlogA / mag
    return phase, mag


# ############ DISPATCH RULES ############
@dispatch(cond=lambda A, *_: all([(Ai.shape[-2] == Ai.shape[-1]) for Ai in A.Ms]))
def slogdet(A: Product, log_alg: Algorithm, trace_alg: Algorithm):
    signs, logdets = zip(*[slogdet(Ai, log_alg, trace_alg) for Ai in A.Ms])
    return product(signs), sum(logdets)


@dispatch
def slogdet(A: Identity, log_alg: Algorithm, trace_alg: Algorithm):
    xnp = A.xnp
    zero = xnp.array(0., dtype=A.dtype, device=A.device)
    return 1. + zero, zero


@dispatch
def slogdet(A: ScalarMul, log_alg: Algorithm, trace_alg: Algorithm):
    xnp = A.xnp
    c = A.c
    phase = c / xnp.abs(c)
    return phase, xnp.log(xnp.abs(c))


@dispatch
def slogdet(A: Diagonal, log_alg: Algorithm, trace_alg: Algorithm):
    xnp = A.xnp
    mag = xnp.abs(A.diag)
    phase = A.diag / mag
    return xnp.prod(phase), xnp.sum(xnp.log(mag))


@dispatch
def slogdet(A: Kronecker, log_alg: Algorithm, trace_alg: Algorithm):
    # logdet(Pi A_i \otimes I) = sum_i logdet(A_i)
    signs, logdets = zip(*[slogdet(Ai, log_alg, trace_alg) for Ai in A.Ms])
    sizes = [Ai.shape[-1] for Ai in A.Ms]
    prod = product(sizes)
    scaled_logdets = [logdets[i] * prod / sizes[i] for i in range(len(sizes))]
    scaled_signs = [signs[i]**(prod / sizes[i]) for i in range(len(sizes))]
    return product(scaled_signs), sum(scaled_logdets)


@dispatch
def slogdet(A: BlockDiag, log_alg: Algorithm, trace_alg: Algorithm):
    # logdet(\bigoplus A_i) = log \prod det(A_i) = sum_i logdet(A_i)
    signs, logdets = zip(*[slogdet(Ai, log_alg, trace_alg) for Ai in A.Ms])
    scaled_logdets = sum(ld * n for ld, n in zip(logdets, A.multiplicities))
    scaled_signs = product(s**n for s, n in zip(signs, A.multiplicities))
    return scaled_signs, scaled_logdets


@dispatch
def slogdet(A: Triangular, log_alg: Algorithm, trace_alg: Algorithm):
    xnp = A.xnp
    diag = xnp.diag(A.A)
    mag = xnp.abs(diag)
    phase = diag / mag
    return xnp.prod(phase), xnp.sum(xnp.log(mag))


@dispatch
def slogdet(A: Permutation, log_alg: Algorithm, trace_alg: Algorithm):
    # TODO: count the parity of the permutation and return an error if it is odd
    xnp = A.xnp
    zero = xnp.array(0., dtype=A.dtype, device=A.device)
    return 1. + zero, zero
