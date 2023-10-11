import numpy as np
from cola.utils import export
from cola.ops import I_like, LinearOperator
from cola.linalg.algorithm_base import Algorithm
from dataclasses import dataclass


@export
@dataclass
class Exact(Algorithm):
    """
    Exact algorithm to compute, element-by-element, the diagonal of the matrix.
    That is, each element of the diagoinal :math:`A_{i,i}` is computed as
    :math:`e_{i}^{*} A(e_{i})`.
    For efficiency, this procedure is done through blocks of elements,
    :math:`A [e_{i_{1}}, \\cdots, e_{i_{B}}]` where :math:`B` is the block size.

    Args:
        bs (int, optional): Block size.
        pbar (bool, optional): Whether to show progress bar.
    """
    bs: int = 100
    pbar: bool = False

    def __call__(self, A, k):
        return exact_diag(A, k, self.bs)


@export
@dataclass
class Hutch(Algorithm):
    """
    Hutchinson's algorithm for estimating the trace of a matrix function.
    Basically, this algorithm uses random probes to approximate
    :math:`\\text{tr}(f(A))`.

    Args:
        tol (float, optional): Approximation relative tolerance.
        max_iters (int, optional): The maximum number of iterations to run.
        bs (int, optional): Number of probes.
        rand (str, optional): type of random probes (either Normal or Rademacher)
        pbar (bool, optional): Whether to show progress bar.
    """
    tol: float = 3e-2
    max_iters: int = 10_000
    bs: int = 100
    rand: str = 'normal'
    pbar: bool = False

    def __call__(self, A, k):
        return hutchinson_diag_estimate(A, k, **self.__dict__)[0]


@export
@dataclass
class HutchPP(Algorithm):
    """
    Hutch++ is an improvement on the Hutchinson's estimator introduced in
    Meyer et al. 2020: Hutch++: Optimal Stochastic Trace Estimator.

    Args:
        tol (float, optional): Approximation relative tolerance.
        max_iters (int, optional): The maximum number of iterations to run.
        bs (int, optional): Number of probes.
        rand (str, optional): type of random probes (either Normal or Rademacher)
        pbar (bool, optional): Whether to show progress bar.
    """
    tol: float = 3e-2
    max_iters: int = 10000
    bs: int = 100
    rand: str = 'normal'
    pbar: bool = False

    def __call__(self, A, k):
        raise NotImplementedError


def get_I_chunk_like(A: LinearOperator, i, bs, shift=0):
    xnp = A.xnp
    k = shift
    Id = I_like(A)
    if k == 0:
        I_chunk = Id[:, i:i + bs].to_dense()
        chunk = I_chunk
        shifted_chunk = I_chunk
    elif k <= 0:
        k = abs(k)
        I_chunk = Id[:, i:i + bs + k].to_dense()
        padded_chunk = A.xnp.zeros((A.shape[0], bs + k), dtype=A.dtype, device=A.device)
        slc = np.s_[:I_chunk.shape[-1]]
        padded_chunk = xnp.update_array(padded_chunk, I_chunk, slice(0, None), slc)
        chunk = I_chunk[:, :bs]
        shifted_chunk = padded_chunk[:, k:k + bs]
    else:
        I_chunk = Id[:, max(i - k, 0):i + bs].to_dense()
        padded_chunk = A.xnp.zeros((A.shape[0], bs + k), dtype=A.dtype, device=A.device)
        slc = np.s_[-I_chunk.shape[-1]:]
        padded_chunk = xnp.update_array(padded_chunk, I_chunk, slice(0, None), slc)
        chunk = I_chunk[:, -bs:]
        shifted_chunk = padded_chunk[:, :bs]
    return chunk, shifted_chunk


# disable backwards for now, TODO: add tests then add back in
# @iterative_autograd(exact_diag_bwd)


def exact_diag(A, k, bs):
    bs = min(100, A.shape[0])
    # lazily create chunks of the identity matrix of size bs
    diag_sum = 0.
    for i in range(0, A.shape[0], bs):
        chunk, shifted_chunk = get_I_chunk_like(A, i, bs, k)
        diag_sum += ((A @ chunk) * shifted_chunk).sum(-1)
    if k <= 0:
        out = diag_sum[abs(k):]
    else:
        out = diag_sum[:(-k or None)]
    return out


def exact_diag_bwd(res, grads, unflatten, *args, **kwargs):
    v = grads[0] if isinstance(grads, (tuple, list)) else grads
    op_args, _ = res
    A = unflatten(op_args)
    xnp = A.xnp
    k = kwargs.get('k')
    bs = kwargs.get('bs')

    def fun(theta, C, shifted_C):
        Aop = unflatten(theta)
        out = ((Aop @ C) * shifted_C).sum(-1)
        return out[abs(k):] if k <= 0 else out[:(-k or None)]

    def d_params(C, shifted_C):
        d_paramsv, _, _ = xnp.vjp_derivs(fun, (op_args, C, shifted_C), v)
        return d_paramsv

    d_p = type(op_args)([0. * arg for arg in op_args])
    for i in range(0, A.shape[0], bs):
        chunk, shifted_chunk = get_I_chunk_like(A, i, bs, k)
        dp_all = d_params(chunk, shifted_chunk)
        for i in range(len(d_p)):
            d_p[i] += dp_all[i]
    dA = unflatten(d_p)
    return (dA, )


@export
def hutchinson_diag_estimate(A: LinearOperator, k=0, bs=100, tol=3e-2, max_iters=10000, pbar=False, rand='normal'):
    """ Extract the (kth) diagonal of a linear operator using stochastic estimation

    Args:
        A (LinearOperator): Linear operator.
        k (int, optional): Index of the diagonal to extract (default 0).
        bs (int, optional): Chunk size (default 100).
        tol (float, optional): Tolerance (default 3e-2).
        max_iters (int, optional): Maximum number of iterations (default 10000).
        pbar (bool, optional): Flag for showing progress bar.

    Returns:
        Array: Extracted diagonal elements.
        Info: Dictionary with information about the method used.
    """
    bs = min(100, A.shape[0])
    # lazily create chunks of the identity matrix of size bs
    xnp = A.xnp
    assert tol > 1e-3, "tolerance chosen too high for stochastic diagonal estimation"
    assert rand in ['normal', 'rademacher'], "rand must be 'normal' or 'rademacher'"

    @xnp.jit
    def body(state):
        # TODO: fix randomness when using with JAX
        i, diag_sum, diag_sumsq, key = state
        key = xnp.next_key(key)
        z = xnp.randn(A.shape[0], bs, dtype=A.dtype, key=key, device=A.device)
        if rand == 'rademacher':
            z = xnp.sign(z)
        z2 = xnp.roll(z, -k, 0)
        z2 = xnp.update_array(z2, 0, slice(0, abs(k)) if k <= 0 else slice(-abs(k), None))
        slc = slice(abs(k), None) if -k > 0 else slice(None, -abs(k) or None)
        estimator = ((A @ z) * z2)[slc]
        return i + 1, diag_sum + estimator.sum(-1), diag_sumsq + (estimator**2).sum(-1), key

    def err(state):
        i, diag_sum, diag_sumsq, _ = state
        mean = diag_sum / (i * bs)
        stderr = xnp.sqrt((diag_sumsq / (i * bs) - mean**2) / (i * bs))
        return xnp.mean(stderr / xnp.maximum(xnp.abs(mean), .1 * xnp.ones_like(mean)))

    def cond(state):
        return (state[0] == 0) | ((state[0] < max_iters) & (err(state) > tol))

    while_loop, infos = xnp.while_loop_winfo(err, tol, max_iters, pbar=pbar)
    zeros = xnp.zeros((A.shape[0] - abs(k), ), dtype=A.dtype, device=A.device)
    n, diag_sum, *_ = while_loop(cond, body, (0, zeros, zeros, xnp.PRNGKey(42)))
    mean = diag_sum / (n * bs)
    return mean, infos
