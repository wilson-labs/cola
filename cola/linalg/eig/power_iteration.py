from cola.utils import export
from cola.ops import LinearOperator
from cola.linalg.algorithm_base import Algorithm
from dataclasses import dataclass


@export
@dataclass
class PowerIteration(Algorithm):
    """ Simple power iteration algorithm for finding the largest eigenvalue and eigenvector.

    Args:
        tol (float, optional): Relative error tolerance.
        max_iters (int, optional): The maximum number of iterations to run.
        pbar (bool, optional): Whether to show progress bar.

    Example:
        >>> A = MyLinearOperator()
        >>> v, eigmax, info = cola.linalg.eig.PowerIteration(tol=1e-3)(A)
    """
    tol: float = 1e-06
    max_iter: int = 100
    pbar: bool = False

    def __call__(self, A: LinearOperator):
        return power_iteration(A, tol=self.tol, max_iter=self.max_iter, pbar=self.pbar)


def power_iteration(A: LinearOperator, tol=1e-6, max_iter=1000, pbar=False, momentum=None):
    """
    Performs power iteration to compute the dominant eigenvector and eigenvalue
    of the operator.

    Args:
        A (LinearOperator): A linear operator of size (n, n).
        tol (float, optional): Stopping criteria.
        max_iters (int, optional): The maximum number of iterations to run.
        pbar (bool, optional): Whether to show a progress bar. Defaults to False.

    Returns:
        tuple:
            - v (Array): dominant eigenvector (n,).
            - eigmax (Array): dominant eigenvalue (1,).
            - info (dict): General information about the iterative procedure.
    """
    xnp = A.xnp
    v = xnp.randn(*A.shape[-1:], dtype=A.dtype, device=A.device, key=xnp.PRNGKey(0))

    @xnp.jit
    def body(state):
        i, v, vprev, eig, eigprev = state
        p = A @ v
        eig, eigprev = v @ p, eig
        if momentum is not None:
            # estimate_optimal_momentum(eig, eigprev, p @ p)
            p = p - momentum * vprev
        return i + 1, p / xnp.norm(p), v, eig, eigprev

    def err(state):
        *_, eig, eigprev = state
        return abs(eigprev - eig) / eig

    def cond(state):
        i = state[0]
        return (i < max_iter) & (err(state) > tol)

    while_loop, infodict = xnp.while_loop_winfo(err, tol, pbar=pbar)
    _, v, _, emax, _ = while_loop(cond, body, (0, v, v, 10., 1.))
    return v, emax, infodict
