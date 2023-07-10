from cola.utils import export
from cola.ops import LinearOperator


@export
def power_iteration(A: LinearOperator, tol=1e-7, max_iter=1000, pbar=False, info=False,
                    momentum=None):
    """Performs power iteration to compute the dominant eigenvector and eigenvalue of a matrix A.

    Args:
        A (LinearOperator): The linear operator representing the matrix A.
        tol (float, optional): The tolerance criteria for convergence. Defaults to 1e-7.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        pbar (bool, optional): Whether to show a progress bar. Defaults to False.
        info (bool, optional): Whether to return additional information. Defaults to False.
        momentum (float, optional): The momentum parameter for accelerated power iteration. Defaults to None.

    Returns:
        Tuple or Array: If info is False, returns a tuple (v, emax).
            If info is True, returns (v, emax, infodict).
    """
    xnp = A.ops
    v = xnp.fixed_normal_samples(A.shape[-1:], A.dtype)

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

    while_loop, infos = xnp.while_loop_winfo(err, tol, pbar=pbar)
    _, v, _, emax, _ = while_loop(cond, body, (0, v, v, 10., 1.))
    return (v, emax, infos) if info else (v, emax)
