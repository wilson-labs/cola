from cola.utils import export
from cola.ops import LinearOperator


@export
def power_iteration(A: LinearOperator, tol=1e-7, max_iter=1000, pbar=False, momentum=None):
    """
    Performs power iteration to compute the dominant eigenvector and eigenvalue
    of the operator.

    Args:
        A (LinearOperator): A linear operator of size (n, n).
        tol (float, optional): Stopping criteria.
        max_iters (int, optional): The maximum number of iterations to run.
        pbar (bool, optional): Whether to show a progress bar. Defaults to False.
        pbar (bool, optional): Show a progress bar.

    Returns:
        tuple:
            - v (Array): dominant eigenvector (n,).
            - eigmax (Array): dominant eigenvalue (1,).
            - info (dict): General information about the iterative procedure.
    """
    xnp = A.xnp
    v = xnp.fixed_normal_samples(A.shape[-1:], A.dtype, device=A.device)

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
