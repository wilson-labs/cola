from cola.utils import export
from cola.ops import LinearOperator

@export
def power_iteration(A: LinearOperator, tol=1e-7, max_iter=1000, pbar=False, info=False,
                    momentum=None):
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
    *_, emax, _ = while_loop(cond, body, (0, v, v, 10., 1.))
    return (emax, infos) if info else emax
