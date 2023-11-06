from cola import Stiefel
from cola.ops import Dense


def shifted_qr(A, shifts):
    xnp = A.xnp
    dtype, device = A.dtype, A.device
    Adense = A.to_dense()
    Id = xnp.eye(*A.shape, dtype=dtype, device=device)
    max_iters = shifts.shape[0]

    def body_fun(idx, state):
        H, V = state
        Q, R = xnp.qr(H - shifts[idx] * Id, full_matrices=True)
        H = R @ Q + shifts[idx] * Id
        V @= Q
        return H, V

    init_val = (Adense, xnp.eye(*A.shape, dtype=dtype, device=device))
    H, V = xnp.for_loop(0, max_iters, body_fun, init_val)
    return Dense(H), Stiefel(Dense(V))
