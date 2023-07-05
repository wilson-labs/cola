import cola.torch_fns as xnp
# import cola.jax_fns as xnp
from cola.ops import Symmetric
from cola.linear_algebra import lazify
from cola.linalg.inverse import inverse
from cola.algorithms.preconditioners import NystromPrecond
from jax.config import config

config.update('jax_platform_name', 'cpu')

N, B = 10, 3
L = lazify(xnp.randn(N, N, dtype=xnp.float32))
A = Symmetric(L.T @ L)
P = NystromPrecond(A, rank=A.shape[0] // 2)
rhs = xnp.randn(N, B, dtype=xnp.float32)
A_inv = inverse(A, method='cg', P=P)
soln = A_inv @ rhs
res = xnp.norm(A @ soln - rhs, axis=0)
print(res)
