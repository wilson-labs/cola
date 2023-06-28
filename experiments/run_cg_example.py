import linops.torch_fns as xnp
# import linops.jax_fns as xnp
from linops.operators import Symmetric
from linops.linear_algebra import lazify
from linops.linalg.inverse import inverse
from linops.algorithms.preconditioners import NystromPrecond
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
