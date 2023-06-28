import cola.torch_fns as xnp
from cola.operators import Diagonal
from cola.linalg.inverse import inverse

dtype = xnp.float32
diag = xnp.Parameter(xnp.array([3., 4., 5.], dtype=dtype))
D = Diagonal(diag)
rhs = xnp.randn(diag.shape[0], 1, dtype=dtype)
soln = inverse(D, method="cg") @ rhs
loss = xnp.norm(soln)
loss.backward()
print(diag.grad)
