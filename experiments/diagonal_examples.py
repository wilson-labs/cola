import linops.torch_fns as xnp
from linops.operators import Diagonal
from linops.linalg.inverse import inverse
from linops.linalg.eigs import eig

dtype = xnp.float32
diag = xnp.array([1., 0.5, 0.25, 0.1], dtype=dtype)
D = Diagonal(diag)
rhs = xnp.randn(diag.shape[0], 1, dtype=dtype)
soln = inverse(D) @ rhs
res = xnp.norm(D @ soln - rhs, axis=0)
print(res)

eigvals, eigvecs = eig(D)
print(eigvals)
