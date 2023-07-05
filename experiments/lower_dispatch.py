import cola.torch_fns as xnp
from cola.linear_algebra import lazify
from cola.ops import LowerTriangular
from cola.linalg.eigs import eig
from cola.utils_test import generate_lower_from_diag

dtype = xnp.float32
diag = xnp.array([1., 0.5, 0.25, 0.1], dtype=dtype)
L = lazify(xnp.array(generate_lower_from_diag(diag), dtype=dtype))
L = LowerTriangular(L)
eig_vals, eig_vecs = eig(L)
print(xnp.norm(eig_vals - diag))
