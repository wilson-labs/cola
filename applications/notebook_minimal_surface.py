import time
import matplotlib.pyplot as plt
from cola.operator_base import CustomLinOp
from cola.experiment_utils import print_time_taken
from functools import partial
from cola.linalg.inverse import inverse
import jax
from jax import vmap, jit
import jax.numpy as jnp
import numpy as np
import os
# from scipy.sparse.linalg import LinearOperator
# import scipy
# import cola as lo

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

N = 50
xgrid = jnp.linspace(-1, 1, N)
xygrid = jnp.stack(jnp.meshgrid(xgrid, xgrid), axis=-1)
dx = xgrid[1] - xgrid[0]


@jit
def pde_deriv(z):
    def deriv(x):
        return jax.scipy.signal.correlate(x, jnp.array([-1 / 2, 0, 1 / 2]) / dx, mode='same')

    def deriv2(x):
        return jax.scipy.signal.correlate(x, jnp.array([1., -2, 1.]) / dx**2, mode='same')

    zx, zy = [jnp.apply_along_axis(deriv, i, z) for i in [0, 1]]
    zxx, zyy = [jnp.apply_along_axis(deriv2, i, z) for i in [0, 1]]
    zxy = jnp.apply_along_axis(deriv, 1, zx)
    return (1 + zx**2) * zyy - 2 * zx * zy * zxy + (1 + zy**2) * zxx


x, y = xygrid.transpose((2, 0, 1))
domain = (jnp.abs(x) < 1) & (jnp.abs(y) < 1)
boundary_vals = np.zeros_like(x)
boundary_vals[:, 0] = 1 - y[:, 0]**2
boundary_vals[:, -1] = 1 - y[:, -1]**2


@jit
def pde_op(u):
    padded_domain = jnp.zeros(boundary_vals.shape) + boundary_vals
    padded_domain = padded_domain.at[domain].set(u.reshape(-1))
    padded_domain = pde_deriv(padded_domain)
    return padded_domain[domain].reshape(u.shape)


@jit
def J_matvec(u, v):
    return jax.jvp(pde_op, (u, ), (v, ))[1]


# Newton Raphson iteration
tic = time.time()
z = jnp.zeros_like(x[domain]).reshape(-1)
tol = 1e-3
err = np.inf
while err > tol:
    Jmvm = partial(J_matvec, z)
    F = pde_op(z)
    err = jnp.max(jnp.abs(F))
    shape = (int(domain.sum()), int(domain.sum()))

    # J = LinearOperator(shape, matvec=Jmvm, matmat=jit(vmap(Jmvm, -1, -1)), dtype=np.float32)
    # delta, info = scipy.sparse.linalg.gmres(J, -F, tol=1e-5)

    # J = lo.Jacobian(pde_op,z)
    J = CustomLinOp(dtype=jnp.float32, shape=shape, matmat=jit(vmap(Jmvm, -1, -1)))
    J_inv = inverse(J, tol=tol, max_iters=F.shape[0] // 2, method="gmres", info=True)
    delta = J_inv @ -F
    info = J_inv.info

    z += delta
    print(f"PDE Error: {err:1.1e}, Update size: {jnp.linalg.norm(delta):1.1e}, info: {info}")
toc = time.time()
print_time_taken(toc - tic)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
img = jnp.zeros(boundary_vals.shape) + boundary_vals
img = img.at[domain].set(z)
ax.plot_surface(x, y, img, cmap=plt.cm.YlGnBu_r)
plt.show()

plt.imshow(img, cmap='twilight')
plt.colorbar()
plt.show()
