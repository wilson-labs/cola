from cola import LinearOperator
from cola.linalg.inverse import inverse
from cola.ops import Symmetric
import matplotlib.pyplot as plt
from jax.config import config
import jax
from jax import vmap, jit
import jax.numpy as jnp
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
config.update("jax_enable_x64", True)

N = 151
ndims = 2

xgrid = jnp.linspace(-1, 1, N)[1:-1]
N = len(xgrid)
xygrid = jnp.stack(jnp.meshgrid(*(ndims * [xgrid])), axis=-1)
dx = xgrid[1] - xgrid[0]


def pad_axis(z, axis):
    # pad = [(0, 0)] * ndims
    # pad[axis] = (1, 1)
    # return jnp.pad(z, pad, mode='wrap')
    return z


def laplacian(x):
    z = x.reshape(ndims * (N, ))

    def cderiv(a):
        return jax.scipy.signal.correlate(a, jnp.array([1., -2, 1.]) / dx**2, mode='same')

    return -sum([jnp.apply_along_axis(cderiv, i, pad_axis(z, i)) for i in range(ndims)]).reshape(-1)


# order=2
x, y = xygrid.transpose((2, 0, 1))
# domain = (((x>0)|(y>0))&((y>-.5)|((x-.5)**2+(y+.5)**2<.5**2)))
# domain &= ~(scipy.signal.convolve2d((~domain).astype(np.float32), jnp.ones(
#     (order + 1, order + 1)), mode='same', fillvalue=1) > 0)
# plt.imshow(domain)
# plt.show()
# domain_ids = domain.reshape(-1).nonzero()[0]
# boundary = (scipy.signal.convolve2d((~domain).astype(np.float32), jnp.ones(
#     (3, 3)), mode='same', fillvalue=1) > 0) & domain
# # boundary_vals = jnp.where(boundary,(x+y)*jnp.cos(2*x),jnp.zeros_like(x))
# BCS = jnp.where(domain,jnp.zeros_like(x),(x+y)*jnp.cos(2*x))
rho = ((1 - x**2) * (1 - y**2)) * ((x + y) * jnp.cos(4 * x) - 2 * x * y * jnp.sin(4 * x))
# rho = 0*x
# rho = rho.at[N//2:N//2+2,N//2:N//2+2].set(1/dx**2)
rho = rho.reshape(-1)

matmat = jit(vmap(laplacian, -1, -1))
Lfull = LinearOperator(jnp.float64, shape=(N**ndims, N**ndims), matmat=matmat)
L = Symmetric(Lfull)

print(type(L @ L))
inv = inverse(L @ L, tol=1e-5, info=True, pbar=False, method='cg')
sol = inv @ rho
infoa = inv.Ms[0].info
infob = inv.Ms[1].info
inv2 = inverse(Symmetric(L @ L), tol=1e-5, info=True, pbar=False, method='cg')
sol2 = inv2 @ rho
info2 = inv2.info
# sol2,info2= solve(Symmetric(L@L),rho,tol=1e-5,info=True,pbar=True,method='cg')

# from scipy.sparse.linalg import LinearOperator
# count = [0]
# def laplacian_wlog(x):
#     count[0]+=1
#     return laplacian(x)
# L2 = LinearOperator((N**ndims,N**ndims),matvec = laplacian_wlog,dtype=np.float32)
# sol2, info2 = scipy.sparse.linalg.cg(L2@L2,rho,tol=1e-4)
# @jit
# def pde_op(u):
#     zero_grid = jnp.zeros(domain.shape).reshape(-1)
#     zero_grid = zero_grid.at[domain_ids].set(u.reshape(-1))
#     zero_grid = laplacian(zero_grid)
#     return zero_grid[domain_ids].reshape(u.shape)

# matmat = vmap(pde_op,-1,-1)

# L2 = LinearOperator(2*(domain.sum(),),matvec = pde_op,dtype=np.float32)

# sol2,info2 = scipy.sparse.linalg.cg(L2,boundary_vals[domain]/dx**2,tol=1e-4)

out_img = sol.reshape(N, N)  # jnp.zeros(domain.shape)
# out_img = out_img.at[domain].set(sol)

# plt.imshow(sol2.reshape(N,N),cmap='twilight')
# plt.colorbar()
# plt.show()
# plt.imshow(rho.reshape(N,N),cmap='twilight')
# plt.colorbar()
# plt.show()

# plt.imshow(out_img,cmap='twilight')
# plt.colorbar()
# plt.show()

# plt.imshow(out_img-sol2.reshape(N,N),cmap='twilight')
# plt.colorbar()
# plt.show()
# plt.imshow(rho.reshape(N,N),cmap='twilight')
# plt.colorbar()
# plt.show()

ea = infoa['errors']
eb = infob['errors']

plt.plot(np.arange(len(ea)), ea, label='split')
plt.plot(np.arange(len(eb)) + len(ea), infob['errors'], label='split')
plt.plot(info2['errors'], label='combined')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('BiPoisson Equation, Dispatch vs Iterative')
plt.legend()
plt.show()
