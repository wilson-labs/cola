#%%
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
import jax.numpy as jnp
from jax import vmap, jit
import scipy as sp
import scipy
import jax

import matplotlib.pyplot as plt

N=400
ndims = 2

xgrid = jnp.linspace(-1,1,N)
xygrid = jnp.stack(jnp.meshgrid(*(ndims*[xgrid])),axis=-1)
dx = xgrid[1]-xgrid[0]

@jit
def laplacian(x):
    x = x.reshape(ndims*(N,))
    cderiv = lambda x: jax.scipy.signal.correlate(x,jnp.array([1.,-2,1.])/dx**2,mode='same')
    return -sum([jnp.apply_along_axis(cderiv,i,x) for i in range(ndims)]).reshape(-1)

x,y = xygrid.transpose((2,0,1))
domain = (((x>0)|(y>0))&((y>-.5)|((x-.5)**2+(y+.5)**2<.5**2)))
domain &= ~(scipy.signal.convolve2d((~domain).astype(np.float32),jnp.ones((3,3)),mode='same',fillvalue=1)>0)
plt.imshow(domain)
plt.show()
domain_ids = domain.reshape(-1).nonzero()[0]
# boundary = (scipy.signal.convolve2d((~domain).astype(np.float32),jnp.ones((3,3)),mode='same',fillvalue=1)>0)&domain
# boundary_vals = jnp.where(boundary,(x+y)*jnp.cos(2*x),jnp.zeros_like(x))
BCS = jnp.where(domain,jnp.zeros_like(x),(x+y)*jnp.cos(2*x))


from cola.ops import CustomLinOp, Symmetric, SelfAdjoint
from cola.linalg import solve_symmetric, solve
from cola import LinearOperator


Lfull = LinearOperator(jnp.float32,shape=(N**ndims,N**ndims),matmat=jit(vmap(laplacian,-1,-1)))
L = Lfull[domain_ids,domain_ids]
RHS = -(Lfull@BCS.reshape(-1))[domain_ids]

sol,info = solve(Symmetric(L),RHS,tol=1e-4,info=True,pbar=True)

#from scipy.sparse.linalg import LinearOperator, eigsh
# @jit
# def pde_op(u):
#     zero_grid = jnp.zeros(domain.shape).reshape(-1)
#     zero_grid = zero_grid.at[domain_ids].set(u.reshape(-1))
#     zero_grid = laplacian(zero_grid)
#     return zero_grid[domain_ids].reshape(u.shape)


#matmat = vmap(pde_op,-1,-1)

#L2 = LinearOperator(2*(domain.sum(),),matvec = pde_op,dtype=np.float32)

#sol2,info2 = scipy.sparse.linalg.cg(L2,boundary_vals[domain]/dx**2,tol=1e-4)

out_img = jnp.zeros(domain.shape)
out_img = out_img.at[domain].set(sol)

plt.imshow(out_img,cmap='twilight')
plt.colorbar()
plt.show()

plt.plot(info['errors'])
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.show()
# %%
