import os
import time
import numpy as np
import jax.numpy as jnp
from jax import vmap, jit
import jax
from jax import jacfwd
import linops as lo
from linops.linalg.eigs import eig
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# from jax.config import config
# config.update("jax_enable_x64", True)
# jax.config.update('jax_platform_name', 'cpu')

tic = time.time()


def square_compactification(x):
    return jnp.arctan(x) * 2 / jnp.pi


def inv_square_compactification(y):
    return jnp.tan(y * jnp.pi / 2)


# define the hydrogen atom Hamiltonian transformed coordinates
N = 1_000
ndims = 1

grid = jnp.linspace(-1 + .001, 1 - .001, N)  # convert to 3d with mesgrid

# grid = jnp.linspace(-1,1,N)
# grid = jnp.linspace(-40,40,N)
print(grid.dtype)
wgrid = jnp.stack(jnp.meshgrid(*(ndims * [grid])), axis=-1).reshape(-1, ndims)
idd = lambda x: x

T = square_compactification  #idd#lambda x: jax.scipy.stats.norm.cdf(norm(x)/3)*x/norm(x)#jnp.log(1e-1+norm(x))*x/norm(x)#radial_hyperbolic_compactification
Tinv = inv_square_compactification  #idd#lambda x: 3*jax.scipy.special.ndtri(norm(x))*x/norm(x)#(jnp.exp(norm(x))-1e-1)*x/norm(x)#inv_radial_hyperbolic_compactification
xyz = vmap(Tinv)(wgrid)
print(xyz[0], xyz[-1])
DT = vmap(jacfwd(T))(xyz)  # (b, 3-out, 3-in)
laplacian_factor2 = DT @ DT.transpose((0, 2, 1))
laplacian_factor1 = vmap(lambda z: (jacfwd(jacfwd(T))(z) * jnp.eye(ndims)[None, :, :]).sum((1, 2)))(
    xyz)
dw = grid[1] - grid[0]
deriv = jnp.array([-1 / 2, 0., 1 / 2]) / dw


# deriv = jnp.array([-1.,1.])/dw
# di = lambda x,i: sp.ndimage.correlate1d(x,deriv,axis=i,mode='constant')
def hdiag(x):
    cderiv = lambda x: jax.scipy.signal.correlate(x, jnp.array([1., -2, 1.]) / dw**2, mode='same')
    dds = jnp.stack([jnp.apply_along_axis(cderiv, i, x).reshape(-1) for i in range(ndims)], axis=0)
    embedded_diag = vmap(jnp.diag, -1, -1)(dds).transpose((2, 0, 1))
    return embedded_diag


jderiv = lambda x: jax.scipy.signal.correlate(x, deriv, mode='same')  # BCS?
di = lambda x, i: jnp.apply_along_axis(jderiv, i, x)
d = lambda x, axis=-1: jnp.stack([di(x, i) for i in range(ndims)], axis=axis)


# lap= lambda x: scipy.ndimage.laplace(x,mode='constant')/dw**2
def lap(x):
    cderiv = lambda x: jax.scipy.signal.correlate(x, jnp.array([1., -2, 1.]) / dw**2, mode='same')
    return sum([jnp.apply_along_axis(cderiv, i, x).reshape(-1) for i in range(ndims)])


def vfn(x):
    return (x * x).sum() / 2


# def vfn(x):
#     return (x*x).sum()/2


@jit
def laplacian(psi):
    psi_grid = psi.reshape(*(ndims * (N, )))
    #     return lap(psi_grid).reshape(psi.shape)
    # return out

    # return (hessian*jnp.eye(ndims)[None]).sum((1,2)).reshape(psi.shape)
    dpsi = d(psi_grid)
    hessian = d(dpsi).reshape(-1, ndims, ndims)
    hessian = jnp.where(jnp.eye(ndims)[None] + 0 * hessian > 0.5, hdiag(psi_grid), hessian)
    l1 = (dpsi.reshape(-1, ndims) * laplacian_factor1).sum(-1)
    l2 = (hessian * laplacian_factor2).sum((1, 2))
    return (l1 + l2).reshape(psi.shape)


L = lo.LinearOperator(jnp.float64, shape=(N**ndims, N**ndims), matmat=jit(vmap(laplacian, -1, -1)))
v = vmap(vfn)(xyz).reshape(-1)
V = lo.diag(v)
H = -L / 2 + V

# @jit
# def H2(psi):
#     KE = -laplacian(psi)/2
#     V = psi*vmap(vfn)(xyz).reshape(psi.shape)
#     return KE+V

# def Hnp(psi):
#     return np.asarray(H2(psi))

# from scipy.sparse.linalg import LinearOperator, eigsh, eigs
# matmat = vmap(H2,-1,-1)

# Hop = LinearOperator(2*(wgrid.shape[0],), matvec = H2,
#                      rmatvec = H2,matmat=matmat,rmatmat=matmat,dtype=np.float32)
# #e,v = eigs(Hop,k=10,which='SR')

# hh = Hop@np.eye(Hop.shape[0])
# e,v = np.linalg.eig(hh)
# order = np.argsort(e)
# e=e[order]
# v=v[:,order]
# k=20
# print(e[:k])
k = 20
# e1,v1 = lo.linalg.eig(H,method='dense')
e2, v2, _ = eig(H, method='arnoldi', max_iters=int(N * 1.))
# e2, v2 = lo.linalg.eig(H, method="dense")
# order = np.argsort(e)
# print("dense gives",np.sort(np.abs(e1))[:k])
print("arnoldi gives", np.sort(np.abs(e2))[:k])
toc = time.time()
print(f"Took {toc - tic:0.2f} sec")
