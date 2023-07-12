from jax.scipy.linalg import block_diag
from jax.random import PRNGKey
from jax.random import normal
from jax import numpy as jnp
from jax.lax import while_loop as _while_loop
from cola.utils.control_flow import while_loop as _while_loop_no_jit
from jax.lax import fori_loop as _for_loop
# from cola.utils.control_flow import for_loop as _for_loop
from jax.lax import conj as conj_lax
from jax.lax import dynamic_slice
# from jax.lax import dynamic_update_slice
from jax.lax import expand_dims
from jax import vjp
from jax import jit, vmap, grad
import jax
from cola.utils.jax_tqdm import pbar_while, while_loop_winfo
from jax.lax.linalg import cholesky
from jax.lax.linalg import svd
from jax.lax.linalg import qr
from jax.scipy.linalg import solve_triangular as solvetri
import numpy as np
import logging

cos = jnp.cos
sin = jnp.sin
exp = jnp.exp
ndarray = jnp.ndarray
arange = jnp.arange
ones_like = jnp.ones_like
sign = jnp.sign
any = jnp.any
stack = jnp.stack
norm = jnp.linalg.norm
inv = jnp.linalg.inv
log = jnp.log
sum = jnp.sum
abs = jnp.abs
where = jnp.where
all = jnp.all
mean = jnp.mean
int32 = jnp.int32
int64 = jnp.int64
float32 = jnp.float32
float64 = jnp.float64
complex64 = jnp.complex64
long = jnp.int64
reshape = jnp.reshape
kron = jnp.kron
eye = jnp.eye
moveaxis = jnp.moveaxis
concatenate = jnp.concatenate
block_diag = block_diag
sqrt = jnp.sqrt
pbar_while = pbar_while
while_loop_winfo = while_loop_winfo
eig = jnp.linalg.eig
eigh = jnp.linalg.eigh
solve = jnp.linalg.solve
sort = jnp.sort
argsort = jnp.argsort
jit = jit
copy = jnp.copy
nan_to_num = jnp.nan_to_num
# randn = np.random.randn  # a little dangerous..
dynamic_slice = dynamic_slice
zeros_like = jnp.zeros_like
svd = svd
cholesky = cholesky
solvetri = solvetri
qr = qr
clip = jnp.clip
while_loop = _while_loop
while_loop_no_jit = _while_loop_no_jit
for_loop = _for_loop
min = jnp.min
max = jnp.max
ones = jnp.ones
concat = jnp.concatenate
vmap = vmap
grad = grad
roll = jnp.roll
maximum = jnp.maximum
PRNGKey = PRNGKey
isreal = jnp.isreal
allclose = jnp.allclose
# convolve = jax.scipy.signal.convolve

def get_default_device():
    return jax.devices()[0]

def device(device_name):
    del device_name
    zeros = jnp.zeros(1)
    return zeros.device


def diag(v, diagonal=0):
    return jnp.diag(v, k=diagonal)


def conj(array):
    if not jnp.iscomplexobj(array):
        return array
    else:
        return conj_lax(array)


def Parameter(array):
    return array


def cast(array, dtype):
    return array.astype(dtype)


def is_array(array):
    return isinstance(array, jnp.ndarray)


def convolve(in1, in2, mode='same'):
    in12 = jnp.pad(in1, ((in2.shape[0] // 2, (in2.shape[0] + 1) // 2 - 1),
                         (in2.shape[1] // 2, (in2.shape[1] + 1) // 2 - 1)), 'symmetric')
    out = jax.scipy.signal.convolve2d(in12, in2, mode='valid')
    return out  # ,boundary='symm')


def canonical(loc, shape, dtype):
    vec = jnp.zeros(shape=shape, dtype=dtype)
    vec = vec.at[loc].set(1.)
    return vec


def permute(array, axes):
    return jnp.transpose(array, axes=axes)


def expand(array, axis):
    return expand_dims(array, dimensions=(axis, ))


def randn(*shape, dtype=None, key=None):
    if key is None:
        print('Non keyed randn used. To be deprecated soon.')
        logging.warning('Non keyed randn used. To be deprecated soon.')
        out = np.random.randn(*shape)
        if dtype is not None:
            out = out.astype(dtype)
        return out
    else:
        z = normal(key, shape, dtype=dtype)
        newkey = jax.random.split(key)[0]
        return z, newkey



def fixed_normal_samples(shape, dtype=None):
    key = PRNGKey(4)
    z = normal(key, shape, dtype=dtype)
    return z


def jvp_derivs(fun, primals, tangents, create_graph=True):
    _, deriv_out = jax.jvp(fun, primals, tangents)
    return deriv_out


def vjp_derivs(fun, primals, duals, create_graph=True):
    _, fn = vjp(fun, *primals)
    return fn(duals)


def linear_transpose(fun, primals, duals):
    return jax.linear_transpose(fun, primals)(duals)[0]


def zeros(shape, dtype):
    return jnp.zeros(shape=shape, dtype=dtype)


def ones(shape, dtype):
    return jnp.ones(shape=shape, dtype=dtype)


def array(arr, dtype=None):
    return jnp.array(arr, dtype=dtype)


def update_array(array, update, *slices):
    return array.at[slices].set(update)
