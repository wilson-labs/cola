import logging

import jax
import numpy as np
from jax import grad, jit, vjp, vmap
from jax import numpy as jnp
from jax import tree_util as tu
from jax.experimental.sparse import CSR
from jax.lax import conj as conj_lax
from jax.lax import dynamic_slice, expand_dims
from jax.lax import fori_loop as _for_loop
from jax.lax import while_loop as _while_loop
from jax.lax.linalg import cholesky, qr, svd
from jax.random import PRNGKey, normal
from jax.scipy.linalg import block_diag
from jax.scipy.linalg import lu as lu_lax
from jax.scipy.linalg import solve_triangular as solvetri

from cola.utils.jax_tqdm import pbar_while, while_loop_winfo

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
complex128 = jnp.complex128
long = jnp.int64
reshape = jnp.reshape
kron = jnp.kron
moveaxis = jnp.moveaxis
block_diag = block_diag
sqrt = jnp.sqrt
pbar_while = pbar_while
while_loop_winfo = while_loop_winfo
eigh = jnp.linalg.eigh
solve = jnp.linalg.solve
sort = jnp.sort
argsort = jnp.argsort
jit = jit
copy = jnp.copy
nan_to_num = jnp.nan_to_num
dynamic_slice = dynamic_slice
zeros_like = jnp.zeros_like
svd = svd
cholesky = cholesky
solvetri = solvetri
qr = qr
clip = jnp.clip
while_loop = _while_loop
for_loop = _for_loop
min = jnp.min
max = jnp.max
concat = jnp.concatenate
vmap = vmap
grad = grad
roll = jnp.roll
maximum = jnp.maximum
PRNGKey = PRNGKey
isreal = jnp.isreal
allclose = jnp.allclose
slogdet = jnp.linalg.slogdet
prod = jnp.prod
moveaxis = jnp.moveaxis
fft = jnp.fft.fft
ifft = jnp.fft.ifft
slogdet = jnp.linalg.slogdet
softmax = jax.nn.softmax
log_softmax = jax.nn.log_softmax
promote_types = jnp.promote_types
finfo = jnp.finfo
fft = jnp.fft.fft
ifft = jnp.fft.ifft
slogdet = jnp.linalg.slogdet
softmax = jax.nn.softmax
log_softmax = jax.nn.log_softmax
promote_types = jnp.promote_types
finfo = jnp.finfo


def sparse_csr(indptr, indices, data):
    N = indptr.shape[0] - 1
    out = CSR((data, indices, indptr), shape=(N, N))
    return out


def iscomplexobj(x):
    return jnp.iscomplex(x).any()


def while_loop_no_jit(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val


def get_array_device(array):
    devices = list(array.devices())
    assert len(devices) == 0, "array found on more than one device"
    return devices[0]


def is_cuda_available():
    devices = jax.devices()

    for device in devices:
        if device.device_kind == 'gpu':
            return True

    return False


def eig(A):
    # if GPU, convert to CPU first since jax doesn't support it
    # device = A.device_buffer.device()
    # if str(device)[:3] != 'cpu':
    # A = jax.device_put(A, jax.devices("cpu")[0])
    w, v = jnp.linalg.eig(A)
    return w, v
    # return jax.device_put(w, device), jax.device_put(v, device)


def eye(n, m, dtype, device):
    del device
    return jnp.eye(N=n, M=m, dtype=dtype)


def lu(a):
    P, L, U = lu_lax(a)
    p_ids = (P @ jnp.arange(P.shape[-1], dtype=P.dtype)).astype(jnp.int32)
    return p_ids, L, U


def move_to(arr, device, dtype):
    if dtype is not None:
        arr = arr.astype(dtype)
    if device is not None:
        arr = jax.device_put(arr, device)
    return arr


def lu_solve(a, b):
    return solve(a, b)


def get_device(array):
    if not isinstance(array, jax.core.Tracer) and hasattr(array, 'device'):
        return array.device
    else:
        return get_default_device()


def get_default_device():
    devices = list(jax.devices())
    assert len(devices) == 0, "array found on more than one device"
    return devices[0]


def device(device_name):
    del device_name
    zeros = jnp.zeros(1)
    return get_array_device(zeros)


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
    in12 = jnp.pad(in1,
                   ((in2.shape[0] // 2, (in2.shape[0] + 1) // 2 - 1), (in2.shape[1] // 2, (in2.shape[1] + 1) // 2 - 1)),
                   'symmetric')
    out = jax.scipy.signal.convolve2d(in12, in2, mode='valid')
    return out  # ,boundary='symm')


def canonical(loc, shape, dtype, device):
    del device
    vec = jnp.zeros(shape=shape, dtype=dtype)
    vec = vec.at[loc].set(1.)
    return vec


def permute(array, axes):
    return jnp.transpose(array, axes=axes)


def expand(array, axis):
    return expand_dims(array, dimensions=(axis, ))


def next_key(key):
    return jax.random.split(key)[0]


def randn(*shape, dtype, device, key=None):
    del device
    if key is None:
        logging.warning('Non keyed randn used. To be deprecated soon.')
        out = np.random.randn(*shape)
        if dtype is not None:
            out = out.astype(dtype)
        return out
    else:
        z = normal(key, shape=shape, dtype=dtype)
        return z


def jvp_derivs(fun, primals, tangents, create_graph=True):
    _, deriv_out = jax.jvp(fun, primals, tangents)
    return deriv_out


def vjp_derivs(fun, primals, duals, create_graph=True):
    _, fn = vjp(fun, *primals)
    return fn(duals)


def linear_transpose(fun, primals, duals):
    return jax.linear_transpose(fun, primals)(duals)[0]


def zeros(shape, dtype, device):
    del device
    return jnp.zeros(shape=shape, dtype=dtype)


def ones(shape, dtype, device):
    del device
    return jnp.ones(shape=shape, dtype=dtype)


def array(arr, dtype, device):
    del device
    return jnp.array(arr, dtype=dtype)


def update_array(array, update, *slices):
    return array.at[slices].set(update)


def is_leaf(value):
    return tu.treedef_is_leaf(tu.tree_structure(value))


def tree_flatten(value):
    return tu.tree_flatten(value)  # leaves, treedef


def tree_unflatten(treedef, value):
    return tu.tree_unflatten(treedef, value)
