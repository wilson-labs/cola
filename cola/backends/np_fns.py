import logging
import sys
import hashlib
import numpy as np
from scipy.linalg import block_diag as _block_diag, lu as _lu, solve_triangular
from scipy.signal import convolve2d
from cola.utils.torch_tqdm import while_loop_winfo
import optree


class NumpyNotImplementedError(NotImplementedError):
    def __init__(self):
        fn_name = sys._getframe(1).f_code.co_name
        super().__init__(f"{fn_name} is not implemented for the numpy backend.")


while_loop_winfo = while_loop_winfo
abs = np.abs
all = np.all
allclose = np.allclose
any = np.any
arange = np.arange
argsort = np.argsort
block_diag = _block_diag
cholesky = np.linalg.cholesky
complex64 = np.complex64
concat = np.concatenate
conj = np.conj
copy = np.copy
cos = np.cos
eigh = np.linalg.eigh
exp = np.exp
float32 = np.float32
float64 = np.float64
int32 = np.int32
int64 = np.int64
inv = np.linalg.inv
isreal = np.isreal
kron = np.kron
log = np.log
long = np.int64
max = np.max
maximum = np.maximum
mean = np.mean
min = np.min
moveaxis = np.moveaxis
nan_to_num = np.nan_to_num
ndarray = np.ndarray
norm = np.linalg.norm
normal = np.random.normal
ones_like = np.ones_like
prod = np.prod
reshape = np.reshape
roll = np.roll
sign = np.sign
sin = np.sin
slogdet = np.linalg.slogdet
solve = np.linalg.solve
solvetri = solve_triangular
sort = np.sort
sqrt = np.sqrt
stack = np.stack
sum = np.sum
svd = np.linalg.svd
where = np.where
fft = np.fft.fft
ifft = np.fft.ifft
slogdet = np.linalg.slogdet
promote_types = np.promote_types
finfo = np.finfo
zeros_like = np.zeros_like
fft = np.fft.fft
ifft = np.fft.ifft
slogdet = np.linalg.slogdet
promote_types = np.promote_types
finfo = np.finfo
iscomplexobj = np.iscomplexobj


def eig(array):
    eigvals, eigvecs = np.linalg.eig(array)
    eigvals, eigvecs = eigvals.astype(complex), eigvecs.astype(complex)
    return eigvals, eigvecs


def lu(a):
    p_ids, L, U = _lu(a, p_indices=True)
    return p_ids, L, U


def clip(array, a_min=None, a_max=None):
    return np.clip(array, a_min=a_min, a_max=a_max)


def qr(a, full_matrices=False):
    mode = "reduced" if not full_matrices else "complete"
    return np.linalg.qr(a, mode=mode)


def Parameter(array):
    return array


def array(arr, dtype=None, device=None):
    return np.array(arr, dtype=dtype)


def canonical(loc, shape, dtype, device):
    vec = np.zeros(shape, dtype=dtype)
    vec[loc] = 1.
    return vec


def cast(array, dtype):
    return array.astype(dtype)


def convolve(in1, in2, mode="same"):
    in12 = np.pad(
        in1,
        (
            (in2.shape[0] // 2, (in2.shape[0] + 1) // 2 - 1),
            (in2.shape[1] // 2, (in2.shape[1] + 1) // 2 - 1),
        ),
        "symmetric",
    )
    out = convolve2d(in12, in2, mode="valid")
    return out  # ,boundary='symm')


def device(device_name):
    return None


def PRNGKey(x):
    return sha_hash(x)


def next_key(key):
    return sha_hash(key)


def diag(v, diagonal=0):
    return np.diag(v, k=diagonal)


def sha_hash(n):
    n_bytes = n.to_bytes((n.bit_length() + 7) // 8, 'big')
    hash_bytes = hashlib.sha256(n_bytes).digest()
    hash_integer = int.from_bytes(hash_bytes, 'big')
    return int(hash_integer % (2**32 - 1))


def dynamic_slice(operand, start_indices, slice_sizes):
    raise NumpyNotImplementedError()


def expand(array, axis):
    return np.expand_dims(array, (axis, ))


def eye(n, m=None, dtype=None, device=None):
    del device
    return np.eye(N=n, M=m, dtype=dtype)


def for_loop(lower, upper, body_fun, init_val):
    state = init_val
    for iter in range(lower, upper):
        state = body_fun(iter, state)
    return state


def get_default_device():
    return None


def get_device(array):
    return None


def grad(fun):
    raise NumpyNotImplementedError()


def is_array(array):
    return isinstance(array, np.ndarray)


def jit(fn, static_argnums=None):
    return fn


def jvp(fun, primals, tangents, has_aux=False):
    raise NumpyNotImplementedError()


def jvp_derivs(fun, primals, tangents, create_graph=True):
    raise NumpyNotImplementedError()


def linear_transpose(fun, primals, duals):
    raise NumpyNotImplementedError()


def lu_solve(a, b):
    return solve(a, b)


def move_to(arr, device, dtype):
    if dtype is not None:
        arr = arr.astype(dtype)
    if device is not None:
        raise RuntimeError("move_to does not take in a device argument for the numpy backend.")
    return arr


def ones(shape, dtype, device):
    return np.ones(shape=shape, dtype=dtype)


def pbar_while(errorfn, tol, desc='', every=1, hide=False):
    raise NumpyNotImplementedError()


def permute(array, axes):
    return np.transpose(array, axes=axes)


def randn(*shape, dtype=None, device=None, key=None):
    if key is None:
        logging.warning("Non keyed randn used. To be deprecated soon.")
        key = PRNGKey(0)

    old_state = np.random.get_state()
    np.random.seed(key)
    z = np.random.randn(*shape).astype(dtype)
    np.random.set_state(old_state)
    return z


def update_array(array, update, *slices):
    array[slices] = update
    return array


def vjp(fun, *primals, has_aux=False):
    raise NumpyNotImplementedError()


def vjp_derivs(fun, primals, duals, create_graph=True):
    raise NumpyNotImplementedError()


def vmap(fun, in_axes=0, out_axes=0):
    raise NumpyNotImplementedError()


def while_loop(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val


def while_loop_no_jit(cond_fun, body_fun, init_val):
    return while_loop(cond_fun, body_fun, init_val)


def zeros(shape, dtype, device=None):
    del device
    return np.zeros(shape=shape, dtype=dtype)


def is_leaf(value):
    return optree.treespec_is_leaf(optree.tree_structure(value, namespace="cola"))


def tree_flatten(value):
    return optree.tree_flatten(value, namespace='cola')


def tree_unflatten(treedef, value):
    return optree.tree_unflatten(treedef, value)
