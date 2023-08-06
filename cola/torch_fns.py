import hashlib
import torch
from torch.autograd.functional import vjp, jvp
from cola.utils.torch_tqdm import while_loop_winfo
from torch.nn import Parameter
from torch._vmap_internals import vmap as _vmap
import logging

logdet = torch.logdet
exp = torch.exp
cos = torch.cos
sin = torch.sin
ndarray = torch.Tensor
arange = torch.arange
ones_like = torch.ones_like
sign = torch.sign
Parameter = Parameter
any = torch.any
inv = torch.linalg.inv
pinv = torch.linalg.pinv
norm = torch.linalg.norm
abs = torch.abs
all = torch.all
mean = torch.mean
int32 = torch.int32
int64 = torch.long
float32 = torch.float32
float64 = torch.float64
complex64 = torch.complex64
complex128 = torch.complex128
long = torch.long
kron = torch.kron
eye = torch.eye
moveaxis = torch.moveaxis
block_diag = torch.block_diag
conj = torch.conj
sqrt = torch.sqrt
eig = torch.linalg.eig
eigh = torch.linalg.eigh
solve = torch.linalg.solve
copy = torch.clone
svd = torch.linalg.svd
diag = torch.diag
zeros_like = torch.zeros_like
cholesky = torch.linalg.cholesky
min = torch.min
max = torch.max
while_loop_winfo = while_loop_winfo
concat = torch.cat
log = torch.log
nan_to_num = torch.nan_to_num
is_array = torch.is_tensor
autograd = torch.autograd
argsort = torch.argsort
sparse_csr = torch.sparse_csr_tensor
roll = torch.roll
maximum = torch.maximum
isreal = torch.isreal
allclose = torch.allclose
jacrev = torch.func.jacrev


def lu(a):
    P, L, U = torch.linalg.lu(a)
    p_ids = (P @ torch.arange(P.shape[-1]).to(P.device, P.dtype)).to(torch.long)
    return p_ids, L, U


def move_to(arr, device, dtype):
    return arr.to(device=device, dtype=dtype)


def lu_solve(a, b):
    return solve(a, b)


def tensordot(a, b, axes=2):
    return torch.tensordot(a, b, dims=axes)


def get_device(array):
    return array.device


def get_default_device():
    return torch.device("cpu")


def device(device_name):
    if device_name == "cpu":
        return torch.device("cpu")
    else:
        return torch.device("gpu:0")


def PRNGKey(x):
    return sha_hash(x)


def vmap(fun, in_axes=0, out_axes=0):
    return _vmap(func=fun, in_dims=in_axes, out_dims=out_axes)


def stack(tensors, axis=0):
    return torch.stack(tensors, dim=axis)


def stop_gradients(x):
    return x.detach()


def canonical(loc, shape, dtype):
    vec = torch.zeros(shape, dtype=dtype)
    vec[loc] = 1.
    return vec


def sum(array, axis=0, keepdims=False):
    return torch.sum(array, dim=axis, keepdims=keepdims)


def permute(array, axes):
    return torch.permute(array, dims=axes)


def ones(shape, dtype):
    return torch.ones(size=shape, dtype=dtype)


def clip(array, a_min=None, a_max=None, out=None):
    return torch.clip(array, min=a_min, max=a_max, out=out)


def solvetri(matrix, rhs, lower=True):
    upper = not lower
    out = torch.linalg.solve_triangular(matrix, rhs, upper=upper)
    return out


def qr(matrix, full_matrices=False):
    mode = "reduced" if not full_matrices else "complete"
    Q = torch.linalg.qr(matrix, mode=mode)
    return Q


def expand(array, axis):
    return array.unsqueeze(axis)


def jit(fn, static_argnums=None):
    del static_argnums
    return fn


def cast(array, dtype):
    return array.to(dtype)


def sort(array):
    sorted, _ = torch.sort(array)
    return sorted


def next_key(key):
    return sha_hash(key)


def sha_hash(n):
    n_bytes = n.to_bytes((n.bit_length() + 7) // 8, 'big')
    hash_bytes = hashlib.sha256(n_bytes).digest()
    hash_integer = int.from_bytes(hash_bytes, 'big')
    return int(hash_integer % (2**32 - 1))


def randn(*shape, dtype=None, key=None):
    if key is None:
        print('Non keyed randn used. To be deprecated soon.')
        logging.warning('Non keyed randn used. To be deprecated soon.')
        key = PRNGKey(0)
    old_state = torch.random.get_rng_state()
    torch.random.manual_seed(key)
    z = torch.randn(*shape, dtype=dtype)
    torch.random.set_rng_state(old_state)
    return z


def fixed_normal_samples(shape, dtype=None):
    # TODO: fix random seed for sample
    return torch.randn(*shape, dtype=dtype)


def vjp_derivs(fun, primals, duals, create_graph=True):
    if isinstance(primals, (list, tuple)):
        conj_primals = type(primals)((torch.conj(primal) for primal in primals))
    else:
        conj_primals = torch.conj(primals)
    if isinstance(duals, (list, tuple)):
        conj_duals = type(duals)((torch.conj(primal) for primal in duals))
    else:
        conj_duals = torch.conj(duals)
    _, output = vjp(fun, inputs=conj_primals, v=conj_duals, create_graph=create_graph)
    if isinstance(output, (list, tuple)):
        conj_output = type(output)((torch.conj(primal) for primal in output))
    else:
        conj_output = torch.conj(output)
    return conj_output


def jvp_derivs(fun, primals, tangents, create_graph=True):
    if isinstance(primals, (list, tuple)):
        conj_primals = type(primals)((torch.conj(primal) for primal in primals))
    else:
        conj_primals = torch.conj(primals)
    if isinstance(tangents, (list, tuple)):
        conj_tangents = type(tangents)((torch.conj(primal) for primal in tangents))
    else:
        conj_tangents = torch.conj(tangents)
    _, output = jvp(fun, inputs=conj_primals, v=conj_tangents, create_graph=create_graph)
    if isinstance(output, (list, tuple)):
        conj_output = type(output)((torch.conj(primal) for primal in output))
    else:
        conj_output = torch.conj(output)
    return conj_output


def grad(fn):
    return lambda x: torch.autograd.grad([fn(x)], x, create_graph=True)[0]


def linear_transpose(fun, primals, duals):
    return vjp_derivs(fun, primals, duals)


def concatenate(arrays, axis=0):
    return torch.cat(arrays, dim=axis)


def for_loop(lower, upper, body_fun, init_val):
    state = init_val
    for iter in range(lower, upper):
        state = body_fun(iter, state)
    return state


def while_loop(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val


def while_loop_no_jit(cond_fun, body_fun, init_val):
    return while_loop(cond_fun, body_fun, init_val)


def where(condition, x, y):
    return torch.where(condition, x, y)


def zeros(shape, dtype, device=None):
    return torch.zeros(size=shape, dtype=dtype, device=device)


def array(arr, dtype=None):
    return torch.tensor(arr, dtype=dtype)


def update_array(array, update, *slices):
    array[slices] = update
    return array
