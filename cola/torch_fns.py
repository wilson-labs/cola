import torch
from torch.autograd.functional import vjp, jvp
from cola.utils.torch_tqdm import while_loop_winfo
from torch.nn import Parameter

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


def vmap(fun, in_axes=0, out_axes=0):
    return torch.vmap(func=fun, in_dims=in_axes, out_dims=out_axes)


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


def randn(*shape, dtype=None):
    return torch.randn(*shape, dtype=dtype)


def fixed_normal_samples(shape, dtype=None):
    # TODO: fix random seed for sample
    return torch.randn(*shape, dtype=dtype)


def vjp_derivs(fun, primals, duals):
    if isinstance(primals, (list, tuple)):
        conj_primals = type(primals)((torch.conj(primal) for primal in primals))
    else:
        conj_primals = torch.conj(primals)
    if isinstance(duals, (list, tuple)):
        conj_duals = type(duals)((torch.conj(primal) for primal in duals))
    else:
        conj_duals = torch.conj(duals)
    _, output = vjp(fun, inputs=conj_primals, v=conj_duals, create_graph=True)
    if isinstance(output, (list, tuple)):
        conj_output = type(output)((torch.conj(primal) for primal in output))
    else:
        conj_output = torch.conj(output)
    return conj_output


def jvp_derivs(fun, primals, tangents):
    # TODO: check conjugates for complex
    _, output = jvp(fun, inputs=torch.conj(primals), v=torch.conj(tangents), create_graph=True)
    return torch.conj(output)


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


def zeros(shape, dtype):
    return torch.zeros(size=shape, dtype=dtype)


def array(arr, dtype=None):
    return torch.tensor(arr, dtype=dtype)


def update_array(array, update, *slices):
    array[slices] = update
    return array
