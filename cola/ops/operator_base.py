from abc import abstractmethod
from typing import Union, Tuple, Any, List, Callable
# import cola.fns  # import dot, add, mul
import cola
import numpy as np
from cola.utils import export
from numbers import Number

Array = Dtype = Any
export(Array)

def get_library_fns(dtype: Dtype):
    """ Given a dtype e.g. jnp.float32 or torch.complex64, returns the appropriate
        namespace for standard array functionality (either torch_fns or jax_fns)."""
    try:
        from jax import numpy as jnp
        if dtype in [jnp.float32, jnp.float64, jnp.complex64, jnp.int32, jnp.int64]:
            import cola.jax_fns as fns
            return fns
    except ImportError:
        pass
    try:
        import torch
        if dtype in [torch.float32, torch.float64, torch.complex64, torch.complex128,
                     torch.int32, torch.int64]:
            import cola.torch_fns as fns
            return fns
    except ImportError:
        pass
    raise ImportError("No supported array library found")


class AutoRegisteringPyTree(type):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            import jax
            jax.tree_util.register_pytree_node_class(cls)
        except ImportError:
            pass

@export
class LinearOperator(metaclass=AutoRegisteringPyTree):
    """ Linear Operator base class """
    def __new__(cls, *args, **kwargs):
        """ Creates attributes for the flatten and unflatten functionality. """
        obj = super().__new__(cls)
        obj._args = args
        obj._kwargs = kwargs
        return obj

    def __init__(self, dtype: Dtype, shape: Tuple, matmat=None):
        self.dtype = dtype
        self.shape = shape
        self.ops = get_library_fns(dtype)
        if matmat is not None:
            self._matmat = matmat
        # self._args
        # self._kwargs

    def to(self, dtype=None, device=None):
        # returns a new linear operator.
        raise NotImplementedError()

    @abstractmethod
    def _matmat(self, X: Array) -> Array:
        """ Defines multiplication AX of the LinearOperator A with a dense array X (d,k)
            where A (self) is shape (c,d)"""
        raise NotImplementedError

    def _rmatmat(self, X: Array) -> Array:
        """ Defines multiplication XA of the LinearOperator A with a dense array X (k,d)
            where A (self) is shape (d,c). By default uses jvp to compute the transpose."""
        XT = X.T
        primals = self.ops.zeros(
            shape=(self.shape[1], XT.shape[1]), dtype=XT.dtype)
        out = self.ops.linear_transpose(
            self._matmat, primals=primals, duals=XT)
        return out.T

    def to_dense(self) -> Array:
        """ Produces a dense array representation of the linear operator. """
        if 3 * self.shape[-2] < self.shape[-1]:
            return self.ops.eye(self.shape[-2], dtype=self.dtype) @ self
        else:
            return self @ self.ops.eye(self.shape[-1], dtype=self.dtype)

    @property
    def T(self):
        """ Matrix Transpose """
        return cola.fns.transpose(self)

    @property
    def H(self):
        """ Matrix complex conjugate transpose (aka hermitian conjugate, adjoint)"""
        return cola.fns.adjoint(self)

    def flatten(self) -> Tuple[Array, ...]:
        return flatten_function(self)

    def __matmul__(self, X: Array) -> Array:
        assert X.shape[0] == self.shape[-1], f"dimension mismatch {self.shape} vs {X.shape}"
        if isinstance(X, LinearOperator):
            return cola.fns.dot(self, X)
        elif len(X.shape) == 1:
            return self._matmat(X.reshape(-1, 1)).reshape(-1)
        elif len(X.shape) >= 2:
            return self._matmat(X)
        else:
            raise NotImplementedError

    def __rmatmul__(self, X: Array) -> Array:
        assert X.shape[-1] == self.shape[-2], f"dimension mismatch {self.shape} vs {X.shape}"
        if isinstance(X, LinearOperator):
            return cola.fns.dot(X, self)
        elif len(X.shape) == 1:
            return self._rmatmat(X.reshape(1, -1)).reshape(-1)
        elif len(X.shape) >= 2:
            return self._rmatmat(X)
        else:
            raise NotImplementedError

    def __add__(self, other):
        # check if is numbers.Number
        
        if isinstance(other, Number):
            if other == 0:
                return self
        return cola.fns.add(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, c):
        # assert isinstance(c, (int, float)), "c must be a scalar"
        return cola.fns.mul(self, c)

    def __rmul__(self, c):
        return self * c

    def __neg__(self):
        return -1 * self

    def __sub__(self, x):
        return self.__add__(-x)

    def __truediv__(self, x):
        return self.__mul__(1 / x)

    def __rtruediv__(self, x):
        return self.__mul__(1 / x)

    def __str__(self):
        # check if class is LinearOperator
        if self.__class__.__name__ != 'LinearOperator':
            return self.__class__.__name__
        alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ'
        return alphabet[hash(id(self)) % 24]

    def __repr__(self):
        M, N = self.shape
        dt = 'dtype=' + str(self.dtype)
        return '<%dx%d %s with %s>' % (M, N, self.__class__.__name__, dt)

    def __getitem__(self, ids: Union[Tuple[int, ...], Tuple[slice, ...]]):
        # TODO: add Tuple[List[int],...] and List[Tuple[int,int]] cases
        # print(type(ids))
        # print(type(ids[0]), type(ids[1]))
        # check if first element is ellipsis
        if ids[0] is Ellipsis:
            ids = ids[1:]
        xnp = self.ops
        match ids:
            case int(i), int(j):
                ej = xnp.canonical(loc=j, shape=(
                    self.A.shape[-1], ), dtype=self.dtype)
                return (self @ ej)[i]
            case int(i), slice() as s:
                ei = xnp.canonical(loc=i, shape=(
                    self.A.shape[-1], ), dtype=self.dtype)
                return (self.T @ ei)[s]
            case slice() as s, int(j):
                ej = xnp.canonical(loc=j, shape=(
                    self.A.shape[-1], ), dtype=self.dtype)
                return (self @ ej)[s]
            case (slice() | xnp.ndarray() | np.ndarray()) as s_i,  \
                 (slice() | xnp.ndarray() | np.ndarray()) as s_j:
                from cola.ops import Sliced
                return Sliced(A=self, slices=(s_i, s_j))
            case list(li), list(lj):
                out = []
                for idx, jdx in zip(li, lj):
                    # TODO: batch jdx
                    ej = xnp.canonical(loc=jdx, shape=(
                        self.A.shape[-1], ), dtype=self.dtype)
                    out.append((self.A @ ej)[idx])
                return xnp.stack(out)
            case _:
                raise NotImplementedError(
                    f"__getitem__ not implemented for this case {type(ids)}")

    def tree_flatten(self):
        # write a routine that sorts args and kwargs into
        # dynamic arguments (LinearOperator, array, etc)
        # and static arguments (int, float, everything else)
        # and then also stores the information needed to reproduce
        # the original _args and _kwargs in its original form
        import jax

        def is_leaf(x):
            return not isinstance(x, (tuple, list, dict, set))
        flat_args, uf = jax.tree_util.tree_flatten(self._args, is_leaf)
        return flat_args, (self._kwargs, uf)

    @classmethod
    def tree_unflatten(cls, aux, children):
        import jax
        new_args = jax.tree_util.tree_unflatten(aux[1], children)
        return cls(*new_args, **aux[0])


def flatten_function(obj) -> Tuple[List[Array], Callable]:
    if is_array(obj):
        return [obj], lambda x: x[0]
    elif isinstance(obj, (LinearOperator, tuple, list)):  # TODO add dict?
        args = obj._args if isinstance(obj, LinearOperator) else obj
        unflatten_fns, flat, slices = [], [], [slice(-1, 0)]
        for arg in args:
            params, unflatten = flatten_function(arg)
            slices.append(
                slice(slices[-1].stop, slices[-1].stop + len(params)))
            unflatten_fns.append(unflatten)
            flat.extend(params)

        def unflatten(params):
            new_params = []
            for slc, unflatten in zip(slices[1:], unflatten_fns):
                new_params.append(unflatten(params[slc]))
            if isinstance(obj, LinearOperator):
                # return obj.tree_unflatten(obj._kwargs, new_params)
                return obj.__class__(*new_params, **obj._kwargs)
            else:
                return obj.__class__(new_params)
        return flat, unflatten
    else:
        raise NotImplementedError


def is_array(obj):
    if not hasattr(obj, 'dtype'):
        return False
    if get_library_fns(obj.dtype).is_array(obj):
        return True
    return False
