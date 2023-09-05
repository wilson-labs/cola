from abc import abstractmethod
from typing import Union, Tuple, Any
from numbers import Number
import numpy as np
import cola
from cola.utils import export
import cola.np_fns as np_fns

Array = Dtype = Any
export(Array)


def get_library_fns(dtype: Dtype):
    """ Given a dtype e.g. jnp.float32 or torch.complex64, returns the appropriate
        namespace for standard array functionality (either torch_fns or jax_fns)."""
    try:
        from jax import numpy as jnp
        if dtype in [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128, jnp.int32, jnp.int64]:
            import cola.jax_fns as fns
            return fns
    except ImportError:
        pass
    try:
        import torch
        if dtype in [
                torch.float32, torch.float64, torch.complex64, torch.complex128, torch.int32,
                torch.int64
        ]:
            import cola.torch_fns as fns
            return fns
        elif dtype in [np.float32, np.float64, np.complex64, np.complex128, np.int32, np.int64]:
            import cola.np_fns as fns
            return fns
    except ImportError:
        pass
    raise ImportError("No supported array library found")


def is_array(obj):
    if not hasattr(obj, 'dtype'):
        return False
    if get_library_fns(obj.dtype).is_array(obj):
        return True
    return False


def is_xnp_array(obj, xnp):
    if not hasattr(obj, 'dtype'):
        return False
    if xnp.is_array(obj):
        return True
    return False


class AutoRegisteringPyTree(type):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls._dynamic = cls._dynamic.copy()
        import optree
        optree.register_pytree_node_class(cls, namespace='cola')
        try:
            import jax
            jax.tree_util.register_pytree_node_class(cls)
        except ImportError:
            pass
        try:
            # TODO: when pytorch migrates to optree, switch as well
            import torch

            def tree_flatten(self):
                return self.tree_flatten()

            def tree_unflatten(ctx, children):
                return cls.tree_unflatten(children, ctx)

            torch.utils._pytree._register_pytree_node(cls, tree_flatten, tree_unflatten)
        except ImportError:
            pass


def find_xnp(obj):
    if is_array(obj):
        return get_library_fns(obj.dtype)
    elif isinstance(obj, LinearOperator) and obj.xnp is not None:
        return obj.xnp
    elif isinstance(obj, (tuple, list, set)):
        for ob in obj:
            xnp = find_xnp(ob)
            if xnp is not None:
                return xnp
    elif isinstance(obj, dict):
        for _, ob in obj.items():
            xnp = find_xnp(ob)
            if xnp is not None:
                return xnp
    try:
        return get_library_fns(obj)
    except (ImportError, AttributeError):
        pass
    return None


def find_device(obj):
    if is_array(obj):
        xnp = get_library_fns(obj.dtype)
        return xnp.get_device(obj)
    elif isinstance(obj, LinearOperator):
        return obj.device
    elif isinstance(obj, (tuple, list, set)):
        for ob in obj:
            device = find_device(ob)
            if device is not None:
                return device
    elif isinstance(obj, dict):
        for _, ob in obj.items():
            device = find_device(ob)
            if device is not None:
                return device
    return None


def definitely_dynamic(obj):
    return is_array(obj) or isinstance(obj, LinearOperator)

@export
class LinearOperator(metaclass=AutoRegisteringPyTree):
    """ Linear Operator base class """
    _dynamic = {key: False for key in ['xnp', 'shape', 'dtype', 'device', 'annotations']}

    def __new__(cls, *args, **kwargs):
        """ Creates attributes for the flatten and unflatten functionality. """
        obj = super().__new__(cls)
        obj.device = find_device([args, kwargs])
        return obj

    def __init__(self, dtype: Dtype, shape: Tuple, matmat=None, annotations={}):
        self.dtype = dtype
        self.shape = shape
        self.xnp = get_library_fns(dtype)
        if matmat is not None:
            self._matmat = matmat
        self.annotations = cola.annotations.get_annotations(self)
        # TODO: reform matrices with the new annotations?
        self.annotations.update(annotations)
        self.device = self.device or self.xnp.get_default_device()

    def __setattr__(self, name, value):
        if name not in self.__class__._dynamic:
            # don't split this into two lines, we want the short circuiting
            cond = definitely_dynamic(value) or any(map(is_array, np_fns.tree_flatten(value)[0]))
            self.__class__._dynamic[name] = cond
        return super().__setattr__(name, value)

    def to(self, device, dtype=None):
        """ Returns a new linear operator with given device and dtype
            WARNING: dtype change is not supported yet. """
        params, unflatten = self.flatten()
        params = [
            self.xnp.move_to(p, device=device, dtype=dtype) if self.xnp.is_array(p) else p
            for p in params
        ]
        return unflatten(params)

    def isa(self, annotation) -> bool:
        """ Returns True if the LinearOperator has the given annotation. """
        return any(issubclass(a, annotation) for a in self.annotations)

    @abstractmethod
    def _matmat(self, X: Array) -> Array:
        """ Defines multiplication AX of the LinearOperator A with a dense array X (d,k)
            where A (self) is shape (c,d)"""
        raise NotImplementedError

    def _rmatmat(self, X: Array) -> Array:
        """ Defines multiplication XA of the LinearOperator A with a dense array X (k,d)
            where A (self) is shape (d,c). By default uses jvp to compute the transpose."""
        XT = X.T
        if self.isa(cola.annotations.SelfAdjoint):
            return self.xnp.conj(self._matmat(self.xnp.conj(XT)).T)
        primals = self.xnp.zeros(shape=(self.shape[1], XT.shape[1]), dtype=XT.dtype,
                                 device=self.device)
        out = self.xnp.linear_transpose(self._matmat, primals=primals, duals=XT)
        return out.T

    def to_dense(self) -> Array:
        """ Produces a dense array representation of the linear operator. """
        if 8 * self.shape[-2] < self.shape[-1]:
            return self.xnp.eye(self.shape[-2], self.shape[-2], dtype=self.dtype,
                                device=self.device) @ self
        else:
            return self @ self.xnp.eye(self.shape[-1], self.shape[-1], dtype=self.dtype,
                                       device=self.device)

    @property
    def T(self):
        """ Matrix Transpose """
        return cola.fns.transpose(self)

    @property
    def H(self):
        """ Matrix complex conjugate transpose (aka hermitian conjugate, adjoint)"""
        return cola.fns.adjoint(self)

    def flatten(self) -> Tuple[Array, ...]:
        vals, tree = self.xnp.tree_flatten(self)

        def unflatten(params):
            return self.xnp.tree_unflatten(tree, params)

        return vals, unflatten

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
        if isinstance(other, Number) and other == 0:
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

    def __getitem__(
            self, ids: Union[Tuple[int, ...], Tuple[slice, ...]]) -> Union[Array, 'LinearOperator']:
        # TODO: add Tuple[List[int],...] and List[Tuple[int,int]] cases
        # print(type(ids))
        # print(type(ids[0]), type(ids[1]))
        # check if first element is ellipsis
        xnp = self.xnp
        from cola.ops import Sliced
        match ids:
            case int(i):
                ei = xnp.canonical(loc=i, shape=(self.shape[-1], ), dtype=self.dtype,
                                   device=self.device)
                return (self.T @ ei)
            case (slice() | xnp.ndarray() | np.ndarray()) as s_i:
                return Sliced(A=self, slices=(s_i, slice(None)))
            case b, int(j):
                ej = xnp.canonical(loc=j, shape=(self.shape[-1], ), dtype=self.dtype,
                                   device=self.device)
                return (self @ ej)[b]
            case int(i), b:
                ei = xnp.canonical(loc=i, shape=(self.shape[-1], ), dtype=self.dtype,
                                   device=self.device)
                return (self.T @ ei)[b]
            case (slice() | xnp.ndarray() | np.ndarray()) as s_i,  \
                 (slice() | xnp.ndarray() | np.ndarray()) as s_j:
                return Sliced(A=self, slices=(s_i, s_j))
            case list(li), list(lj):
                out = []
                for idx, jdx in zip(li, lj):
                    # TODO: batch jdx
                    ej = xnp.canonical(loc=jdx, shape=(self.A.shape[-1], ), dtype=self.dtype,
                                       device=self.device)
                    out.append((self.A @ ej)[idx])
                return xnp.stack(out)
            case _:
                raise NotImplementedError(f"__getitem__ not implemented for this case {type(ids)}")

    def tree_flatten(self):
        # separate all_elems into pytrees and aux
        pytrees, aux = [], []
        for key, val in sorted(vars(self).items()):
            if self._dynamic[key]:
                pytrees.append(val)
                aux.append((key, ))
            else:
                aux.append((key, val))
        return pytrees, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        fields = {}
        child_iter = iter(children)
        for keyv in aux:
            if len(keyv) == 1:
                fields[keyv[0]] = next(child_iter)
            else:
                fields[keyv[0]] = keyv[1]
        obj = object.__new__(cls)
        for k, v in fields.items():
            if k in ['device']:  # ,'dtype']: TODO: also separate dtype in case .to was called
                continue
            setattr(obj, k, v)
        # dtypes = [dt for dt in map(maybe_get_dtype, children) if dt is not None]
        # obj.dtype = reduce(obj.xnp.promote_types, dtypes) if len(dtypes) > 0 else None
        obj.device = find_device(fields) or fields['device']
        return obj


def maybe_get_dtype(obj):
    try:
        return obj.dtype
    except AttributeError:
        return None
