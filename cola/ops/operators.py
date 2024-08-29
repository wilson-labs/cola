from functools import partial, reduce

import numpy as np
from plum import parametric
from scipy.sparse import coo_array

import cola
from cola.backends import get_library_fns
from cola.ops.operator_base import Array, LinearOperator


class Dense(LinearOperator):
    """ LinearOperator wrapping of a dense matrix. O(n^2) memory and time mvms.

    Args:
        A (array_like): Dense matrix to be wrapped.

    Example:
        >>> A = jnp.array([[1., 2.], [3., 4.]])
        >>> op = Dense(A)
    """
    def __init__(self, A: Array):
        self.A = A
        super().__init__(dtype=A.dtype, shape=A.shape)

    def _matmat(self, X: Array) -> Array:
        dtype = self.xnp.promote_types(self.dtype, X.dtype)
        return self.xnp.cast(self.A, dtype) @ self.xnp.cast(X, dtype)

    def _rmatmat(self, X: Array) -> Array:
        # TODO: check if this is a correct fix
        # xnp = self.xnp
        # self.A = self.A.to(xnp.get_array_device(X))
        dtype = self.xnp.promote_types(self.dtype, X.dtype)
        return self.xnp.cast(X, dtype) @ self.xnp.cast(self.A, dtype)

    def to_dense(self):
        return self.A


class Triangular(Dense):
    """ Triangular Linear Operator."""
    def __init__(self, A: Array, lower=True):
        super().__init__(A)
        self.lower = lower


class Sparse(LinearOperator):
    """ Sparse linear operator.

    Args:
        data (array_like): 1-D array representing the nonzero values of the sparse matrix.
        row_indices (array_like): 1-D array representing the row indices of the nonzero values.
        col_indices (array_like): 1-D array representing the column indices of the nonzero values.
        shape (tuple): Shape of the sparse matrix.

    Example:
        >>> data = jnp.array([1, 2, 3, 4, 5, 6])
        >>> rol_indices = jnp.array([0, 0, 1, 2, 2, 2])
        >>> col_indices = jnp.array([1, 3, 3, 0, 1, 2])
        >>> shape = (3, 4)
        >>> op = Sparse(data, row_indices, col_indices, shape)
    """
    def __init__(self, data, row_indices, col_indices, shape):
        super().__init__(dtype=data.dtype, shape=shape)
        xnp = self.xnp
        indx = xnp.argsort(row_indices)
        self.data = data[indx]
        self.row_indices = row_indices[indx]
        self.col_indices = col_indices[indx]
        A = coo_array((xnp.to_np(self.data), (xnp.to_np(self.row_indices), xnp.to_np(self.col_indices))),
                      shape=shape).tocsr()
        row_pointers = xnp.array(A.indptr, dtype=xnp.int32, device=data.device)
        indices = xnp.array(A.indices, dtype=xnp.int32, device=data.device)
        self.A = xnp.sparse_csr(row_pointers, indices, self.data, shape)

    def _matmat(self, V):
        return self.A @ V

    def _rmatmat(self, V):
        return (self.T @ V.T).T


class ScalarMul(LinearOperator):
    """ Linear Operator representing scalar multiplication"""
    def __init__(self, c, shape, dtype=None, device=None):
        super().__init__(dtype=dtype or type(c), shape=shape)
        self.c = self.xnp.array(c, dtype=dtype, device=device)
        self.device = device

    #     self.ensure_const_register_as_array()

    # def ensure_const_register_as_array(self):
    #     self._args = (self.c, )
    #     self._kwargs = {"dtype": self.dtype, "shape": self.shape}

    def _matmat(self, v):
        return self.c * v

    def __str__(self):
        return f"{self.c}"


class Identity(LinearOperator):
    """ Linear Operator representing the identity matrix. Can also be created from I_like(A)

        Args:
            shape (tuple): Shape of the identity matrix.
            dtype: Data type of the identity matrix.

        Example:
            >>> shape = (3, 3)
            >>> dtype =  jnp.float64
            >>> op = Identity(shape, dtype)
    """
    def __init__(self, shape, dtype):
        super().__init__(dtype=dtype, shape=shape)

    def __str__(self):
        return "I"

    def _matmat(self, X):
        return X

    def to(self, device):
        self.device = device
        return self


def I_like(A: LinearOperator) -> Identity:
    """ A function that produces an Identity operator with the same
        shape, dtype and device as A """
    Op = Identity(dtype=A.dtype, shape=A.shape)
    Op.to(A.device)
    return Op


@parametric
class Product(LinearOperator):
    """ Matrix Multiply Product of Linear ops """
    def __init__(self, *Ms):
        self.Ms = tuple(cola.fns.lazify(M) for M in Ms)
        devices = [M.device for M in self.Ms]
        assert all(x == devices[0] for x in devices), "There is a device mismatch"
        for M1, M2 in zip(Ms[:-1], Ms[1:]):
            if M1.shape[-1] != M2.shape[-2]:
                raise ValueError(f"dimension mismatch {M1.shape} vs {M2.shape}")
        shape = (Ms[0].shape[-2], Ms[-1].shape[-1])
        dtype = reduce(self.Ms[0].xnp.promote_types, (M.dtype for M in Ms))
        super().__init__(dtype, shape)
        self.device = devices[0]

    def _matmat(self, v):
        for M in self.Ms[::-1]:
            v = M @ v
        return v

    def _rmatmat(self, v):
        for M in self.Ms:
            v = v @ M
        return v

    def __str__(self):
        return "".join(str(M) for M in self.Ms)


@parametric
class Sum(LinearOperator):
    """ Sum of Linear ops """
    def __init__(self, *Ms):
        self.Ms = tuple(cola.fns.lazify(M) for M in Ms)
        devices = [M.device for M in self.Ms]
        assert all(x == devices[0] for x in devices), "There is a device mismatch"
        shape = Ms[0].shape
        for M in Ms:
            if M.shape != shape:
                raise ValueError(f"dimension mismatch {M.shape} vs {shape}")
        dtype = Ms[0].dtype
        super().__init__(dtype, shape)
        self.device = devices[0]

    def _matmat(self, v):
        return sum(M @ v for M in self.Ms)

    def _rmatmat(self, v):
        return sum(v @ M for M in self.Ms)

    def __str__(self):
        if len(self.Ms) > 5:
            return "Sum({}...)".format(", ".join(str(M) for M in self.Ms[:2]))
        return "+".join(str(M) for M in self.Ms)


def product(c):
    return reduce(lambda a, b: a * b, c)


@parametric
class Kronecker(LinearOperator):
    """ Kronecker product of linear ops Kronecker([M1,M2]):= M1⊗M2

    Args:
        *Ms (array_like): Sequence of linear operators representing the Kronecker product operands.

    Example:
        >>> M1 = jnp.array([[1, 2], [3, 4]])
        >>> M2 = jnp.array([[5, 6], [7, 8]])
        >>> op = Kronecker(M1, M2)
    """
    def __init__(self, *Ms):
        self.Ms = tuple(cola.fns.lazify(M) for M in Ms)
        shape = product([Mi.shape[-2] for Mi in Ms]), product([Mi.shape[-1] for Mi in Ms])
        dtype = reduce(self.Ms[0].xnp.promote_types, (M.dtype for M in Ms))
        super().__init__(dtype, shape)

    def _matmat(self, v):
        ev = v.reshape(*[Mi.shape[-1] for Mi in self.Ms], -1)
        for i, M in enumerate(self.Ms):
            ev_front = self.xnp.moveaxis(ev, i, 0)
            shape = M.shape[0], *ev_front.shape[1:]
            Mev_front = (M @ ev_front.reshape(M.shape[-1], -1)).reshape(shape)
            ev = self.xnp.moveaxis(Mev_front, 0, i)
        return ev.reshape(self.shape[-2], ev.shape[-1])

    def to_dense(self):
        Ms = [M.to_dense() if isinstance(M, LinearOperator) else M for M in self.Ms]
        return reduce(self.xnp.kron, Ms)

    def __str__(self):
        return "⊗".join(str(M) for M in self.Ms)


def kronsum(A, B):
    xnp = get_library_fns(A.dtype)
    device = xnp.get_device(A)
    IA = xnp.eye(A.shape[-2], A.shape[-2], dtype=A.dtype, device=device)
    IB = xnp.eye(B.shape[-2], B.shape[-2], dtype=B.dtype, device=device)
    return xnp.kron(A, IB) + xnp.kron(IA, B)


@parametric
class KronSum(LinearOperator):
    """ Kronecker Sum Linear Operator, KronSum(A,B):= A ⊕ B = A ⊗ I + I ⊗ B

        Args:
            *Ms (array_like): Sequence of matrices representing the Kronecker sum operands.

        Example:
            >>> M1 = jnp.array([[1, 2], [3, 4]])
            >>> M2 = jnp.array([[5, 6], [7, 8]])
            >>> op = KronSum(M1, M2)
    """
    def __init__(self, *Ms):
        self.Ms = tuple(cola.fns.lazify(M) for M in Ms)
        shape = product([Mi.shape[-2] for Mi in Ms]), product([Mi.shape[-1] for Mi in Ms])
        dtype = reduce(self.Ms[0].xnp.promote_types, (M.dtype for M in Ms))
        super().__init__(dtype, shape)

    def _matmat(self, v):
        ev = v.reshape(*[Mi.shape[-1] for Mi in self.Ms], -1)
        out = 0 * ev
        xnp = self.xnp
        for i, M in enumerate(self.Ms):
            ev_front = xnp.moveaxis(ev, i, 0)
            Mev_front = (M @ ev_front.reshape(M.shape[-1], -1)).reshape(M.shape[0], *ev_front.shape[1:])
            out += xnp.moveaxis(Mev_front, 0, i)
        return out.reshape(self.shape[-2], ev.shape[-1])

    def to_dense(self):
        Ms = [M.to_dense() if isinstance(M, LinearOperator) else M for M in self.Ms]
        return reduce(kronsum, Ms)

    def __str__(self):
        return "⊕ₖ".join(str(M) for M in self.Ms)


@parametric
class BlockDiag(LinearOperator):
    """ Block Diagonal Linear Operator. BlockDiag([A,B]):= [A 0; 0 B]

    Args:
        *Ms (array_like): Sequence of matrices representing the blocks.
        multiplicities (list, optional): List of integers representing the multiplicities
            of the corresponding blocks in *Ms. Default is None, which assigns a multiplicity
            of 1 to each block.
    Example:
        >>> M1 = jnp.array([[1, 2], [3, 4]])
        >>> M2 = jnp.array([[5, 6], [7, 8]])
        >>> op = BlockDiag(M1, M2, multiplicities=[2, 3])
    """
    def __init__(self, *Ms, multiplicities=None):
        self.Ms = tuple(cola.fns.lazify(M) for M in Ms)
        self.multiplicities = [1 for _ in Ms] if multiplicities is None else multiplicities
        shape = (sum(Mi.shape[-2] * c for Mi, c in zip(Ms, self.multiplicities)),
                 sum(Mi.shape[-1] * c for Mi, c in zip(Ms, self.multiplicities)))
        dtype = reduce(self.Ms[0].xnp.promote_types, (M.dtype for M in Ms))
        super().__init__(dtype, shape)

    def _matmat(self, v):  # (n,k)
        # n = v.shape[0]
        k = v.shape[1] if len(v.shape) > 1 else 1
        i = 0
        y = []
        for M, multiplicity in zip(self.Ms, self.multiplicities):
            i_end = i + multiplicity * M.shape[-1]
            elems = M @ v[i:i_end].T.reshape(k * multiplicity, M.shape[-1]).T
            y.append(elems.T.reshape(k, multiplicity * M.shape[0]).T)
            i = i_end
        y = self.xnp.concat(y, axis=0)  # concatenate over rep axis
        return y

    def to_dense(self):
        Ms_all = [M for M, c in zip(self.Ms, self.multiplicities) for _ in range(c)]
        Ms_all = [Mi.to_dense() if isinstance(Mi, LinearOperator) else Mi for Mi in Ms_all]
        return self.xnp.block_diag(*Ms_all)

    def __str__(self):
        if len(self.Ms) > 5:
            return "BlockDiag({}...)".format(", ".join(str(M) for M in self.Ms[:2]))
        return "⊕".join(str(M) for M in self.Ms)


class Diagonal(LinearOperator):
    """ Diagonal LinearOperator. O(n) time and space matmuls.

        Args:
            diag (array_like): 1-D array representing the diagonal elements of the matrix.

        Example:
            >>> d = jnp.array([1, 2, 3])
            >>> op = Diagonal(d)
        """
    def __init__(self, diag):
        assert len(diag.shape) == 1, f"diagonal is not a vector, it is of shape {diag.shape=}"
        self.diag = diag
        super().__init__(dtype=diag.dtype, shape=(len(diag), ) * 2)

    def _matmat(self, X: Array) -> Array:
        return self.diag[:, None] * X

    def _rmatmat(self, X: Array) -> Array:
        return self.diag[None, :] * X

    def to_dense(self):
        return self.xnp.diag(self.diag)

    def __str__(self):
        return f"diag({self.diag})"


class Tridiagonal(LinearOperator):
    """ Tridiagonal linear operator. O(n) time and space matmuls.

    Args:
        alpha (array_like): 1-D array representing lower band of the operator.
        beta (array_like): 1-D array representing diagonal of the operator.
        gamma (array_like): 1-D array representing upper band of the operator.
    """
    def __init__(self, alpha: Array, beta: Array, gamma: Array):
        alpha, beta = ensure_vec_is_matrix(alpha), ensure_vec_is_matrix(beta)
        gamma = ensure_vec_is_matrix(gamma)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        super().__init__(dtype=beta.dtype, shape=(beta.shape[0], beta.shape[0]))

    def _matmat(self, X: Array) -> Array:
        xnp = self.xnp
        output = self.beta * X
        zeros = xnp.zeros(shape=(1, X.shape[-1]), dtype=X.dtype, device=xnp.get_device(X))
        aux_gamma = xnp.concat([self.gamma * X[1:], zeros], axis=0)
        zeros = xnp.zeros(shape=(1, X.shape[-1]), dtype=X.dtype, device=xnp.get_device(X))
        aux_alpha = xnp.concat([zeros, self.alpha * X[:-1]], axis=0)
        return output + aux_alpha + aux_gamma


def ensure_vec_is_matrix(vec):
    if len(vec.shape) == 1:
        vec = vec.reshape(-1, 1)
    return vec


@parametric
class Transpose(LinearOperator):
    """ Transpose of a Linear Operator"""
    def __init__(self, A):
        self.A = A
        super().__init__(dtype=A.dtype, shape=(A.shape[1], A.shape[0]))
        self.device = A.device

    def _matmat(self, x):
        return self.A._rmatmat(x.T).T

    def _rmatmat(self, x):
        return self.A._matmat(x.T).T

    def __str__(self):
        return f"{str(self.A)}ᵀ"


@parametric
class Adjoint(LinearOperator):
    """ Complex conjugate transpose of a Linear Operator (aka adjoint)"""
    def __init__(self, A):
        self.A = A
        super().__init__(dtype=A.dtype, shape=(A.shape[1], A.shape[0]))
        self.device = A.device

    def _matmat(self, x):
        return self.xnp.conj(self.A._rmatmat(self.xnp.conj(x).T)).T

    def _rmatmat(self, x):
        return self.xnp.conj(self.A._matmat(self.xnp.conj(x).T)).T

    def __str__(self):
        return f"{str(self.A)}*"


@parametric
class Sliced(LinearOperator):
    """ Slicing of another linear operator A.
        Equivalent to A[slices[0], :][:, slices[1]] """
    def __init__(self, A, slices):
        self.A = A
        self.slices = slices
        new_shape = np.arange(A.shape[0])[slices[0]].shape + np.arange(A.shape[1])[slices[1]].shape
        super().__init__(dtype=A.dtype, shape=new_shape)

    def _matmat(self, X: Array) -> Array:
        xnp = self.xnp
        start_slices, end_slices = self.slices
        device = xnp.get_device(X)
        Y = xnp.zeros(shape=(self.A.shape[-1], X.shape[-1]), dtype=self.dtype, device=device)
        Y = xnp.update_array(Y, X, end_slices)
        output = self.A @ Y
        return output[start_slices]

    def _rmatmat(self, X: Array) -> Array:
        xnp = self.xnp
        start_slices, end_slices = self.slices
        device = xnp.get_device(X)
        Y = xnp.zeros(shape=(X.shape[0], self.A.shape[0]), dtype=self.dtype, device=device)
        Y = xnp.update_array(Y, X, ..., start_slices)
        output = Y @ self.A
        return output[..., end_slices]

    def __str__(self):
        has_length = hasattr(self.slices[0], '__len__')
        if has_length:
            has_many = (len(self.slices[0]) > 5 or len(self.slices[1]) > 5)
            if has_many:
                return f"{str(self.A)}[slc1, slc2]"
        return f"{str(self.A)}[{self.slices[0]},{self.slices[1]}]"


class Jacobian(LinearOperator):
    """ Jacobian (linearization) of a function f: R^n -> R^m at point x.
        Matrix has shape (m, n)

    Args:
        f (callable): Function representing the mapping from R^n to R^m.
        x (array_like): 1-D array representing the point at which to compute the Jacobian.

    Example:
        >>> def f(x):
        ...     return  jnp.array([x[0]**2, x[1]**3, jnp.sin(x[2])])
        >>> x =  jnp.array([1, 2, 3])
        >>> op = Jacobian(f, x)
    """
    def __init__(self, f, x):
        self.f = f
        self.x = x
        # could perhaps relax this with automatic reshaping of x and y
        # assert len(x.shape) == 1, "x must be a vector"
        y_shape = f(x).shape
        # assert len(y_shape) == 1, "y must be a vector"

        super().__init__(dtype=x.dtype, shape=(y_shape[0], x.shape[0]))

    def _matmat(self, X):
        fn = self.xnp.vmap(partial(self.xnp.jvp_derivs, self.f, (self.x, )))
        out = fn((X.T, )).T
        if self.xnp.__name__ == 'cola.torch_fns':  # pytorch converts to double silently
            out = out.to(dtype=self.dtype)
        return out

    def _rmatmat(self, X):
        def vjp(v):
            return self.xnp.vjp_derivs(self.f, (self.x, ), v)

        fn = self.xnp.vmap(vjp)
        out = fn(X)[0]
        if self.xnp.__name__ == 'cola.torch_fns':  # pytorch converts to double silently
            out = out.to(dtype=self.dtype)
        return out

    def __str__(self):
        return "J"


class Hessian(LinearOperator):
    """ Hessian of a scalar function f: R^n -> R at point x.
        Matrix has shape (n, n)

    Args:
        f (callable): Function representing the mapping from R^n to R.
        x (array_like): 1-D array representing the point at which to compute the Hessian.

    Example:
        >>> def f(x):
        ...     return x[1]**3+np.sin(x[2])
        >>> x =  jnp.array([1, 2, 3])
        >>> op = Hessian(f, x)
    """
    def __init__(self, f, x):
        self.f, self.x = f, x
        assert len(x.shape) == 1, "x must be a vector"
        super().__init__(dtype=x.dtype, shape=(x.shape[0], x.shape[0]))

    def _matmat(self, X):
        xnp = self.xnp
        mvm = partial(xnp.jvp_derivs, xnp.grad(self.f), (self.x, ), create_graph=False)
        out = xnp.vmap(mvm)((X.T, )).T
        return out

    def __str__(self):
        return "H"


class Permutation(LinearOperator):
    """ Permutation matrix.

    Args:
        perm (array_like): 1-D array representing the permutation.
        dtype (optional): specify the dtype to operate on (not int)

    Example:
        >>> P = Permutation(np.array([1, 0, 3, 2]))
    """
    def __init__(self, perm, dtype=None):
        self.perm = perm
        fns = get_library_fns(self.perm.dtype)
        # Need to map dtype back to float
        dtype = fns.float32 if dtype is None else dtype
        super().__init__(dtype=dtype, shape=(len(perm), len(perm)))

    def _matmat(self, v):
        return v[self.perm]


@parametric
class Concatenated(LinearOperator):
    """ Produces a linear operator equivalent to concatenating
        a collection of matrices Ms along specified axis

    Args:
        *Ms (array_like): Sequence of matrices representing the blocks.
        axis (int, optional): specify which axis to concatenate on (0 or 1)
    Example:
        >>> M1 = jnp.array([[1, 2], [3, 4]])
        >>> M2 = jnp.array([[5, 6], [7, 8]])
        >>> A = Concatenated(M1, M2, axis=1)
        >>> A.shape
        >>> (2,4)
    """
    def __init__(self, *Ms, axis=0):
        self.Ms = Ms
        assert all(M.shape[axis] == Ms[0].shape[axis] for M in Ms), \
            f"Trying to concatenate matrices of different sizes {[M.shape for M in Ms]}"
        concat_size = sum(M.shape[axis] for M in Ms)
        shape = (Ms[0].shape[0], concat_size) if axis == 1 else (concat_size, Ms[0].shape[1])
        self.axis = axis
        super().__init__(Ms[0].dtype, shape)

    def _matmat(self, V):
        return self.xnp.concat([M @ V for M in self.Ms], axis=self.axis)


class ConvolveND(LinearOperator):
    """ n-Dimensional convolution Linear operator (only works in jax right now.) """
    def __init__(self, filter, array_shape, mode='same'):
        assert filter.dtype in [np.float32, np.float64], "Only supporting jax right now"
        self.filter = filter
        self.array_shape = array_shape
        assert mode == 'same'
        import jax.numpy as jnp
        super().__init__(dtype=filter.dtype, shape=(np.prod(array_shape), jnp.prod(array_shape)))
        self.conv = self.xnp.vmap(partial(self.xnp.convolve, in2=filter, mode=mode))

    def _matmat(self, X):
        Z = X.T.reshape(X.shape[-1], *self.array_shape)
        return self.conv(Z).reshape(X.shape[-1], -1).T


class Householder(LinearOperator):
    """ Householder rotation matrix."""
    def __init__(self, vec, beta=2.):
        super().__init__(shape=(vec.shape[-2], vec.shape[-2]), dtype=vec.dtype)
        self.vec = vec
        self.beta = self.xnp.array(beta, dtype=vec.dtype, device=self.device)

    def _matmat(self, X: Array) -> Array:
        xnp = self.xnp
        angle = xnp.sum(X * xnp.conj(self.vec), axis=-2, keepdims=True)
        out = X - self.beta * angle * self.vec
        return out


class Kernel(LinearOperator):
    """ Kernel operator based on a given function f where the matvec is evaluated on the fly.
    That is, [Kv]_i = \\sum_{j} f(x1_i, x2_j) v_j.
    The variables block_size1 and block_size2 determine the memory usage of the matvec
    and matmat operations.
        Args:
            x1 (array): N-D array
            x2 (array): N-D array
            fn (callable): function that defines the kernel
            block_size1 (int): block size for x1
            block_size2 (int): block size for x2
    """
    def __init__(self, x1, x2, fn, block_size1, block_size2):
        self.x1 = x1
        self.x2 = x2
        self.fn = fn
        self.block_size1 = block_size1
        self.block_size2 = block_size2
        super().__init__(dtype=x1.dtype, shape=(x1.shape[0], x2.shape[0]))
        self.iters1 = self.shape[0] // block_size1
        self.iters2 = self.shape[1] // block_size2

    def _matmat(self, V):
        xnp = self.xnp
        out = xnp.zeros(shape=V.shape, dtype=V.dtype, device=V.device)
        for idx in range(self.iters1):
            fit1 = None if idx + 1 == self.iters1 else (idx + 1) * self.block_size1
            loc1 = slice(idx * self.block_size1, fit1)
            update = xnp.zeros(shape=(self.x1[loc1].shape[0], V.shape[1]), dtype=V.dtype, device=V.device)
            for jdx in range(self.iters2):
                fit2 = None if jdx + 1 == self.iters2 else (jdx + 1) * self.block_size2
                loc2 = slice(jdx * self.block_size2, fit2)
                update += self.fn(self.x1[loc1], self.x2[loc2]) @ V[loc2]
            out = xnp.update_array(out, update, loc1)
        return out

    def __str__(self):
        return "Ker(x1, x2, fn)"


class FFT(LinearOperator):
    """ FFT matrix. Uses convention so matrix is unitary."""
    def __init__(self, n, dtype=None):
        super().__init__(shape=(n, n), dtype=dtype, annotations={cola.Unitary})

    def _matmat(self, X):
        return self.xnp.fft(X, axis=0, norm='ortho')

    def _rmatmat(self, X):
        return self.xnp.ifft(X.conj(), axis=1, norm='ortho').conj()


def FIM(logits_fn, theta):
    """ Fisher information matrix for a probability model log p(y|theta)
        where p is a classifier probability distribution. Averages over batch dimensions.

        Args:
            logit_fn function that maps parameters to logits of shape (*, n_classes)
            theta (array_like): parameter vector to eval Fisher at

        Returns:
            Hessian(KL(p(y|theta')||p(y|theta))) (w.r.t. theta)
    """
    xnp = get_library_fns(theta.dtype)
    probs = xnp.softmax(logits_fn(theta), axis=-1)

    def entropy(theta):
        log_probs = xnp.log_softmax(logits_fn(theta), axis=-1)
        return -xnp.sum(probs * log_probs, axis=-1).mean()
