from functools import reduce, partial
from cola.ops.operator_base import LinearOperator
from cola.ops.operator_base import Array, get_library_fns
import numpy as np
from cola.utils.dispatch import parametric
import cola

class Dense(LinearOperator):
    """ LinearOperator wrapping of a dense matrix. O(n^2) memory and time mvms.

    Args:
        A (array_like): Dense matrix to be wrapped.

    Example:
        >>> A = jnp.array([[1, 2], [3, 4]])
        >>> op = Dense(A)
    """
    def __init__(self, A: Array):
        self.A = A
        super().__init__(dtype=A.dtype, shape=A.shape)

    def _matmat(self, X: Array) -> Array:
        return self.A @ X

    def _rmatmat(self, X: Array) -> Array:
        return X @ self.A


class LowerTriangular(Dense):
    """ Lower Triangular Linear Operator. (Wraps dense)"""
    pass


class Sparse(LinearOperator):
    """ Sparse CSR linear operator.

    Args:
        data (array_like): 1-D array representing the nonzero values of the sparse matrix.
        indices (array_like): 1-D array representing the column indices of the nonzero values.
        indptr (array_like): 1-D array representing the index pointers for the rows of the matrix.
        shape (tuple): Shape of the sparse matrix.

    Example:
        >>> data = jnp.array([1, 2, 3, 4, 5, 6])
        >>> indices = jnp.array([0, 2, 1, 0, 2, 1])
        >>> indptr = jnp.array([0, 2, 4, 6])
        >>> shape = (3, 3)
        >>> op = Sparse(data, indices, indptr, shape)
    """
    def __init__(self, data, indices, indptr, shape):
        super().__init__(dtype=data.dtype, shape=shape)
        self.A = self.ops.sparse_csr(indptr, indices, data)

    def _matmat(self, V):
        return self.A @ V


class ScalarMul(LinearOperator):
    """ Linear Operator representing scalar multiplication"""
    def __init__(self, c, shape, dtype=None):
        self.c = c
        super().__init__(dtype=dtype or type(c), shape=shape)

    def _matmat(self, v):
        return self.c * v

    def __str__(self):
        return f"{self.c}*"


class Identity(ScalarMul):
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
        super().__init__(1, shape, dtype)


def I_like(A: LinearOperator) -> Identity:
    """ A function that produces an Identity operator with the same
        shape and dtype as A """
    return Identity(dtype=A.dtype, shape=A.shape)


@parametric
class Product(LinearOperator):
    """ Matrix Multiply Product of Linear ops """
    def __init__(self, *Ms):
        self.Ms = tuple(cola.fns.lazify(M) for M in Ms)
        for M1, M2 in zip(Ms[:-1], Ms[1:]):
            if M1.shape[-1] != M2.shape[-2]:
                raise ValueError(f"dimension mismatch {M1.shape} vs {M2.shape}")
        shape = (Ms[0].shape[-2], Ms[-1].shape[-1])
        dtype = Ms[0].dtype
        super().__init__(dtype, shape)

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
        shape = Ms[0].shape
        for M in Ms:
            if M.shape != shape:
                raise ValueError(f"dimension mismatch {M.shape} vs {shape}")
        dtype = Ms[0].dtype
        super().__init__(dtype, shape)

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
        dtype = Ms[0].dtype
        super().__init__(dtype, shape)

    def _matmat(self, v):
        ev = v.reshape(*[Mi.shape[-1] for Mi in self.Ms], -1)
        for i, M in enumerate(self.Ms):
            ev_front = self.ops.moveaxis(ev, i, 0)
            shape = M.shape[0], *ev_front.shape[1:]
            Mev_front = (M @ ev_front.reshape(M.shape[-1], -1)).reshape(shape)
            ev = self.ops.moveaxis(Mev_front, 0, i)
        return ev.reshape(self.shape[-2], ev.shape[-1])

    def to_dense(self):
        Ms = [M.to_dense() if isinstance(M, LinearOperator) else M for M in self.Ms]
        return reduce(self.ops.kron, Ms)

    def __str__(self):
        return "⊗".join(str(M) for M in self.Ms)


def kronsum(A, B):
    xnp = get_library_fns(A.dtype)
    IA = xnp.eye(A.shape[-2])
    IB = xnp.eye(B.shape[-2])
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
        dtype = Ms[0].dtype
        super().__init__(dtype, shape)

    def _matmat(self, v):
        ev = v.reshape(*[Mi.shape[-1] for Mi in self.Ms], -1)
        out = 0 * ev
        xnp = self.ops
        for i, M in enumerate(self.Ms):
            ev_front = xnp.moveaxis(ev, i, 0)
            Mev_front = (M @ ev_front.reshape(M.shape[-1], -1)).reshape(
                M.shape[0], *ev_front.shape[1:])
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
        super().__init__(Ms[0].dtype, shape)

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
        y = self.ops.concatenate(y, axis=0)  # concatenate over rep axis
        return y

    def to_dense(self):
        Ms_all = [M for M, c in zip(self.Ms, self.multiplicities) for _ in range(c)]
        Ms_all = [Mi.to_dense() if isinstance(Mi, LinearOperator) else Mi for Mi in Ms_all]
        return self.ops.block_diag(*Ms_all)

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
        super().__init__(dtype=diag.dtype, shape=(len(diag), ) * 2)
        self.diag = diag

    def _matmat(self, X: Array) -> Array:
        return self.diag[:, None] * X

    def _rmatmat(self, X: Array) -> Array:
        return self.diag[None, :] * X

    def to_dense(self):
        return self.ops.diag(self.diag)

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
        super().__init__(dtype=beta.dtype, shape=(beta.shape[0], beta.shape[0]))
        alpha, beta = ensure_vec_is_matrix(alpha), ensure_vec_is_matrix(beta)
        gamma = ensure_vec_is_matrix(gamma)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def _matmat(self, X: Array) -> Array:
        xnp = self.ops
        aux_alpha = xnp.zeros(shape=X.shape, dtype=X.dtype)
        aux_gamma = xnp.zeros(shape=X.shape, dtype=X.dtype)

        output = self.beta * X

        ind = xnp.array([i + 1 for i in range(self.alpha.shape[0])])
        aux_alpha = xnp.update_array(aux_alpha, self.alpha * X[:-1], ind)
        output += aux_alpha

        ind = xnp.array([i for i in range(self.gamma.shape[0])])
        aux_gamma = xnp.update_array(aux_gamma, self.gamma * X[1:], ind)
        output += aux_gamma
        return output


def ensure_vec_is_matrix(vec):
    if len(vec.shape) == 1:
        vec = vec.reshape(-1, 1)
    return vec


@parametric
class Transpose(LinearOperator):
    """ Transpose of a Linear Operator"""
    def __init__(self, A):
        super().__init__(dtype=A.dtype, shape=(A.shape[1], A.shape[0]))
        self.A = A

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
        super().__init__(dtype=A.dtype, shape=(A.shape[1], A.shape[0]))
        self.A = A

    def _matmat(self, x):
        return self.ops.conj(self.A._rmatmat(self.ops.conj(x).T)).T

    def _rmatmat(self, x):
        return self.ops.conj(self.A._matmat(self.ops.conj(x).T)).T

    def __str__(self):
        return f"{str(self)}*"


@parametric
class Sliced(LinearOperator):
    """ Slicing of another linear operator A.
        Equivalent to A[slices[0], :][:, slices[1]] """
    def __init__(self, A, slices):
        self.A = A
        self.slices = slices
        xnp = A.ops
        new_shape = xnp.arange(A.shape[0])[slices[0]].shape + xnp.arange(A.shape[1])[slices[1]].shape
        super().__init__(dtype=A.dtype, shape=new_shape)

    def _matmat(self, X: Array) -> Array:
        xnp = self.ops
        start_slices, end_slices = self.slices
        Y = xnp.zeros(shape=(self.A.shape[-1], X.shape[-1]), dtype=self.dtype)
        Y = xnp.update_array(Y, X, end_slices)
        output = self.A @ Y
        return output[start_slices]

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
        assert len(x.shape) == 1, "x must be a vector"
        y_shape = f(x).shape
        assert len(y_shape) == 1, "y must be a vector"

        super().__init__(dtype=x.dtype, shape=(y_shape[0], x.shape[0]))

    def _matmat(self, X):
        # primals = self.x[:,None]+self.ops.zeros((1,X.shape[1],), dtype=self.x.dtype)
        return self.ops.vmap(partial(self.ops.jvp_derivs, self.f, (self.x, )))((X.T, )).T

    def _rmatmat(self, X):
        # primals = self.x[None,:]+self.ops.zeros((X.shape[0],1), dtype=self.x.dtype)
        return self.ops.vmap(partial(self.ops.vjp_derivs, self.f, (self.x, )))((X, ))

    def __str__(self):
        return "J"


@parametric
class SelfAdjoint(LinearOperator):
    """ SelfAdjoint property for Linearops. """
    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], LinearOperator):
            self.A = args[0]
            self._matmat = self.A._matmat
            super().__init__(self.A.dtype, self.A.shape)
        else:
            super().__init__(*args, **kwargs)

    def _rmatmat(self, X: Array) -> Array:
        return self.ops.conj(self._matmat(self.ops.conj(X).T)).T

    @property
    def T(self):
        return self

    @property
    def H(self):
        return self

    def __str__(self):
        if hasattr(self, 'A'):
            return f"S[{str(self.A)}]"
        else:
            return super().__str__()


class Hessian(SelfAdjoint):
    """ Hessian of a scalar function f: R^n -> R at point x.
        Matrix has shape (n, n)

    Args:
        f (callable): Function representing the mapping from R^n to R^m.
        x (array_like): 1-D array representing the point at which to compute the Jacobian.

    Example:
        >>> def f(x):
        ...     return x[1]**3+np.sin(x[2])
        >>> x =  jnp.array([1, 2, 3])
        >>> op = Hessian(f, x)
    """
    def __init__(self, f, x):
        self.f = f
        self.x = x
        assert len(x.shape) == 1, "x must be a vector"
        super().__init__(dtype=x.dtype, shape=(x.shape[0], x.shape[0]))

    def _matmat(self, X):
        xnp = self.ops
        mvm = partial(xnp.jvp_derivs, xnp.grad(self.f), (self.x, ), create_graph=False)
        # hack to make it work with pytorch
        if xnp.__name__=='cola.torch_fns':
            out = xnp.zeros((self.shape[-1], X.shape[-1]), dtype=self.dtype)
            for i in range(X.shape[1]):
                out[:, i] = mvm(X[:, i])
            return out
        else:
            return xnp.vmap(mvm)((X.T, )).T

    def __str__(self):
        return "H"


class Permutation(LinearOperator):
    """ Permutation matrix.

    Args:
        perm (array_like): 1-D array representing the permutation.

    Example:
        >>> P = Permutation(np.array([1, 0, 3, 2]))
    """
    def __init__(self, perm):
        self.perm = perm
        super().__init__(dtype=perm.dtype, shape=(len(perm), len(perm)))


# @parametric
class Concatenated(LinearOperator):
    pass


class ConvolveND(LinearOperator):
    """ n-Dimensional convolution Linear operator (only works in jax right now.) """
    def __init__(self, filter, array_shape, mode='same'):
        self.filter = filter
        self.array_shape = array_shape
        assert mode == 'same'
        super().__init__(dtype=filter.dtype, shape=(np.prod(array_shape), jnp.prod(array_shape)))
        self.conv = self.ops.vmap(partial(self.ops.convolve, in2=filter, mode=mode))

    def _matmat(self, X):
        Z = X.T.reshape(X.shape[-1], *self.array_shape)
        return self.conv(Z).reshape(X.shape[-1], -1).T


Symmetric = SelfAdjoint


@parametric
class PSD(SelfAdjoint):
    """ Positive Semi-Definite property for Linearops.
        Implies SelfAdjoint. """
    pass


@parametric
class Unitary(LinearOperator):
    def __init__(self, A: LinearOperator):
        self.A = A
        super().__init__(dtype=A.dtype, shape=A.shape)


class Householder(SelfAdjoint):
    """ Householder rotation matrix."""
    def __init__(self, vec, beta=2.):
        super().__init__(shape=(vec.shape[-2], vec.shape[-2]), dtype=vec.dtype)
        self.vec = vec
        self.beta = self.ops.array(beta, dtype=vec.dtype)

    def _matmat(self, X: Array) -> Array:
        xnp = self.ops
        # angle = xnp.sum(X * xnp.conj(self.vec), axis=-2, keepdims=True)
        angle = xnp.sum(X * self.vec, axis=-2, keepdims=True)
        out = X - self.beta * angle * self.vec
        return out
