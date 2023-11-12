import inspect
import itertools
import pytest

from cola.backends import get_library_fns, get_xnp, all_backends
from cola.backends.np_fns import NumpyNotImplementedError
import numpy as np
import functools

get_xnp = get_xnp


def strip_parens(string):
    return string.replace('(', '').replace(')', '')


def _add_marks(case, is_tricky=False):
    # This function is maybe hacky, but it adds marks based on the names of the parameters supplied
    # In particular, it adds the 'torch', 'jax', and 'big' marks
    case = case if isinstance(case, list) or isinstance(case, tuple) else [case]
    marks = []
    args = tuple(str(arg) for arg in case)
    if any('big' in arg for arg in args):
        marks.append(pytest.mark.big)
    if is_tricky:
        marks.append(pytest.mark.tricky)
    for backend in all_backends:
        if any(backend in arg for arg in args):
            marks.append(getattr(pytest.mark, backend))
    return pytest.param(*case, marks=marks)


def ignore_numpy_notimplemented(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NumpyNotImplementedError:
            pass

    return wrapper


def index(cases, idx):
    match idx:
        case slice() as s:
            return cases[s]
        case list() as li:
            return li
        case tuple() as t:
            return t
        case _:
            return (idx, )


class parametrize:
    """ Expands test cases with pytest.mark.parametrize but with argnames
            assumed and ids given by the ids=[str(case) for case in cases]

        Cases indexed using excluding will be marked with pytest.mark.tricky
        Can use no excluding to instead index which cases to include

        usage:
            @parametrize([a1,a2,...], [b1,b2,...], ...).excluding[:,[b2,b4,b5],:2,...]
            def test_fn(a,b,...):

            @parametrize([a1,a2,...], [b1,b2,...], ...).excluding[[(a1,b2,c2), (a2,b4,c5), ...]]
            def test_fn(a,b,...):

            @parametrize([a1,a2,...], [b1,b2,...], ...)[:3, [b2,b4,b5], ...]  # only include those
        """
    def __init__(self, *cases, ids=None):
        self.cases = cases
        self.ids = ids
        if len(cases) > 1:
            self.all_cases = [tuple(elem) for elem in itertools.product(*cases)]
        else:
            self.all_cases = cases[0]
        self.indexed_cases = set(self.all_cases)
        self.indexing = True

    @property
    def excluding(self):
        self.indexing = False
        self.indexed_cases = set()
        return self

    def __getitem__(self, indexed_cases):
        # multiple arguments, need to use cross product
        if isinstance(indexed_cases, tuple) and len(indexed_cases) > 1:
            expanded_indexed_cases = [index(c, t) for t, c in zip(indexed_cases, self.cases)]
            indexed_cases = {tuple(elem) for elem in itertools.product(*expanded_indexed_cases)}
        else:  # single argument
            match indexed_cases:
                case slice() as s:
                    indexed_cases = set(self.all_cases[s])
                case list() as li:
                    indexed_cases = set(li)
                case tuple() as t:
                    indexed_cases = set(t)
                case _:
                    indexed_cases = set((indexed_cases, ))
        # Potentially add marks
        assert indexed_cases - set(self.all_cases) == set(), "indexed_cases cases must be in the list of cases"
        self.indexed_cases = indexed_cases
        return self

    def __call__(self, test_fn):
        all_cases = [_add_marks(case, (case in self.indexed_cases) ^ self.indexing) for case in self.all_cases]
        argnames = ','.join(inspect.getfullargspec(test_fn).args)
        theids = [strip_parens(str(case)) for case in all_cases] if self.ids is None else self.ids
        test_fn = ignore_numpy_notimplemented(test_fn)
        return pytest.mark.parametrize(argnames, all_cases, ids=theids)(test_fn)


def relative_error(v, w, xnp=None):
    if xnp is None:
        xnp = get_library_fns(v.dtype)
    abs_err = xnp.norm(v - w)
    denom = (xnp.norm(v) + xnp.norm(w)) / 2.
    rel_err = abs_err / max(denom, 1e-16)
    return rel_err.item()


def construct_e_vec(i, size):
    e_vec = np.zeros(shape=(size, ))
    e_vec[i] = 1.0
    return e_vec


def transform_to_csr(sparse_matrix, xnp, dtype):
    data = xnp.array(sparse_matrix.data, dtype=dtype, device=None)
    indices = xnp.array(sparse_matrix.indices, dtype=xnp.int64, device=None)
    indptr = xnp.array(sparse_matrix.indptr, dtype=xnp.int64, device=None)
    return data, indices, indptr, sparse_matrix.shape


def generate_lower_from_diag(diag, dtype=np.float32, orthogonalize=True, seed=None):
    if seed:
        np.random.seed(seed=seed)
    L = np.random.normal(size=(diag.shape[0], diag.shape[0])).astype(dtype)
    if orthogonalize:
        L, _ = np.linalg.qr(L, mode='reduced')
    is_complex = True if dtype in [np.complex64, np.complex128] else False
    if is_complex:
        L += (1 / 100) * np.random.randn(*L.shape) * 1j
    L = np.tril(L)
    np.fill_diagonal(L, diag)
    return L


def generate_diagonals(diag, seed=None):
    if seed:
        np.random.seed(seed=seed)
    L = np.empty(shape=(diag.shape[0], diag.shape[0]), dtype=np.complex64)
    L.real = np.random.normal(size=(diag.shape[0], diag.shape[0])).astype(np.float32)
    L.imag = np.random.normal(size=(diag.shape[0], diag.shape[0])).astype(np.float32)
    Q, _ = np.linalg.qr(L, mode='reduced')
    A = Q.conj().T @ np.diag(diag) @ Q
    return A


def generate_pd_from_diag(diag, dtype, seed=None, normalize=True):
    if seed:
        np.random.seed(seed=seed)
    L = np.random.normal(size=(diag.shape[0], diag.shape[0])).astype(dtype)
    if normalize:
        Q, _ = np.linalg.qr(L, mode='reduced')
    else:
        Q = L
    A = Q.T @ np.diag(diag) @ Q
    return A


def generate_beta_spectrum(coeff, scale, size, alpha=1., beta=1., seed=48, dtype=np.float32, y_min=1e-6):
    if seed:
        np.random.seed(seed=seed)
    x = np.random.beta(a=alpha, b=beta, size=(size, )).astype(dtype)
    x.sort()
    y = 1 - x**coeff
    y_max = np.max(y)
    y *= scale / y_max
    y += y_min
    return y


def generate_spectrum(coeff, scale, size, dtype=np.float32):
    is_complex = True if dtype in [np.complex64, np.complex128] else False
    x = np.linspace(0, 1, num=size + 1)[:-1].astype(dtype)
    y = 1 - x**coeff
    y *= scale
    if is_complex:
        random_imaginary = (scale / 100) * np.random.randn(y.shape[0]) * 1j
        y += random_imaginary
    return y


def generate_clustered_spectrum(clusters, sizes, std=0.025, seed=None, dtype=np.float32):
    assert len(clusters) == len(sizes)
    if seed:
        np.random.seed(seed=seed)

    diag = []
    for idx, cl in enumerate(clusters):
        eps = np.random.normal(scale=std, size=(sizes[idx], )).astype(dtype)
        sub_diags = np.abs(cl + eps)
        sub_diags = np.sort(sub_diags)[::-1]
        diag.append(sub_diags)
    diag = np.concatenate(diag, axis=0)
    return np.sort(diag)[::-1]
