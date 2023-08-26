import inspect
import itertools
import os
import pytest
from types import ModuleType

from cola.ops import get_library_fns
import numpy as np


# Try importing jax and torch, for the get_framework function
try:
    from . import jax_fns
except ImportError:
    jax_fns = None

try:
    from . import torch_fns
except ImportError:
    torch_fns = None


def strip_parens(string):
    return string.replace('(', '').replace(')', '')


def _add_marks(case):
    # This function is maybe hacky, but it adds marks based on the names of the parameters supplied
    # In particular, it adds the 'torch', 'jax', and 'big' marks
    case = case if isinstance(case, list) or isinstance(case, tuple) else [case]
    marks = []
    args = tuple(str(arg) for arg in case)
    if any('torch' in arg for arg in args):
        marks.append(pytest.mark.torch)
    if any('jax' in arg for arg in args):
        marks.append(pytest.mark.jax)
    if any('big' in arg for arg in args):
        marks.append(pytest.mark.big)
    return pytest.param(*case, marks=marks)


def parametrize(*cases, ids=None):
    """ Expands test cases with pytest.mark.parametrize but with argnames
        assumed and ids given by the ids=[str(case) for case in cases] """
    if len(cases) > 1:
        all_cases = [tuple(elem) for elem in itertools.product(*cases)]
    else:
        all_cases = cases[0]

    # Potentially add marks
    all_cases = [_add_marks(case) for case in all_cases]

    def decorator(test_fn):
        argnames = ','.join(inspect.getfullargspec(test_fn).args)
        theids = [strip_parens(str(case)) for case in all_cases] if ids is None else ids
        return pytest.mark.parametrize(argnames, all_cases, ids=theids)(test_fn)

    return decorator


def relative_error(v, w):
    xnp = get_library_fns(v.dtype)
    abs_err = xnp.norm(v - w)
    denom = (xnp.norm(v) + xnp.norm(w)) / 2.
    rel_err = abs_err / max(denom, 1e-16)
    return rel_err


def construct_e_vec(i, size):
    e_vec = np.zeros(shape=(size, ))
    e_vec[i] = 1.0
    return e_vec


def generate_lower_from_diag(diag, dtype=np.float32, seed=None, orthogonalize=True):
    if seed:
        np.random.seed(seed=seed)
    Q = np.random.normal(size=(diag.shape[0], diag.shape[0])).astype(dtype)
    if orthogonalize:
        Q, _ = np.linalg.qr(Q, mode='reduced')
    Q = np.tril(Q)
    np.fill_diagonal(Q, diag)
    return Q


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


def generate_beta_spectrum(coeff, scale, size, alpha=1., beta=1., seed=48, dtype=np.float32,
                           y_min=1e-6):
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
    x = np.linspace(0, 1, num=size + 1)[:-1].astype(dtype)
    y = 1 - x**coeff
    y *= scale
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


def get_xnp(backend: str) -> ModuleType:
    match backend:
        case "torch":
            if torch_fns is None:  # There was an import error with torch
                raise RuntimeError("Could not import torch. It is likely not installed.")
            else:
                return torch_fns
        case "jax":
            if jax_fns is None:  # There was an import error with jax
                raise RuntimeError("Could not import jax. It is likely not installed.")
            else:
                from jax.config import config
                config.update('jax_platform_name', 'cpu')  # Force tests to run tests on CPU
                # config.update("jax_enable_x64", True)
                return jax_fns
        case _:
            raise ValueError(f"Unknown backend {backend}.")
