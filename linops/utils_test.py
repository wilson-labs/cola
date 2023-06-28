import pytest
import inspect
from linops.operator_base import get_library_fns
import numpy as np


def strip_parens(string):
    return string.replace('(', '').replace(')', '')


def parametrize(cases, ids=None):
    """ Expands test cases with pytest.mark.parametrize but with argnames
        assumed and ids given by the ids=[str(case) for case in cases] """
    def decorator(test_fn):
        argnames = ','.join(inspect.getfullargspec(test_fn).args)
        theids = [strip_parens(str(case)) for case in cases] if ids is None else ids
        return pytest.mark.parametrize(argnames, cases, ids=theids)(test_fn)

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
