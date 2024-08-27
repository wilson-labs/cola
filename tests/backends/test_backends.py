import numpy as np
import pytest

from cola.backends.backends import check_valid_dtype


def test_check_valid_dtype():
    assert check_valid_dtype(np.array(0.1), dtype=True, alloc_fn=_as) is False

    torch = pytest.importorskip("torch")
    assert check_valid_dtype(torch.tensor(0.1), dtype=torch.Tensor, alloc_fn=_to) is False

    jnp = pytest.importorskip("jax.numpy")
    assert check_valid_dtype(jnp.array(0.1), dtype=True, alloc_fn=_as) is False


def _to(x, y):
    return x.to(y)


def _as(x, y):
    return x.astype(y)
