import numpy as np
import pytest
import torch
from jax import numpy as jnp

from cola.backends.backends import check_valid_dtype


def test_check_valid_dtype():
    def _to(x, y):
        return x.to(y)

    def _as(x, y):
        return x.astype(y)

    cases = [
        (torch.tensor(0.1), torch.Tensor, _to),
        (np.array(0.1), True, _as),
        (jnp.array(0.1), True, _as),
    ]
    for arr, dty, alloc in cases:
        with pytest.raises(TypeError):
            check_valid_dtype(arr, dtype=dty, alloc_fn=alloc)
