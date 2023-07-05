from plum import dispatch
from cola.ops import Array
from cola.ops import LinearOperator
from typing import Callable


@dispatch
def apply_unary(f: Callable, A: LinearOperator, tol=1e-7, info=False, pbar=True) -> Array:
    raise NotImplementedError
