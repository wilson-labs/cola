from plum import dispatch
from cola.operator_base import Array
from cola.operator_base import LinearOperator
from typing import Callable


@dispatch
def apply_unary(f: Callable, A: LinearOperator, tol=1e-7, info=False, pbar=True) -> Array:
    raise NotImplementedError
