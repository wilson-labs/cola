from typing import Tuple
from linops.operator_base import LinearOperator
from linops.operator_base import Array
from plum import dispatch


@dispatch
def svd(A: LinearOperator, k: int, which='high', tol=1e-7, info=False) -> Tuple[Array]:
    xnp = A.ops
    output = xnp.svd(A.to_dense())
    return output
