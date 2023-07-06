import cola
import numpy as np
from cola.ops import Diagonal,Identity,Sum,BlockDiag,ScalarMul,Dense
from cola.algorithms import exact_diag, approx_diag
from jax.config import config
from cola.utils_test import parametrize, relative_error
from cola import jax_fns
from cola import torch_fns
config.update('jax_platform_name', 'cpu')

@parametrize([torch_fns])#, jax_fns])
def test_exact_diag(xnp):
    A = Dense(xnp.array([[1,2,3],[4,5,6],[7,8,9.]],dtype=xnp.float32))
    for u in [-2,-1,0,1,2]:
        d1 = exact_diag(A,u)
        d2 = xnp.diag(A.to_dense(),u)
        print(d1,d2)
        assert relative_error(d1, d2)<1e-5


@parametrize([torch_fns])#, jax_fns])
def test_approx_diag(xnp):
    A = Dense(xnp.array([[1,2,3],[4,5,6],[7,8,9.]],dtype=xnp.float32))
    for u in [-2,-1,0,1,2]:
        d1 = approx_diag(A,u,tol=3e-2)
        d2 = xnp.diag(A.to_dense(),u)
        print(d1,d2)
        assert relative_error(d1, d2)<6e-2
# %%
