# import torch
# from torch.utils._pytree import tree_flatten
# from optree import tree_flatten
from jax import numpy as jnp
from jax.tree_util import tree_flatten
import sepes as sp
from cola.ops.operators_sepes import Dense
# from cola.ops.operators import Dense

# A = torch.randn(3, 3)
A = jnp.array([[1, 2, 3], [1, 2, 3]])
AOp = Dense(A)
aux = tree_flatten(AOp)
print(type(aux[0][0]))
print(tree_flatten(AOp))
print(sp.tree_diagram(AOp))
