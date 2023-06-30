__version__ = '0.1.0'
from .linear_algebra import *
from cola.linalg import *
from cola.ops import *
from cola.utils import *
# import cola.operators
#  from .operator_base import LinearOperator
#  from .operators import *
#  try:
#      import jax
#      # walk through each of the operators in .operators and register them
#      # with jax.tree_util.register_pytree_node_class()
#      from . import operators
#      for name in dir(operators):
#          op = getattr(operators, name)
#          print(f"Registered {op} with JAX")
#          if isinstance(op, LinearOperator):
#              jax.tree_util.register_pytree_node_class(op)
#
#  except ImportError:
#      pass
