__version__ = '0.1.0'
# from .ops import LinearOperator
import cola.ops
from .basic_operations import *
from .linalg import *
# from .ops import *
from .utils import *
# import cola.ops
#  from .ops import LinearOperator
#  from .ops import *
#  try:
#      import jax
#      # walk through each of the ops in .ops and register them
#      # with jax.tree_util.register_pytree_node_class()
#      from . import ops
#      for name in dir(ops):
#          op = getattr(ops, name)
#          print(f"Registered {op} with JAX")
#          if isinstance(op, LinearOperator):
#              jax.tree_util.register_pytree_node_class(op)
#
#  except ImportError:
#      pass
