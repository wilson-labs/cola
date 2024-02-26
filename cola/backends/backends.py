from cola.utils import export
from types import ModuleType
import numpy as np


@export
def get_library_fns(dtype):
    """ Given a dtype e.g. jnp.float32 or torch.complex64, returns the appropriate
        namespace for standard array functionality (either torch_fns or jax_fns)."""
    try:
        from jax import numpy as jnp
        if dtype in [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128, jnp.int32, jnp.int64]:
            from cola.backends import jax_fns as fns
            return fns
    except ImportError:
        pass
    try:
        import torch
        if dtype in [torch.float32, torch.float64, torch.complex64, torch.complex128, torch.int32, torch.int64]:
            from cola.backends import torch_fns as fns
            return fns
    except ImportError:
        pass

    if dtype in [np.float32, np.float64, np.complex64, np.complex128, np.int32, np.int64]:
        from cola.backends import np_fns as fns
        return fns
    raise ImportError("No supported array library found")


@export
def get_xnp(backend: str) -> ModuleType:
    try:
        match backend:
            case "torch":
                from cola.backends import torch_fns as fns
                return fns
            case "jax":
                from cola.backends import jax_fns as fns
                import jax
                jax.config.update('jax_platform_name', 'cpu')  # Force tests to run tests on CPU
                # TODO: do we actually want this here?
                return fns
            case "numpy":
                from cola.backends import np_fns as fns
                return fns
            case _:
                raise ValueError(f"Unknown backend {backend}.")
    except ImportError:
        raise RuntimeError(f"Could not import {backend}. It is likely not installed.")


@export
class AutoRegisteringPyTree(type):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls._dynamic = cls._dynamic.copy()
        import optree
        optree.register_pytree_node_class(cls, namespace='cola')
        try:
            import jax
            jax.tree_util.register_pytree_node_class(cls)
        except ImportError:
            pass
        try:
            # TODO: when pytorch migrates to optree, switch as well
            import torch

            def tree_flatten(self):
                return self.tree_flatten()

            def tree_unflatten(ctx, children):
                return cls.tree_unflatten(children, ctx)

            torch.utils._pytree._register_pytree_node(cls, tree_flatten, tree_unflatten)
        except ImportError:
            pass
