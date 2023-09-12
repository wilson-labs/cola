""" CoLA Backends"""
from cola.utils import import_from_all, import_every

__all__ = []
import_from_all("backends", globals(), __all__, __name__)
all_backends = ["torch", "jax"]
__all__ += ["all_backends"]