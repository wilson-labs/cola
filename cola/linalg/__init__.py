# __all__ = []

# import pkgutil
# import inspect

# for loader, name, is_pkg in pkgutil.walk_packages(__path__):
#     module = loader.find_module(name).load_module(name)

#     for name, value in inspect.getmembers(module):
#         if name.startswith('__') or not hasattr(value,"__doc__") or value.__doc__ is None:
#             continue

#         globals()[name] = value
#         __all__.append(name)

# from .inverse import inverse
# from .logdet import logdet
# from .svd import svd

# __all__ = ["inverse", "logdet", 'svd']

import importlib
import pkgutil
__all__ = []
for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
    module = importlib.import_module('.'+module_name,package=__name__)
    try: 
        globals().update({k: getattr(module, k) for k in module.__all__})
        __all__ += module.__all__
    except AttributeError: continue