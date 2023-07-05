from .dispatch import dispatch, parametric
import sys
def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn

import inspect
import importlib
import pkgutil
import logging

def import_every(module_name,namespace,_all,_name,condition=lambda name,value: True):
    """Import all functions from a module into the namespace and add to __all__.
    example usage: import_every("operators",globals(),__all__,__name__)
    """
    module = importlib.import_module('.'+module_name,package=_name)
    for name, value in inspect.getmembers(module):
        if name.startswith('_'):# or not hasattr(value,"__doc__") or value.__doc__ is None:
            continue
            
        if condition(name,value):
            namespace[name] = value
            _all.append(name)

def import_from_all(module_name,namespace,_all,_name):
    """Import all functions from module.__all__ into the namespace and add to __all__.
    example usage: import_every("operators",globals(),__all__,__name__)
    """
    module = importlib.import_module('.'+module_name,package=_name)
    if not hasattr(module,"__all__"):
        logging.debug(f"empty {module_name}.__all__")
        return
    namespace.update({k: getattr(module, k) for k in module.__all__})
    _all += module.__all__

__all__ = ['dispatch','parametric','export','import_every','import_from_all']