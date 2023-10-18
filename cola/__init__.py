from cola.utils import import_from_all

# Read version number as written by setuptools_scm
try:
    from .version import version as __version__
except Exception:  # pragma: no cover
    __version__ = "Unknown"  # pragma: no cover

__all__ = []
# for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
import_from_all("fns", globals(), __all__, __name__)
import_from_all("annotations", globals(), __all__, __name__)
import_from_all("linalg", globals(), __all__, __name__)
import_from_all("utils", globals(), __all__, __name__)

__all__.append("LinearOperator")
# import_from_all("ops", globals(), __all__,__name__)
