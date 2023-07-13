""" Linear Operators in CoLA"""
from cola.utils import import_from_all, import_every
__all__ = []
import_from_all("operator_base", globals(), __all__, __name__)
#is_operator = lambda name,value: isinstance(value,type) and issubclass(value,LinearOperator)
has_docstring = lambda name, value: hasattr(value, "__doc__") and value.__doc__ is not None
import_every("operators", globals(), __all__, __name__)  #,has_docstring)
# import_from_all("decompositions", globals(), __all__, __name__)

