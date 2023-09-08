Annotations for Linear Operators
================================

Applying an annotation to a linear operator will add to its set of annotations attribute.

.. code-block:: python

   A = cola.SelfAdjoint(LinearOperator(...))
   A.annotations
   # Output:
   # {cola.SelfAdjoint}

These annotations can be to make simplifications, such as for transposes or the algorithms used.

.. code-block:: python

   A.isa(cola.SelfAdjoint)
   # Output:
   # True

.. automodule:: cola.annotations
    :members:
    :show-inheritance:
    :member-order: bysource