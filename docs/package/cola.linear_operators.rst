Linear Operator Base Class
==========================
The base class of all linear operators is :class:`LinearOperator`. It
implements the basic machinery for matrix vector multiplies.

To implement a new linear operator, you need to implement or call
:any:`__init__(dtype,shape)`, and implement :meth:`_matmat`.

The methods :meth:`_rmatmat`, :meth:`to_dense`, :meth:`T`, and :meth:`H` will all be
defined automatically.


.. autoclass:: cola.ops.LinearOperator
    :members: __init__, _matmat, _rmatmat, to_dense, T, H
    :member-order: bysource
