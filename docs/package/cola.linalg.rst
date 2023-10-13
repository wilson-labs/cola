Linear Algebra
=====================================

Inverses & Solves
-----------------

.. autofunction:: cola.linalg.inv
.. autofunction:: cola.linalg.solve

Algorithms
~~~~~~~~~~

   .. autoclass:: cola.linalg.Auto
    :members:
    :show-inheritance:
   * if A is PSD and small, uses Cholesky
   * if A is not PSD and small, uses LU
   * if A is PSD and large, uses CG
   * if A is not PSD and large, uses GMRES

   .. autoclass:: cola.linalg.CG
    :members:
    :show-inheritance:
   .. autoclass:: cola.linalg.GMRES
    :members:
    :show-inheritance:
   .. autoclass:: cola.linalg.LU
    :members:
    :show-inheritance:
   .. autoclass:: cola.linalg.Cholesky
    :members:
    :show-inheritance:


Eigenvalues and Eigenvectors
-----

.. autofunction:: cola.linalg.eig

Algorithms
~~~~~~~~~~

   .. autoclass:: cola.linalg.Auto
    :members:
    :show-inheritance:
   * if A is Hermitian and small, uses Eigh
   * if A is not Hermitian and small, use Eig
   * if A is Hermitian and large, uses Lanczos
   * if A is not Hermitian and large, uses Arnoldi

   .. autoclass:: cola.linalg.Arnoldi
    :members:
    :show-inheritance:
   .. autoclass:: cola.linalg.Lanczos
    :members:
    :show-inheritance:
   .. autoclass:: cola.linalg.Eig
    :members:
    :show-inheritance:
   .. autoclass:: cola.linalg.Eigh
    :members:
    :show-inheritance:
   .. autoclass:: cola.linalg.PowerIteration
    :members:
    :show-inheritance:


Matrix Functions: exp, log, sqrt, isqrt, etc.
----------------------------------------------

.. autofunction:: cola.linalg.exp
.. autofunction:: cola.linalg.log
.. autofunction:: cola.linalg.sqrt
.. autofunction:: cola.linalg.isqrt
.. autofunction:: cola.linalg.pow
.. autofunction:: cola.linalg.apply_unary

Algorithms
~~~~~~~~~~

   .. autoclass:: cola.linalg.Auto
    :members:
    :show-inheritance:
   * if A is Hermitian and small, uses Eigh
   * if A is not Hermitian and small, uses Eig
   * if A is Hermitian and large, uses Lanczos
   * if A is not Hermitian and large, uses Arnoldi

   .. autoclass:: cola.linalg.Lanczos
    :members:
    :show-inheritance:
   .. autoclass:: cola.linalg.Arnoldi
    :members:
    :show-inheritance:
   .. autoclass:: cola.linalg.Eig
    :members:
    :show-inheritance:
   .. autoclass:: cola.linalg.Eigh
    :members:
    :show-inheritance:

Trace, Diagonal, Frobenius Norm
--------------------------------

.. autofunction:: cola.linalg.trace
.. autofunction:: cola.linalg.diag
.. autofunction:: cola.linalg.norm

Algorithms
~~~~~~~~~~

   .. autoclass:: cola.linalg.Auto
    :no-index:
    :members:
    :show-inheritance:
   * if :math:`\tfrac{1}{\sqrt{10n}} < \epsilon` use Hutch :math:`O(\tfrac{1}{\delta^2})`
   * otherwise use Exact :math:`O(n)`

   .. autoclass:: cola.linalg.Hutch
    :members:
   .. autoclass:: cola.linalg.Exact
    :members:
   .. autoclass:: cola.linalg.HutchPP
    :members:

Log Determinants
-----------------

.. autofunction:: cola.linalg.logdet
.. autofunction:: cola.linalg.slogdet

Log Algs
~~~~~~~~~~

   .. autoclass:: cola.linalg.Auto
    :members:
    :show-inheritance:
   * if A is PSD and small, uses Cholesky
   * if A is not PSD and small, uses LU
   * if A is PSD and large, uses Lanczos
   * if A is not PSD and large, uses Arnoldi

   .. autoclass:: cola.linalg.Lanczos
    :members:
    :show-inheritance:
   .. autoclass:: cola.linalg.Arnoldi
    :members:
    :show-inheritance:
   .. autoclass:: cola.linalg.Cholesky
    :members:
    :show-inheritance:
   .. autoclass:: cola.linalg.LU
    :members:
    :show-inheritance:

Trace Algs
~~~~~~~~~~

   .. autoclass:: cola.linalg.Auto
    :members:
   * if :math:`\tfrac{1}{\sqrt{10n}} < \epsilon` use Hutch :math:`O(\tfrac{1}{\delta^2})`
   * otherwise use Exact :math:`O(n)`

   .. autoclass:: cola.linalg.Hutch
    :members:
   .. autoclass:: cola.linalg.Exact
    :members:
   .. autoclass:: cola.linalg.HutchPP
    :members:
