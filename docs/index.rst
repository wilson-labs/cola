Compositional Linear Algebra (CoLA)
===================================

Introduction
------------
Many areas of machine learning (ML) and science involve large-scale linear algebra problems,
such as performing eigendecompositions, solving linear systems, computing matrix exponentials,
and doing trace estimation.
The linear operators involved often have Kronecker, convolutional, block diagonal, sum, or product structure.
Yet, to exploit this structure, that is, in order to use specific algorithms that
have faster runtimes than general algorithms, a user must manually implement these efficient routines on
a case-by-case basis and be familiar with the different algorithms that exist for
different structures. This process leads to a notorious implementation bottleneck!

To eliminate this bottleneck we introduce ``CoLA``, a numerical linear algebra library designed to 
automatically exploit the structure present in a diverse set of linear operators.
To achieve this, ``CoLA`` automatically exploits compositional structure by leveraging over 70 dispatch
rules which select different algorithms for the diverse structure present in a linear
operator. Additionally, given our emphasis on ML applications, ``CoLA`` also
supports both ``PyTorch`` and ``JAX``, leverage GPU and TPU acceleration, supports low
precision, provides automatic computation of
gradients, diagonals, transposes and adjoints of linear
operators, and incorporates specialty algorithms such as SVRG and a novel
variation of Hutchinson's diagonal estimator which exploit the large-scale sum
structure of several linear operators found in ML applications.

Furthermore, regardless of whether there is structure that can be exploited or not,
``CoLA`` can be used as a general purpose numerical linear algebra package
for large-scale linear operators.
``CoLA`` provides an implementation of classical iterative algorithms for 
solving linear systems, performing eigendecompositions and more for
PSD, symmetric, non-symmetric, real and complex linear operators.

Installation
------------
``CoLA`` requires Python >= 3.10

We recommend installing via ``pip``:

.. code-block:: bash

    pip install cola

The installation requires the following packages:

* ``PyTorch`` or ``JAX`` (or both)
* ``plum-dispatch``

Or for the latest version,

.. code-block:: bash

    git clone https://github.com/wilson-labs/cola

Design Choices
--------------
``CoLA`` is designed with the following criteria in mind:

* We enable easy extensibility by allowing users to define dispatch rules and linear
  operators.
* We adhere to the same API used for dense matrix operations.
* We use multiple dispatch to exploit structure of a linear operator.
* We provide support for both PyTorch and JAX.
* We leverage automatic differentiation to define operations like transpose but also to
  derive gradients of linear operators.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   notebooks/Quick_Start.ipynb
   notebooks/LinOpIntro.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Linear Algebra

   notebooks/Linear_Solves.ipynb
   notebooks/Eigendecomposition.ipynb
   notebooks/SVD.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   notebooks/Adding_Linear_Operators.ipynb
   notebooks/Defining_Dispatch_Rules.ipynb
   notebooks/Lower_Precision.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Example Applications

   notebooks/01_PCA.ipynb
   notebooks/02_Linear_Regression.ipynb
   notebooks/03_GPs.ipynb
   notebooks/04_Spectral_Clustering.ipynb
   notebooks/05_Shrodinger_Equation.ipynb
   notebooks/06_Minimal_Surface.ipynb

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API Reference

   package/linops.operators
   package/linops.linear_algebra

.. toctree::
   :maxdepth: 1
   :caption: Research Highlight

.. toctree::
   :maxdepth: 1
   :caption: Notes

   testing.md
   CHANGELOG.md

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation

   documentation.md

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
