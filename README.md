# Compositional Linear Algebra (CoLA)

[![Documentation](https://readthedocs.org/projects/emlp/badge/)](https://cola.readthedocs.io/en/latest/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mfinzi/equivariant-MLP/blob/master/docs/notebooks/colabs/all.ipynb)
[![tests](https://github.com/wilson-labs/cola/actions/workflows/python-package.yml/badge.svg)](https://github.com/wilson-labs/cola/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/wilson-labs/cola/branch/main/graph/badge.svg?token=bBnkfHv30C)](https://codecov.io/gh/wilson-labs/cola)

CoLA is a numerical linear algebra framework that exploits the structure usually found on machine learning problems and beyond.
CoLA supports both PyTorch and JAX.

## Installation
```shell
git clone https://github.com/wilson-labs/cola
```

## Features in CoLA
* Provides several compositional rules to exploit problem structure through multiple dispatch.
* Works with PyTorch and JAX
* Supports hardware acceleration through GPU an TPU (JAX).
* Supports different types of numerical precision.
* Has memory-efficient Autograd routines for different iterative algorithms.
* Provides operations for both symmetric and non-symmetric matrices.
* Runs with real and complex numbers.
* Contains several randomized linear algebra algorithms.

### Features being added
Linear Algebra Operations
- [x] inverse: $A^{-1}$
- [x] eig: $U \Lambda U^{-1}$
- [ ] diag
- [ ] trace
- [ ] exp
- [ ] logdet
- [ ] $f(A)$
      
Linear Operators
- [x] Diag
- [x] BlockDiag
- [x] Kronecker
- [x] KronSum
- [ ] Concatenated
- [ ] Triangular
- [ ] Tridiagonal
- [ ] CholeskyDecomposition
- [ ] LUDecomposition
- [ ] EigenDecomposition
      
Attributes
- [x] SelfAdjoint
- [ ] PSD
- [ ] Unitary

## Citing us
If you use CoLA, please cite the following paper:
<!--
> [Andres Potapczynski, Marc Finzi, Geoff Pleiss, and Andrew Gordon Wilson. "Exploiting Compositional Structure for Automatic and Efficient Numerical Linear Algebra." Advances in Neural Information Processing Systems (2023).]()
```
@article{potapczynski2023cola,
  title={{Exploiting Compositional Structure for Automatic and Efficient Numerical Linear Algebra}},
  author={Andres Potapczynski and Marc Finzi and Geoff Pleiss and Andrew Gordon Wilson},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```
-->

## Quick start guide
1. **Defining a dispatch rule**. The way that we exploit structure in CoLA is through
   dispatch rules. To illustrate, we will show how the dispatch rules for a diagonal
   operator work. In terms of linear solves, we write
```python
from plum import dispatch

@dispatch
def inverse(A: Diagonal, **kwargs):
    return Diagonal(1. / A.diag)
```
and once we have that dispatch rule available we can then use it in the following manner:
```python
import cola.torch_fns as xnp
from cola.operators import Diagonal
from cola.linalg.inverse import inverse

dtype = xnp.float32
diag = xnp.array([1., 0.5, 0.25, 0.1], dtype=dtype)
D = Diagonal(diag)
rhs = xnp.randn(diag.shape[0], 1, dtype=dtype)
soln = inverse(D) @ rhs
res = xnp.norm(D @ soln - rhs, axis=0)
print(res)
```
If we now want to define a rule to get the eigendecomposition of a diagonal operator we
would write:
```python
@dispatch
def eig(A: Diagonal, eig_slice=slice(0, None, None), **kwargs):
    xnp = A.ops
    eig_vecs = I_like(A).to_dense()
    sorted_ind = xnp.argsort(A.diag)
    eig_vals = A.diag[sorted_ind]
    eig_vecs = eig_vecs[:, sorted_ind]
    return eig_vals[eig_slice], eig_vecs[:, eig_slice]

```
and continuing the previous example we would have
```python
from cola.linalg.eigs import eig
eigvals, eigvecs = eig(D)
print(eigvals)
```

2. **Solving a symmetric linear system** using CG and Nystr&ouml;m preconditioning using
   PyTorch
```python
import cola.torch_fns as xnp
from cola.operators import Symmetric
from cola.linear_algebra import lazify
from cola.linalg.inverse import inverse
from cola.algorithms.preconditioners import NystromPrecond

N, B = 10, 3
L = lazify(xnp.randn(N, N, dtype=xnp.float32))
A = Symmetric(L.T @ L)
P = NystromPrecond(A, rank=A.shape[0] // 2)
rhs = xnp.randn(N, B, dtype=xnp.float32)
A_inv = inverse(A, method='cg', P=P)
soln = A_inv @ rhs
res = xnp.norm(A @ soln - rhs, axis=0)
print(res)
```
To change the backend to JAX simply modify the first import to `import cola.jax_fns as xnp`.

3. **Take the gradient of a linear solve**.
A linear operator can be conceived as a structured container of some parameters.
In CoLA, we have incorporated memory-efficient and fast routines to backpropagate through
some algebraic operations such as a solve:
```python
import cola.torch_fns as xnp
from cola.operators import Diagonal
from cola.linalg.inverse import inverse

dtype = xnp.float32
diag = xnp.Parameter(xnp.array([3., 4., 5.], dtype=dtype))
D = Diagonal(diag)
rhs = xnp.randn(diag.shape[0], 1, dtype=dtype)
soln = inverse(D, method="cg") @ rhs
loss = xnp.norm(soln)
loss.backward()
print(diag.grad)
```

## Use cases and examples
See our examples and tutorials on how to use CoLA for different problems.

## API reference

## Contributing
See the contributing guidelines `CONTRIBUTING.md` for information on submitting issues
and pull requests.

<!--
## Team
-->

## Acknowledgements
This work is supported by XXX.

## Licence
CoLA is XXX licensed.

## Support and contact
Please raise an issue if you find a bug or inadequate performance when using CoLA.
