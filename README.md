<p align="center">
 <img src="https://user-images.githubusercontent.com/6753639/251633368-1ec42732-1759-45d7-b949-51df6429a90a.svg"  width="300" height="150">
</p>


# Compositional Linear Algebra (CoLA)

[![Documentation](https://readthedocs.org/projects/cola/badge/)](https://cola.readthedocs.io/en/latest/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wilson-labs/cola/blob/master/docs/notebooks/colabs/all.ipynb)
[![tests](https://github.com/wilson-labs/cola/actions/workflows/python-package.yml/badge.svg)](https://github.com/wilson-labs/cola/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/wilson-labs/cola/branch/main/graph/badge.svg?token=bBnkfHv30C)](https://codecov.io/gh/wilson-labs/cola)

CoLA is a framework for scalable linear algebra, automatically exploiting the structure often found in machine learning problems and beyond. 
CoLA supports both PyTorch and JAX.

## Installation
```shell
pip install git+https://github.com/wilson-labs/cola.git
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
- [x] diag
- [x] trace
- [ ] exp
- [ ] logdet
- [ ] $f(A)$
- [ ] SVD
- [ ] pseudoinverse
      
Linear ops
- [x] Diag
- [x] BlockDiag
- [x] Kronecker
- [x] KronSum
- [x] Sparse
- [x] Jacobian
- [x] Hessian
- [ ] Fisher
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
1. **LinearOperators** The core object in CoLA is the LinearOperator. You can add and subtract them `+, -`,
multiply by constants `*, /`, matrix multiply them `@` and combine them in other ways:
`kron, kronsum, block_diag` etc.
```python
import jax.numpy as jnp
from cola import ops
import jax.numpy as jnp
from cola import ops
A = ops.Diagonal(jnp.arange(5)+.1)
B = ops.Dense(jnp.array([[2.,1.,],[-2.,1.1],[.01,.2]]))
C = B.T@B
D = C+0.01*ops.I_like(C)
E = ops.Kronecker(A,ops.Dense(jnp.ones((2,2))))
F = ops.BlockDiag(E,D)

v = jnp.ones(F.shape[-1])
print(F@v)
```
```
[0.2       0.2       2.2       2.2       4.2       4.2       6.2
 6.2       8.2       8.2       7.8121004 2.062    ]
```

2. **Performing Linear Algebra** With these objects we can perform linear algebra operations even when they are very big.
```python
print(cola.linalg.trace(F))
Q = F.T@F+1e-3*I_like(F)
b = cola.linalg.inverse(Q)@v
print(jnp.linalg.norm(Q@b-v))
print(cola.linalg.eig(F)[0][:5])
print(cola.sqrt(A))
#print(cola.logdet(D))
```

```
31.2701
0.0010193728
[ 2.0000000e-01+0.j  0.0000000e+00+0.j  2.1999998e+00+0.j
 -1.1920929e-07+0.j  4.1999998e+00+0.j]
diag([0.31622776 1.0488088  1.4491377  1.7606816  2.0248456 ])
```

For many of these functions, if we know additional information about the matrices we can annotate them
to enable the algorithms to run faster.

```python
Qs = ops.Symmetric(Q)
%timeit cola.linalg.inverse(Q)@v
%timeit cola.linalg.inverse(Qs)@v
```

3. **Jax and Pytorch** You can freely use jax or pytorch with Cola:
```python
import torch

A = ops.Dense(torch.Tensor([[1.,2],[3,4]]))
print(cola.linalg.trace(cola.kron(A,A)))

import jax.numpy as jnp
A = ops.Dense(jnp.array([[1.,2],[3,4]]))
print(cola.linalg.trace(cola.kron(A,A)))
```

```
tensor(25.)
25.0
```

and both support autograd (and jit):
```python
from jax import grad, jit,vmap

def myloss(x):
  A = ops.Dense(jnp.array([[1.,2],[3,x]]))

  return jnp.ones(2)@cola.linalg.inverse(A)@jnp.ones(2)
g = jit(vmap(grad(myloss)))(jnp.array([.5,10.]))
print(g)
```

```
[-0.06611571 -0.12499995]
```

See https://cola.readthedocs.io/en/latest/ for our full documentation and many examples.

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
