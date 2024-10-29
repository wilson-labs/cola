<p align="center">
 <img src="https://user-images.githubusercontent.com/6753639/251633368-1ec42732-1759-45d7-b949-51df6429a90a.svg"  width="300" height="150">
</p>

<!--
<p align="center">
  <img src="https://github.com/wilson-labs/cola/assets/6753639/28630ef8-5dcb-41c2-9f36-3cbba52f3d88.svg" width="300" height="139.29">
</p> -->
<!--
<p align = "center">
  <img src="https://github.com/wilson-labs/cola/assets/6753639/8b02c51e-0e1e-44f5-a52a-47ad428688e4.svg" width="300" height="139.29">
</p>-->


# Compositional Linear Algebra (CoLA)

[![Documentation](https://readthedocs.org/projects/cola/badge/)](https://cola.readthedocs.io/en/stable/)
[![tests](https://github.com/wilson-labs/cola/actions/workflows/python-package.yml/badge.svg)](https://github.com/wilson-labs/cola/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/wilson-labs/cola/branch/main/graph/badge.svg?token=bBnkfHv30C)](https://codecov.io/gh/wilson-labs/cola)
[![PyPI version](https://img.shields.io/pypi/v/cola-ml)](https://pypi.org/project/cola-ml/)
[![Paper](https://img.shields.io/badge/arXiv-2309.03060-red)](https://arxiv.org/abs/2309.03060)
[![Downloads](https://static.pepy.tech/badge/cola-ml)](https://pepy.tech/project/cola-ml)
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wilson-labs/cola/blob/master/docs/notebooks/colabs/all.ipynb) -->

CoLA is a framework for scalable linear algebra, providing: 

(1) Fast hardware-sensitive (GPU accelerated) iterative algorithms for general matrix operations; <br>
(2) Algorithms that can exploit matrix structure for efficiency; <br>
(3) A mechanism to rapidly prototype different matrix structures and compositions of structures.

CoLA natively supports PyTorch, JAX, as well as (limited) NumPy if JAX is not installed.

## Installation
```shell
pip install cola-ml
```

## Features in CoLA
* Large scale linear algebra routines for `solve(A,b)`, `eig(A)`, `logdet(A)`, `exp(A)`, `trace(A)`, `diag(A)`, `sqrt(A)`.
* Provides (user extendible) compositional rules to exploit structure through multiple dispatch.
* Has memory-efficient autodiff rules for iterative algorithms.
* Works with PyTorch or JAX, supporting GPU hardware acceleration.
* Supports operators with complex numbers and low precision.
* Provides linear algebra operations for both symmetric and non-symmetric matrices.

See https://cola.readthedocs.io/en/latest/ for our full documentation and many examples.


## Quick start guide
1. **LinearOperators**. The core object in CoLA is the LinearOperator. You can add and subtract them `+, -`,
multiply by constants `*, /`, matrix multiply them `@` and combine them in other ways:
`kron, kronsum, block_diag` etc.
```python
import jax.numpy as jnp
import cola

A = cola.ops.Diagonal(jnp.arange(5) + .1)
B = cola.ops.Dense(jnp.array([[2., 1.], [-2., 1.1], [.01, .2]]))
C = B.T @ B
D = C + 0.01 * cola.ops.I_like(C)
E = cola.ops.Kronecker(A, cola.ops.Dense(jnp.ones((2, 2))))
F = cola.ops.BlockDiag(E, D)

v = jnp.ones(F.shape[-1])
print(F @ v)
```
```
[0.2       0.2       2.2       2.2       4.2       4.2       6.2
 6.2       8.2       8.2       7.8       2.1    ]
```

2. **Performing Linear Algebra**. With these objects we can perform linear algebra operations even when they are very big.
```python
print(cola.linalg.trace(F))
Q = F.T @ F + 1e-3 * cola.ops.I_like(F)
b = cola.linalg.inv(Q) @ v
print(jnp.linalg.norm(Q @ b - v))
print(cola.linalg.eig(F, k=F.shape[0])[0][:5])
print(cola.linalg.sqrt(A))
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
Qs = cola.SelfAdjoint(Q)
%timeit cola.linalg.inv(Q) @ v
%timeit cola.linalg.inv(Qs) @ v
```

3. **JAX and PyTorch**. We support both ML frameworks.
```python
import torch
A = cola.ops.Dense(torch.Tensor([[1., 2.], [3., 4.]]))
print(cola.linalg.trace(cola.kron(A, A)))

import jax.numpy as jnp
A = cola.ops.Dense(jnp.array([[1., 2.], [3., 4.]]))
print(cola.linalg.trace(cola.kron(A, A)))
```

```
tensor(25.)
25.0
```

CoLA also supports autograd (and jit):
```python
from jax import grad, jit, vmap


def myloss(x):
    A = cola.ops.Dense(jnp.array([[1., 2.], [3., x]]))
    return jnp.ones(2) @ cola.linalg.inv(A) @ jnp.ones(2)


g = jit(vmap(grad(myloss)))(jnp.array([.5, 10.]))
print(g)
```

```
[-0.06611571 -0.12499995]
```

## Citing us
If you use CoLA, please cite the following paper:

[Andres Potapczynski, Marc Finzi, Geoff Pleiss, and Andrew Gordon Wilson. "CoLA: Exploiting Compositional Structure for Automatic and Efficient Numerical Linear Algebra." 2023.](https://arxiv.org/abs/2309.03060)
```
@article{potapczynski2023cola,
  title={{CoLA: Exploiting Compositional Structure for Automatic and Efficient Numerical Linear Algebra}},
  author={Andres Potapczynski and Marc Finzi and Geoff Pleiss and Andrew Gordon Wilson},
  journal={arXiv preprint arXiv:2309.03060},
  year={2023}
}
```

### Features implemented

| Linear Algebra    | inverse | eig | diag | trace | logdet | exp | sqrt | f(A) | SVD | pseudoinverse |
|:-----------------:|:-------:|:---:|:----:|:-----:|:------:|:---:|:----:|:--------:|:---:|:-------------:|
| **Implementation**|    ✓    |  ✓  |   ✓  |   ✓  |    ✓   |  ✓  |   ✓  |    ✓     |   ✓ |       ✓       |

| LinearOperators   | Diag | BlockDiag | Kronecker | KronSum | Sparse | Jacobian | Hessian | Fisher | Concatenated | Triangular | FFT | Tridiagonal |
|:-----------------:|:----:|:---------:|:---------:|:-------:|:------:|:--------:|:-------:|:------:|:------------:|:----------:|:---:|:-----------:|
| **Implementation**|   ✓  |     ✓     |     ✓     |    ✓    |   ✓  |    ✓     |    ✓    |   ✓    |      ✓       |     ✓      |   ✓  |      ✓      |

| Annotations      | SelfAdjoint | PSD | Unitary |
|:----------------:|:-----------:|:---:|:-------:|
| **Implementation**|      ✓      |  ✓  |    ✓   |


| Backends      | PyTorch | JAX | NumPy |
|:----------------:|:-----------:|:---:|:-------:|
| **Implementation**|      ✓      |  ✓  |Most operations|

## Contributing
See the contributing guidelines [docs/CONTRIBUTING.md](https://cola.readthedocs.io/en/latest/contributing.html) for information on submitting issues
and pull requests.

CoLA is Apache 2.0 licensed.

## Support and contact
Please raise an issue if you find a bug or slow performance when using CoLA.
