import numpy as np
import torch

from cola.linalg.tbd.logdet_quad import logdet_quad, logdet_quad_accurate
from cola.utils.utils_for_tests import generate_pd_from_diag, generate_spectrum

dtype = torch.float64
device = "cpu"
N, M = 13, 1
diag = generate_spectrum(coeff=0.75, scale=1.0, size=N, dtype=np.float32)
A = torch.tensor(generate_pd_from_diag(diag, dtype=diag.dtype), dtype=dtype, device=device)
rhs = torch.ones(N, M, dtype=dtype, device=device)

A = A.clone().detach().requires_grad_(True)
out, *_ = logdet_quad(A, rhs)
out.backward()
print(out)
print(A.grad.shape)

A = A.clone().detach().requires_grad_(True)
out, *_ = logdet_quad_accurate(A, rhs)
out.backward()
print(out)
print(A.grad.shape)
