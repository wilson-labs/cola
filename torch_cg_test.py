import time

import numpy as np
import torch

from cola.linalg.inverse.torch_cg import cg, run_batched_cg
from cola.utils.utils_for_tests import generate_pd_from_diag, generate_spectrum

dtype = torch.float32
repeat_n = 3
# dtype = torch.float64
device = "cpu"
N, M = 25_000, 5
diag = generate_spectrum(coeff=0.75, scale=1.0, size=N, dtype=np.float32)
A = torch.tensor(generate_pd_from_diag(diag, dtype=diag.dtype), dtype=dtype, device=device)
rhs = torch.ones(N, M, dtype=dtype, device=device)
tic = time.time()
soln = torch.linalg.solve(A, rhs)
print(f"{time.time() - tic:1.5e} sec")
diff = torch.linalg.norm(A @ soln - rhs)
print(f"{diff=:1.5e}")

x0 = torch.zeros_like(rhs)
# max_iters = A.shape[0] + 10
max_iters = 100
tol = 1e-15
preconditioner = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)

tic = time.time()
approx, *_ = run_batched_cg(A, rhs, x0, max_iters, tol, preconditioner=preconditioner)
print(f"\n{time.time() - tic:1.5e} sec")
diff = torch.linalg.norm(A @ approx - rhs)
print(f"{diff=:1.5e}")
soln_diff = torch.linalg.norm(approx - soln)
print(f"{soln_diff=:1.5e}")

for _ in range(repeat_n):
    tic = time.time()
    approx, *_ = cg(A, rhs, x0, preconditioner, tol, max_iters)
    print(f"\n{time.time() - tic:1.5e} sec")
    diff = torch.linalg.norm(A @ approx - rhs)
    print(f"{diff=:1.5e}")
    soln_diff = torch.linalg.norm(approx - soln)
    print(f"{soln_diff=:1.5e}")
