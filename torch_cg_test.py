import time

import torch

from cola.linalg.inverse.torch_cg import cg, run_batched_cg

max_iters = 100
tol = 1e-15
# dtype = torch.float32
dtype = torch.float64
repeat_n = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
# N, M = 20_000, 5
N, M = 100, 5
# N, M = 10, 5
A = torch.randn(N, dtype=dtype, device=device)
eps = 1e-6
A = A @ A.T + eps * torch.eye(A.shape[0], dtype=dtype, device=device)
rhs = torch.ones(N, M, dtype=dtype, device=device)
soln = torch.linalg.solve(A, rhs)
print(f"Size({N},{N})")
tic = time.time()
# L = torch.linalg.cholesky(A)
L, _ = torch.linalg.cholesky_ex(A)
if torch.cuda.is_available():
    torch.cuda.synchronize()
diff = torch.linalg.norm(A @ soln - rhs)
print(f"{time.time() - tic:1.5e} sec")
print(f"{diff=:1.5e}")

x0 = torch.zeros_like(rhs)
P = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)

tic = time.time()
approx, *_ = run_batched_cg(A, rhs, x0, max_iters, tol, preconditioner=P)
diff = torch.linalg.norm(A @ approx - rhs)
print(f"\n{time.time() - tic:1.5e} sec")
print(f"{diff=:1.5e}")
soln_diff = torch.linalg.norm(approx - soln)
print(f"{soln_diff=:1.5e}")

for _ in range(repeat_n):
    tic = time.time()
    approx, _, cg_iters = cg(A, rhs, x0, P, tol, max_iters)
    diff = torch.linalg.norm(A @ approx - rhs)
    print(f"{diff=:1.5e} | CG iters: {cg_iters}")
    print(f"\n{time.time() - tic:1.5e} sec")
    soln_diff = torch.linalg.norm(approx - soln)
    print(f"{soln_diff=:1.5e}")
