import time

import torch

from cola.linalg.inverse.torch_cg import cg, run_batched_cg

# from cola.utils.utils_for_tests import generate_pd_from_diag, generate_spectrum

# dtype = torch.float32
dtype = torch.float64
repeat_n = 5
device = "cuda"
N, M, normalize = 20_000, 5, False
# N, M, normalize = 10, 5, False
# diag = generate_spectrum(coeff=0.75, scale=1.0, size=N, dtype=np.float32)
# A = torch.tensor(generate_pd_from_diag(diag, dtype=diag.dtype, normalize=normalize), dtype=dtype, device=device)

A = torch.randn(N, dtype=dtype, device=device)
eps = 1e-1
A = A @ A.T + eps * torch.eye(A.shape[0], dtype=dtype, device=device)
rhs = torch.ones(N, M, dtype=dtype, device=device)
tic = time.time()
soln = torch.linalg.solve(A, rhs)
print(f"Size({N},{N})")
diff = torch.linalg.norm(A @ soln - rhs)
print(f"{time.time() - tic:1.5e} sec")
print(f"{diff=:1.5e}")

x0 = torch.zeros_like(rhs)
# max_iters = A.shape[0] + 10
max_iters = 10
tol = 1e-15
preconditioner = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)

tic = time.time()
approx, *_ = run_batched_cg(A, rhs, x0, max_iters, tol, preconditioner=preconditioner)
diff = torch.linalg.norm(A @ approx - rhs)
print(f"\n{time.time() - tic:1.5e} sec")
print(f"{diff=:1.5e}")
soln_diff = torch.linalg.norm(approx - soln)
print(f"{soln_diff=:1.5e}")

for _ in range(repeat_n):
    tic = time.time()
    approx, *_ = cg(A, rhs, x0, preconditioner, tol, max_iters)
    diff = torch.linalg.norm(A @ approx - rhs)
    print(f"{diff=:1.5e}")
    print(f"\n{time.time() - tic:1.5e} sec")
    soln_diff = torch.linalg.norm(approx - soln)
    print(f"{soln_diff=:1.5e}")
