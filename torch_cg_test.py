import numpy as np
import torch

from cola.linalg.inverse.torch_cg import cg, run_batched_cg
from cola.utils.utils_for_tests import generate_pd_from_diag, generate_spectrum

dtype = torch.float32
# dtype = torch.float64
device = "cpu"
N, M = 25, 5
diag = generate_spectrum(coeff=0.75, scale=1.0, size=N, dtype=np.float32)
A = torch.tensor(generate_pd_from_diag(diag, dtype=diag.dtype), dtype=dtype, device=device)
rhs = torch.ones(N, M, dtype=dtype, device=device)
soln = torch.linalg.solve(A, rhs)
diff = torch.linalg.norm(A @ soln - rhs)
print(f"{diff:1.5e}")

x0 = torch.zeros_like(rhs)
max_iters = A.shape[0] + 10
tol = 1e-15
preconditioner = torch.eye(A.shape[0], dtype=dtype)

approx, *_ = run_batched_cg(A, rhs, x0, max_iters, tol, preconditioner=preconditioner)
diff = torch.linalg.norm(A @ approx - rhs)
print(f"{diff:1.5e}")
diff = torch.linalg.norm(approx - soln)
print(f"{diff:1.5e}")

approx, *_ = cg(A, rhs, x0, preconditioner, tol, max_iters)
diff = torch.linalg.norm(A @ approx - rhs)
print(f"{diff:1.5e}")
diff = torch.linalg.norm(approx - soln)
print(f"{diff:1.5e}")
