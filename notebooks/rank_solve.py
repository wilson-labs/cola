import time
import numpy as np
from cola import torch_fns
from cola.basic_operations import lazify
from cola.algorithms.preconditioners import AdaNysPrecond
from cola.algorithms.preconditioners import NystromPrecond
from cola.ops import Identity
from cola.algorithms.cg import run_batched_cg
from tests.utils import generate_spectrum, generate_pd_from_diag

xnp = torch_fns
dtype = xnp.float32
size = int(2 ** 10)
# coeffs = [0.05, 0.1, 0.15, 0.15, 0.05, 0.5]
coeffs = [0.005 for _ in range(10)]
rank_init, mult = 4, 1.5
case = 2
bounds = (0.2, 0.4, 0.6)
# bounds = (0.05, 0.1, 0.2)
# bounds = (0.2, 0.7, 1.0)
rank = rank_init
tic = time.time()
cg_iters = np.zeros(shape=len(coeffs))
for idx, coeff in enumerate(coeffs):
    print("\n" + "*" * 50)
    print(f"Initial rank {rank:2d}")
    print(f"Coeff: {coeff:1.2f}")
    diag = generate_spectrum(coeff=coeff, scale=1.0, size=size)
    A = lazify(xnp.array(generate_pd_from_diag(diag, dtype=diag.dtype), dtype=dtype))
    if case == 1:
        Nys = NystromPrecond(A=A, rank=100, adjust_mu=False)
    elif case == 2:
        Nys = AdaNysPrecond(A=A, rank=rank, bounds=bounds, mult=mult, adjust_mu=False)
        rank = Nys.rank
        print(f"Selected rank {rank:2d}")
        print(f"Error: {Nys.error:1.5e}")
    else:
        Nys = Identity(A.dtype, A.shape)
    b = xnp.ones(shape=(A.shape[0], 1), dtype=dtype)
    x0 = xnp.zeros_like(b)
    out = run_batched_cg(A, b, x0, max_iters=1000, tol=1e-11, preconditioner=Nys, pbar=False)
    cg_iters[idx] = out[-1]['iterations']
    print(f"CG iters: {int(cg_iters[idx]):3d}")
toc = time.time()
print(f"\nTime taken: {toc - tic:2.5f} sec")
print(f"CG Q75 {np.quantile(cg_iters, q=0.75):2.2f}")
print(f"CG Q50 {np.quantile(cg_iters, q=0.50):2.2f}")
print(f"CG Q25 {np.quantile(cg_iters, q=0.25):2.2f}")
