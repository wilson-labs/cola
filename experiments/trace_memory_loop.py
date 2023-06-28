import time
import tracemalloc
import numpy as np
from cola.operators import I_like
# from cola.operators import Dense
from cola.operators import Diagonal
import cola.jax_fns as xnp
from cola.experiment_utils import print_time_taken, save_object
# from cola.utils_test import generate_spectrum, generate_lower_from_diag
from cola.utils_test import generate_spectrum
from experiments.trace_fns import trace_cg
from jax.config import config

config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)

save_output = True
output_path = "./logs/trace_loop.pkl"
results, dtype, seed, repeat = {}, xnp.float64, 98, 3
Ns = [10_000, 5_000, 1_000, 500, 100, 50]
# Ns = [2_000, 1_000, 500, 100, 50]
# Ns = [100, 50, 10, 5]


def compute_cg_loss(diag):
    N, dtype = diag.shape[0], diag.dtype
    rhs = xnp.randn(N, 1, dtype=dtype)
    A = Diagonal(diag)
    # A = Dense(diag)
    x0 = xnp.zeros_like(rhs)
    max_iters, tol = N, 1e-20
    preconditioner = I_like(A)
    soln, *_ = trace_cg(A, rhs, x0, max_iters, tol, preconditioner)
    # soln, *_, it, _ = trace_cg(A, rhs, x0, max_iters, tol, preconditioner)
    out = xnp.norm(soln)
    return out


tic = time.time()
for N in Ns:
    print(f"Size: {N:,d} | Seed: {seed:d}")
    peaks, times = np.zeros(repeat), np.zeros(repeat)
    # diag = xnp.array(np.random.randint(low=10, high=20, size=N), dtype=dtype)
    # diag = xnp.array(np.random.normal(size=(N, N)), dtype=dtype)
    diag = generate_spectrum(0.5, 1., N, dtype=np.float64)
    # diag = generate_lower_from_diag(diag ** 0.5, dtype=np.float64, seed=seed, orthogonalize=False)
    # diag = diag @ diag.T
    diag = xnp.array(diag, dtype=dtype)
    for idx in range(repeat):
        tracemalloc.start()

        t0 = time.time()
        fn = xnp.grad(compute_cg_loss)
        out = fn(diag)
        out.block_until_ready()
        print(f"out: {xnp.sum(out)}")
        t1 = time.time()

        current, peak = tracemalloc.get_traced_memory()
        print(f"Current: {current / 10**6} MB | Peak: {peak / 10**6} MB | Time: {t1 -t0:.4f}")
        tracemalloc.stop()
        times[idx] = t1 - t0
        peaks[idx] = peak
    del diag
    results[N] = {"memory": peaks, "time": times}

toc = time.time()
print_time_taken(toc - tic)
if save_output:
    save_object(results, filepath=output_path)
