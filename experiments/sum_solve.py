import time
import numpy as np
from cola import jax_fns
from cola.linear_algebra import lazify
from cola.linalg.inverse import inverse
from cola.ops import SelfAdjoint
from cola.algorithms.cg import solve_cg
from cola.utils_test import generate_spectrum
from cola.utils_test import generate_diagonals
from cola.experiment_utils import print_time_taken
from cola.experiment_utils import save_object

save_output = False
output_path = "./logs/sum_solve.pkl"
xnp = jax_fns
dtype = xnp.float32
# N = 5_000
N = 100
sum_size = 100
coeffs = [0.05, 0.25, 0.5, 1.0]
np.random.seed(seed=48)
diag1 = generate_spectrum(coeff=0.5, scale=1.0, size=N, dtype=np.float32)
K = SelfAdjoint(lazify(xnp.array(generate_diagonals(diag1), dtype=dtype)))
for idx in range(sum_size - 1):
    coeff = coeffs[int(np.random.randint(0, len(coeffs), size=1))]
    diag = generate_spectrum(coeff=coeff, scale=1.0, size=N, dtype=np.float32)
    Ai = SelfAdjoint(lazify(xnp.array(generate_diagonals(diag), dtype=dtype)))
    K += Ai

K = SelfAdjoint(K)
results = {}
print(f"Problem size: {K.shape[0]:,d}")
rhs = xnp.array(np.random.normal(size=(K.shape[0], )), dtype=dtype)
results["info"] = {"size": K.shape[0], "sum_size": sum_size}

tic = time.time()
repeat = 5
times = np.zeros(shape=(repeat, ))
res = np.zeros(shape=(repeat, ))
K_dense = K.to_dense()
for idx in range(repeat):
    t0 = time.time()
    soln = xnp.solve(K_dense, rhs)
    t1 = time.time()
    times[idx] = t1 - t0
    res[idx] = xnp.norm(K_dense @ soln - rhs)
results["dense"] = {"res": res, "times": times}
print(f"\nTimes {np.mean(times[1:]):1.5e} sec")
print(f"Res   {np.mean(res):1.5e}")

repeat = 5
times = np.zeros(shape=(repeat, ))
res = np.zeros(shape=(repeat, ))
iterations = np.zeros(shape=(repeat, ))
Kinv = inverse(K, method="svrg", info=True, tol=1e-8)
for idx in range(repeat):
    t0 = time.time()
    soln = Kinv @ rhs
    t1 = time.time()
    times[idx] = t1 - t0
    # iterations[idx] = sum([M.info["iterations"] for M in Kinv.Ms])
    res[idx] = xnp.norm(K @ soln - rhs)
results["dispatch"] = {"res": res, "times": times, "iterations": iterations}
print(f"\nTimes {np.mean(times[1:]):1.5e} sec")
print(f"Res   {np.mean(res):1.5e}")

repeat = 5
times = np.zeros(shape=(repeat, ))
res = np.zeros(shape=(repeat, ))
iterations = np.zeros(shape=(repeat, ))
for idx in range(repeat):
    t0 = time.time()
    soln, info = solve_cg(K, rhs)
    t1 = time.time()
    times[idx] = t1 - t0
    res[idx] = xnp.norm(K @ soln - rhs)
    iterations[idx] = info["iterations"]
results["iterative"] = {
    "res": res,
    "times": times,
    "iterations": iterations,
    "residuals": info["residuals"]
}
print(f"\nTimes {np.mean(times[1:]):1.5e} sec")
print(f"Res   {np.mean(res):1.5e}")
toc = time.time()
print_time_taken(toc - tic)
if save_output:
    save_object(results, filepath=output_path)
