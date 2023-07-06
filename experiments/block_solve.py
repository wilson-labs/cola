import time
import numpy as np
from cola import jax_fns
from cola.basic_operations import lazify
from cola.linalg.inverse import inverse
from cola.ops import SelfAdjoint
from cola.ops import Diagonal
from cola.basic_operations import block_diag
from cola.algorithms.cg import cg
from cola.utils_test import generate_spectrum
from cola.utils_test import generate_beta_spectrum
from cola.utils_test import generate_clustered_spectrum
from cola.utils_test import generate_diagonals
from cola.experiment_utils import config_and_get_dtype_case
from cola.experiment_utils import print_time_taken
from cola.experiment_utils import save_object

save_output = True
xnp = jax_fns
# case = "single_gpu"
# case = "single_cpu"
case = "double_cpu"
dtype = config_and_get_dtype_case(case, xnp)
output_path = f"./logs/block_solve_{case}.pkl"
N1, N2, N3 = 1_000, 20_000, 1_000
diag1 = generate_spectrum(coeff=0.50, scale=1.0, size=N1, dtype=np.float32)
A1 = SelfAdjoint(lazify(xnp.array(generate_diagonals(diag1, seed=48), dtype=dtype)))
diag2 = generate_beta_spectrum(scale=1_000, alpha=0.5, beta=0.1, size=N2, coeff=0.5)
A2 = Diagonal(xnp.array(diag2, dtype=dtype))
clusters = [0.1, 0.3, 0.85, 0.99]
sizes = [N3 // 4 for _ in range(len(clusters))]
diag3 = generate_clustered_spectrum(clusters, sizes, std=0.025, seed=48)
A3 = SelfAdjoint(lazify(xnp.array(generate_diagonals(diag3, seed=21), dtype=dtype)))

results = {}
K = block_diag(A1, A2, A3)
print(f"Problem size: {K.shape[0]:,d}")
rhs = xnp.array(np.random.normal(size=(K.shape[0], )), dtype=dtype)
results["info"] = {"size": K.shape[0]}

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
Kinv = inverse(K, method="cg", info=True, tol=1e-8)
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
    soln, info = cg(K, rhs)
    t1 = time.time()
    times[idx] = t1 - t0
    res[idx] = xnp.norm(K @ soln - rhs)
    # iterations[idx] = info["iterations"]
results["iterative"] = {"res": res, "times": times, "iterations": iterations,
                        "residuals": info["residuals"]}
print(f"\nTimes {np.mean(times[1:]):1.5e} sec")
print(f"Res   {np.mean(res):1.5e}")
toc = time.time()
print_time_taken(toc - tic)
if save_output:
    save_object(results, filepath=output_path)
