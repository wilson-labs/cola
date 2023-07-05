import time
import numpy as np
from cola import jax_fns
from cola.basic_operations import lazify
from cola.basic_operations import kron
from cola.linalg.inverse import inverse
from cola.ops import SelfAdjoint
from cola.algorithms.cg import solve_cg
from cola.utils_test import generate_spectrum
from cola.utils_test import generate_clustered_spectrum
from cola.utils_test import generate_diagonals
from cola.experiment_utils import config_and_get_dtype_case
from cola.experiment_utils import print_time_taken
from cola.experiment_utils import save_object

save_output = True
xnp = jax_fns
# case = "single_gpu"
case = "single_cpu"
# case = "double_cpu"
dtype = config_and_get_dtype_case(case, xnp)
output_path = f"./logs/kronecker_solve_{case}.pkl"
diag1 = generate_spectrum(coeff=0.50, scale=1.0, size=50, dtype=np.float32)
clusters = [0.1, 0.3, 0.85, 0.99]
sizes = [8, 5, 3, 3]
A1 = xnp.array(generate_diagonals(diag1, seed=48), dtype=dtype)
A1 = SelfAdjoint(lazify(A1))
diag2 = generate_clustered_spectrum(clusters, sizes, std=0.025, seed=48)
A2 = xnp.array(generate_diagonals(diag2, seed=21), dtype=dtype)
A2 = SelfAdjoint(lazify(A2))
clusters = [0.1, 0.11, 0.12, 0.13, 0.14, 0.2, 0.25, 0.28, 0.3, 0.4, 0.5, 0.99]
diag3 = generate_clustered_spectrum(clusters, [1 for _ in range(len(clusters))], std=0.0, seed=48)
A3 = xnp.array(generate_diagonals(diag3, seed=123), dtype=dtype)
A3 = SelfAdjoint(lazify(A3))

results = {}
K = kron(A1, kron(A2, A3))
print(f"Problem size: {K.shape[0]:,d}")
rhs = xnp.array(np.random.normal(size=(K.shape[0], )), dtype=dtype)

tic = time.time()
# repeat = 0
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
    iterations[idx] = sum([M.info["iterations"] for M in Kinv.Ms])
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
results["iterative"] = {"res": res, "times": times, "iterations": iterations,
                        "residuals": info["residuals"]}
print(f"\nTimes {np.mean(times[1:]):1.5e} sec")
print(f"Res   {np.mean(res):1.5e}")
toc = time.time()
print_time_taken(toc - tic)
if save_output:
    save_object(results, filepath=output_path)
