import time
import numpy as np
from jax.scipy.sparse.linalg import cg
from cola import jax_fns
from cola.experiment_utils import get_data_class1
from cola.experiment_utils import print_time_taken
from cola.experiment_utils import save_object
from jax.config import config
config.update('jax_platform_name', 'cpu')

save_output = True
case = "scipy"
output_path = f"./logs/timings_{case}.pkl"
xnp = jax_fns
dtype = xnp.float32
tic = time.time()
results = {}
Ns = [100, 900, 10_000, 90_000, 1_000_000]
repeat = 3
times = np.zeros(shape=(len(Ns), repeat))
res = np.zeros(shape=(len(Ns), repeat))
for idx, N in enumerate(Ns):
    K = get_data_class1(N, xnp, dtype)
    rhs = xnp.array(np.random.normal(size=(K.shape[0], )), dtype=dtype)
    for jdx in range(repeat):
        print(f"Problem size: {K.shape[0]:,d}")

        t0 = time.time()
        soln, Z = cg(A=K, b=rhs, tol=1e-8)
        t1 = time.time()
        times[idx, jdx] = t1 - t0
        res[idx, jdx] = xnp.norm(K @ soln - rhs) / xnp.norm(rhs)

results["times"] = times
results["sizes"] = Ns
results["res"] = res
print(results)
print(f"\nTimes {np.mean(times[1:]):1.5e} sec")

toc = time.time()
print_time_taken(toc - tic)
if save_output:
    save_object(results, filepath=output_path)
