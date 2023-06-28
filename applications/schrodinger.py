import time
import numpy as np
from scipy.sparse.linalg import LinearOperator as LOS
from linops.experiment_utils import print_time_taken, save_object
from linops.experiment_utils import construct_schrodinger_setting
from linops.experiment_utils import get_times_schrodinger
from linops.experiment_utils import get_times_schrodinger_jax

save_output = True
Ns = [5_000, 1_000, 500, 100]
ndims = 1
repeat, results, k = 2, {}, 20
output_path = "./logs/schrodinger.pkl"

tic = time.time()
for N in Ns:
    H = construct_schrodinger_setting(N, ndims)
    results[N] = {}
    HOS = LOS(H.shape, matvec=lambda X: H @ X[:, None], matmat=lambda X: H @ X, dtype=np.float32)
    get_times_schrodinger_jax(HOS, (k, ndims), results[N], repeat)
    get_times_schrodinger(H, (k, ndims), results[N], repeat)

toc = time.time()
print_time_taken(toc - tic)
if save_output:
    save_object(results, filepath=output_path)
