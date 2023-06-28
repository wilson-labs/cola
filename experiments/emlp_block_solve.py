import jax
import time
from linops import jax_fns as xnp
from linops.experiment_utils import construct_emlp_ops
from linops.experiment_utils import get_times_emlp
from linops.experiment_utils import print_time_taken, save_object

jax.config.update('jax_platform_name', 'cpu')

save_output = True
output_path = "./logs/emlp_block_solve.pkl"
# Ns = [16_384, 8192, 4096, 2048, 1024, 512]
Ns = [15_000, 9_000, 5_000, 2_500, 1_000, 500]
dense_N_max = 11_000
results, dtype, repeat = {}, xnp.float32, 1

tic = time.time()
for N in Ns:
    results[N] = {}
    rep, Rdense, Riterative = construct_emlp_ops(N)

    if N < dense_N_max:
        get_times_emlp(Rdense, results[N], repeat, key="dense")

    get_times_emlp(Riterative, results[N], repeat, key="iterative")
    get_times_emlp(rep, results[N], repeat, key="linops")

toc = time.time()
print_time_taken(toc - tic)
if save_output:
    save_object(results, filepath=output_path)
