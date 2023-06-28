import time
from functools import partial
from jax import jit, vmap
from jax.config import config
import linops.jax_fns as xnp
from linops.operator_base import LinearOperator
from linops.operators import Symmetric
from linops.experiment_utils import generate_biposson_data, construct_laplacian
from linops.experiment_utils import get_dense_times, get_times
from linops.experiment_utils import print_time_taken, save_object

config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)
save_output = True
Ns = [1_000, 500, 250, 150, 50]
# Ns = [10, 15, 25, 50]
ndims, repeat, dense_N_max, dtype, results = 2, 3, 200, xnp.float64, {}
output_path = "./logs/bipoisson_solve.pkl"
tic = time.time()
for N in Ns:
    rhs, dx, N = generate_biposson_data(N, ndims)
    results[N] = {}  # N gets altered fron generate_biposson_data, needs to be below
    laplacian = jit(vmap(partial(construct_laplacian, N=N, dx=dx, ndims=ndims), -1, -1))
    shape = (N**ndims, N**ndims)
    L = Symmetric(LinearOperator(dtype, shape=shape, matmat=laplacian))
    LL = L @ L
    Lit = Symmetric(LinearOperator(dtype, shape=shape, matmat=lambda x: L @ (L @ x)))
    print(f"\nActual dataset size {N:,d} | linear system size: {N ** ndims:,d}")

    if N < dense_N_max:
        results = get_dense_times(LL, rhs, results, N, xnp, repeat)

    it_kwargs = {"method": "cg", "info": True, "tol": 1e-11, "max_iters": 10_000}
    results = get_times(Lit, rhs, it_kwargs, results, N, xnp, repeat, key="iterative")

    lin_kwargs = {"method": "auto", "info": True, "tol": 1e-11, "max_iters": 10_000}
    results = get_times(LL, rhs, lin_kwargs, results, N, xnp, repeat, key="linops")

toc = time.time()
print_time_taken(toc - tic)
if save_output:
    save_object(results, filepath=output_path)
