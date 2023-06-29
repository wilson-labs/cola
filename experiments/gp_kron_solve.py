import time
import torch
from gpytorch.kernels import RBFKernel
from gpytorch.kernels import MultitaskKernel
from cola import torch_fns as xnp
from cola.operators import SelfAdjoint
from cola.experiment_utils import get_dense_times, get_times, get_times_cg2
from cola.experiment_utils import print_time_taken, save_object

save_output = True
output_path = "./logs/gp_kron_solve.pkl"
Ns = [1_000, 500, 100, 50, 25, 10, 5]
D, num_tasks, repeat, results, dtype, dense_N_max = 33, 11, 3, {}, xnp.float32, 100
tic = time.time()
ls = xnp.array([[0.1]], dtype=dtype)
for N in Ns:
    results[N] = {}
    train_x = torch.randn(N, D)

    rbf = RBFKernel()
    rbf.lengthscale = ls
    covar_module = MultitaskKernel(RBFKernel(), num_tasks=num_tasks, rank=1)
    K = covar_module(train_x)
    KA = SelfAdjoint(covar_module(train_x))
    rhs = torch.randn(K.shape[-1], 3)

    if N < dense_N_max:
        results = get_dense_times(K, rhs, results, N, xnp, repeat)

    # it_kwargs = {"method": "cg", "info": True, "tol": 1e-6, "max_iters": 1_000}
    # results = get_times(K, rhs, it_kwargs, results, N, xnp, repeat, key="iterative")
    it_kwargs = {"tol": 1e-6, "max_iters": 1_000}
    results[N] = get_times_cg2(KA, rhs, it_kwargs, results[N], xnp, repeat)

    lin_kwargs = {"method": "auto", "info": True, "tol": 1e-6, "max_iters": 1_000}
    results = get_times(K, rhs, lin_kwargs, results, N, xnp, repeat, key="cola")

toc = time.time()
print_time_taken(toc - tic)
if save_output:
    save_object(results, filepath=output_path)
