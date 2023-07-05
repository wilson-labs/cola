import time
from jax.config import config
from cola import jax_fns as xnp
from cola.gp_fns import construct_rffs
from cola.ops import Symmetric, Dense, I_like
from cola.experiment_utils import load_uci_data, print_time_taken, save_object
from cola.experiment_utils import get_times_cg2, get_times_svrg

config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)
save_output = True
output_path = "./logs/rff_solve.pkl"
train_x, train_y, *_ = load_uci_data(data_dir="/home/ubu/Downloads/", dataset="elevators")
# train_x, train_y, *_ = load_uci_data(data_dir="/home/ubu/Downloads/", dataset="buzz")
num_features, repeat = int(1e3), 3
N = 100
train_x, train_y = train_x[:N], train_y[:N]  # set N=-1 for full dataset
results, dtype = {}, xnp.float64
tic = time.time()
ls = xnp.array([[0.01]], dtype=dtype)
# ls = xnp.exp(xnp.randn(train_x.shape[1], 1, dtype=dtype))
sigma = xnp.array([[0.1]], dtype=dtype)
Z = construct_rffs(train_x, ls, xnp, num_features)
ZT = Dense(Z.T.to_dense())  # to exponse Dense not Transpose for Z.T
RF = Z @ ZT + sigma * I_like(Z @ ZT)
RFS = Symmetric(Z @ ZT + sigma * I_like(Z @ ZT))
rhs = xnp.array(train_y, dtype=dtype)[:, None]

it_kwargs = {"tol": 1e-8, "max_iters": 100}
results = get_times_cg2(RFS, rhs, it_kwargs, results, xnp, repeat)

lin_kwargs = {"tol": 1e-8, "max_iters": 10_000, "bs": 100, "lr_scale": 1e-0, "info": True}
results = get_times_svrg(RF, rhs, lin_kwargs, results, xnp, repeat)

toc = time.time()
print_time_taken(toc - tic)
if save_output:
    save_object(results, filepath=output_path)
