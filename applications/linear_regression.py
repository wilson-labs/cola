import time
import cola.jax_fns as xnp
from cola.linear_algebra import lazify
from cola.operators import SelfAdjoint
from cola.operators import Diagonal
from cola.experiment_utils import load_uci_data, print_time_taken, save_object
from cola.experiment_utils import get_times, get_times_sk_linear
# import jax.config
# jax.config.update("jax_enable_x64", True)

save_output = True
output_path = "./logs/linear_regression.pkl"
dtype, results = xnp.float32, {}
cases = [
    ("protein", [34_000, 30_000, 20_000, 10_000, 5_000, 1_000, 500, 100]),
    ("song", [400_000, 100_000, 50_000, 34_000, 30_000, 20_000, 10_000, 5_000, 1_000, 500, 100]),
    # ("bike", [12_000, 10_000, 5_000, 1_000, 500, 100]),
]
repeat, mu = 4, 0.1
tic = time.time()
for dataset, Ns in cases:
    results[dataset] = {}
    # train_x, train_y, *_ = load_uci_data("/home/ubu/Downloads/", dataset)
    train_x, train_y, *_ = load_uci_data(f"/datasets/uci/{dataset}", dataset)
    print(f"Dataset (N={train_x.shape[0]:,d} | D={train_x.shape[1]:,d})")
    for N in Ns:
        results[dataset][N] = {}
        X = xnp.array(train_x[:N], dtype=dtype)
        y = xnp.array(train_y[:N], dtype=dtype)
        XTX = lazify(X.T) @ lazify(X)
        # XTX += xnp.array(mu, dtype=dtype) * Identity(shape=(X.shape[1], X.shape[1]), dtype=dtype)
        XTX += Diagonal(mu * xnp.ones(shape=(X.shape[1],), dtype=dtype))
        XTX = SelfAdjoint(XTX)
        rhs = X.T @ y

        lin_kwargs = {"method": "cg", "info": True, "tol": 1e-4, "max_iters": 1000}
        get_times(XTX, rhs, lin_kwargs, results[dataset][N], xnp, repeat, key="linops")
        get_times_sk_linear(X, y, mu, results[dataset][N], repeat)

toc = time.time()
print_time_taken(toc - tic)
if save_output:
    save_object(results, filepath=output_path)
