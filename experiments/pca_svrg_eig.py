import time
import jax.config
import cola.jax_fns as xnp
import cola as lo
from cola.linear_algebra import lazify
from cola.algorithms.svrg import svrg_eigh_max
from cola.experiment_utils import load_uci_data, print_time_taken, save_object
jax.config.update("jax_enable_x64", True)

save_output = True
output_path = "./logs/pca_svrg_eig.pkl"
dtype, results = xnp.float64, {}
train_x, *_ = load_uci_data(data_dir="/home/ubu/Downloads/", dataset="buzz")
print(f"Dataset (N={train_x.shape[0]:,d} | D={train_x.shape[1]:,d})")
N = 430_000
X = xnp.array(train_x[:N], dtype=dtype)
XTX = lazify(X.T / X.shape[0]) @ lazify(X)
bs = 10_000

tic = time.time()
(eigs, _), info = svrg_eigh_max(XTX, pbar=True, info=True, max_iters=300, bs=bs, lr_scale=3e-2,
                                tol=1e-11)
results["svrg"] = info["errors"]
out2, info2 = lo.linalg.eigmax(XTX, pbar=True, info=True, max_iters=300, tol=1e-7)
results["power"] = info2["errors"]

print(f"SVRG: {eigs[0]:1.5e} | Power: {out2:1.5e}")

toc = time.time()
print_time_taken(toc - tic)
if save_output:
    save_object(results, filepath=output_path)
