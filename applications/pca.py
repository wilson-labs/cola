import time
import jax.config
import cola.jax_fns as xnp
from cola.basic_operations import lazify
from cola.experiment_utils import get_times_pca, get_times_pca_sk
from cola.experiment_utils import load_uci_data, print_time_taken, save_object

jax.config.update("jax_enable_x64", True)

save_output = True
output_path = "./logs/pca.pkl"
dtype, results, repeat = xnp.float64, {}, 3
train_x, *_ = load_uci_data(data_dir="/home/ubu/Downloads/", dataset="buzz")
print(f"Dataset (N={train_x.shape[0]:,d} | D={train_x.shape[1]:,d})")
N, rank = 430_000, 3_000
ks = [50, 25, 10]
X = xnp.array(train_x[:N], dtype=dtype)
XTX = lazify(X.T / X.shape[0]) @ lazify(X)

tic = time.time()
for pca_num in ks:
    results[pca_num] = {}
    get_times_pca(XTX, (pca_num, rank), results[pca_num], repeat)
    get_times_pca_sk(X, pca_num, results[pca_num], repeat)

toc = time.time()
print_time_taken(toc - tic)
if save_output:
    save_object(results, filepath=output_path)
