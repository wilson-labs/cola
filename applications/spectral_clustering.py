import time
import linops.torch_fns as xnp
from linops.experiment_utils import load_graph_data
from linops.linalg.sqrt import sqrt
from linops.linalg.inverse import inverse
from linops.operators import Sparse, Diagonal, SelfAdjoint, I_like
from linops.experiment_utils import get_times_spectral, get_times_spectral_sklearn
from linops.experiment_utils import transform_to_csr
from linops.experiment_utils import print_time_taken, save_object

save_output = True
embedding_size, n_clusters = 8, 8
repeat, results, N_max = 3, {}, 500_000
Ns = [(500_000, 100), (100_000, 100), (50_000, 50), (10_000, 20), (5_000, 20), (1_000, 20)]
dtype = xnp.float64
output_path = "./logs/spectral.pkl"
filepath = "/home/ubu/Downloads/cit-HepPh.txt"

tic = time.time()
for num_edges, lanczos_iters in Ns:
    results[num_edges] = {}
    sparse_matrix = load_graph_data(filepath=filepath, num_edges=num_edges)
    data, indices, indptr, shape = transform_to_csr(sparse_matrix, xnp, dtype)
    print(f"Edges: {data.shape[0]:,d}")
    As = Sparse(data, indices, indptr, shape)
    degrees = Diagonal(As @ xnp.ones(shape=(As.shape[0], ), dtype=dtype))
    laplacian = SelfAdjoint(I_like(degrees) - inverse(sqrt(degrees)) @ As @ inverse(sqrt(degrees)))

    args = (lanczos_iters, embedding_size, n_clusters, data.shape[0])
    results[num_edges] = get_times_spectral(laplacian, results[num_edges], repeat, args)
    results[num_edges] = get_times_spectral_sklearn(sparse_matrix, results[num_edges], repeat,
                                                    eigen_solver="amg")
    if num_edges < N_max:
        results[num_edges] = get_times_spectral_sklearn(sparse_matrix, results[num_edges], repeat,
                                                        eigen_solver="arpack")

toc = time.time()
print_time_taken(toc - tic)
if save_output:
    save_object(results, filepath=output_path)
