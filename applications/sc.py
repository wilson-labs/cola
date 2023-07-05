from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import SpectralClustering
import torch
import numpy as np
from sklearn.cluster import KMeans
import cola.torch_fns as xnp
from cola.linalg.eigs import eig
from cola.linalg.sqrt import sqrt
from cola.linalg.inverse import inverse
from cola.experiment_utils import data_to_csr_neighbors
from cola.ops import SelfAdjoint
from cola.ops import I_like
from cola.ops import Sparse
from cola.ops import Diagonal


torch.manual_seed(21)
np.random.seed(21)
dtype = xnp.float64
n_clusters = 2
embedding_size = 8
lanczos_iters = 10
# noisy_data = datasets.make_circles(n_samples=500, factor=0.5, noise=0.05)
noisy_data = datasets.make_moons(n_samples=500, noise=0.05)
x, _ = noisy_data
SC = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors").fit(x)
x = xnp.array(x, dtype=dtype)
data, indices, indptr, shape = data_to_csr_neighbors(x, xnp, dtype)
# data = xnp.array(SC.affinity_matrix_.data, dtype=dtype)
# indices = xnp.array(SC.affinity_matrix_.indices, dtype=xnp.int64)
# indptr = xnp.array(SC.affinity_matrix_.indptr, dtype=xnp.int64)

As = Sparse(data, indices, indptr, shape)
diff = np.linalg.norm(SC.affinity_matrix_.todense() - np.array(As.to_dense()))
degrees = Diagonal(As @ xnp.ones(shape=(As.shape[0], ), dtype=dtype))
# laplacian = SelfAdjoint(degrees - As)
laplacian = SelfAdjoint(I_like(degrees) - inverse(sqrt(degrees)) @ As @ inverse(sqrt(degrees)))
eigvals, eigvecs = eig(laplacian, method="lanczos", max_iters=lanczos_iters)
x_emb = eigvecs[:, :-embedding_size]
kmeans = KMeans(n_clusters=n_clusters).fit(x_emb)

# plt.figure()
# plt.scatter(x[:, 0], x[:, 1])
# plt.show()

plt.figure()
for cl in range(n_clusters):
    mask = kmeans.labels_ == cl
    plt.scatter(x[mask, 0], x[mask, 1])
plt.show()
