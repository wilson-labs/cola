from matplotlib import pyplot as plt
import torch
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans
import cola.torch_fns as xnp
from cola.linalg.eigs import eig
# from cola.linalg.sqrt import sqrt
from cola.linalg.inverse import inverse
from cola.experiment_utils import data_to_csr
# from cola.experiment_utils import data_to_csr_neighbors
from cola.operators import SelfAdjoint
from cola.operators import I_like
from cola.operators import Sparse
from cola.operators import Diagonal

sns.set(style="whitegrid", font_scale=3.0, rc={"lines.linewidth": 4.0})
sns.set_palette("Set2")

torch.manual_seed(21)
np.random.seed(21)
dtype = xnp.float64
n_clusters = 2
embedding_size = 2
lanczos_iters = 50
# noisy_data = datasets.make_circles(n_samples=500, factor=0.5, noise=0.05)
noisy_data = datasets.make_moons(n_samples=500, noise=0.05)
x, _ = noisy_data
x = xnp.array(x, dtype=dtype)
data, indices, indptr, shape = data_to_csr(x, xnp, gamma=30., threshold=0.1)
# data, indices, indptr, shape = data_to_csr_neighbors(x, xnp, dtype)
As = Sparse(data, indices, indptr, shape)
# diff = xnp.norm(As.to_dense() - As.T.to_dense())
degrees = Diagonal(As @ xnp.ones(shape=(As.shape[0], ), dtype=dtype))
# laplacian = SelfAdjoint(degrees - As)
laplacian = SelfAdjoint(I_like(degrees) - inverse(degrees) @ As)
# laplacian = SelfAdjoint(I_like(degrees) - inverse(sqrt(degrees)) @ As @ inverse(sqrt(degrees)))
eigvals, eigvecs = eig(laplacian, method="lanczos", max_iters=lanczos_iters)
x_emb = eigvecs[:, :embedding_size]
kmeans = KMeans(n_clusters=n_clusters).fit(x_emb)

idx = -1
plt.figure(dpi=100, figsize=(12, 10))
plt.plot(np.arange(len(eigvecs[:, idx])), np.array(eigvecs[:, idx]))
plt.scatter(np.arange(len(eigvecs[:, idx])), np.array(eigvecs[:, idx]))
plt.xlabel("Index")
plt.ylabel(f"Eigenvector ({idx})")
plt.ylim([-0.2, 0.2])
plt.tight_layout()
plt.show()

# plt.figure(dpi=100, figsize=(12, 10))
# plt.plot(np.arange(len(eigvals)), np.array(eigvals)[::-1])
# plt.scatter(np.arange(len(eigvals)), np.array(eigvals)[::-1])
# plt.xlabel("Index")
# plt.ylabel("Eigenvalues")
# plt.tight_layout()
# plt.show()
#
#
# plt.figure()
# for cl in range(n_clusters):
#     mask = kmeans.labels_ == cl
#     plt.scatter(x[mask, 0], x[mask, 1])
# plt.tight_layout()
# plt.show()
