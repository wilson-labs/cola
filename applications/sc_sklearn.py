from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import SpectralClustering

n_clusters = 2
embedding_size = 8
# noisy_data = datasets.make_circles(n_samples=500, factor=0.5, noise=0.05)
noisy_data = datasets.make_moons(n_samples=500, noise=0.05)
x, _ = noisy_data
# SC = SpectralClustering(n_clusters=n_clusters).fit(x)
SC = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors").fit(x)

plt.figure()
for cl in range(n_clusters):
    mask = SC.labels_ == cl
    plt.scatter(x[mask, 0], x[mask, 1])
plt.show()
