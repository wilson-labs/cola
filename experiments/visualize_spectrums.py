import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from cola.utils_test import generate_spectrum
from cola.utils_test import generate_clustered_spectrum
from cola.utils_test import generate_beta_spectrum

sns.set(style="whitegrid", font_scale=2.0, rc={"lines.linewidth": 3.0})
sns.set_palette("Set1")


def plot_diag(diag):
    plt.figure(dpi=100, figsize=(8, 6))
    plt.plot(np.arange(diag.shape[0]), diag, label="spectrum")
    plt.scatter(np.arange(diag.shape[0]), diag)
    plt.xlabel("index")
    plt.ylabel("eigenvalue")
    plt.legend()
    plt.tight_layout()
    plt.show()


diag = generate_beta_spectrum(coeff=0.5, alpha=0.5, beta=0.1, seed=48, scale=100.0,
                              size=100)
plot_diag(diag)
diag = generate_spectrum(coeff=0.50, scale=1.0, size=50, dtype=np.float32)
# plot_diag(diag)
clusters = [0.1, 0.3, 0.85, 0.99]
sizes = [8, 5, 3, 3]
diag = generate_clustered_spectrum(clusters, sizes, std=0.025, seed=48)
# plot_diag(diag)
clusters = [0.1, 0.11, 0.12, 0.13, 0.14, 0.2, 0.25, 0.28, 0.3, 0.4, 0.5, 0.99]
diag = generate_clustered_spectrum(clusters, [1 for _ in range(len(clusters))], std=0.0, seed=48)
# plot_diag(diag)
