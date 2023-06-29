from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from cola.experiment_utils import load_object

sns.set(style="whitegrid", font_scale=4.0)
sns.set_palette("Set2")
palette = sns.color_palette()

cola_cpu = load_object("./logs/timings_cola_cpu.pkl")
cola_gpu = load_object("./logs/timings_cola_gpu.pkl")
scipy = load_object("./logs/timings_scipy.pkl")
results = [cola_cpu, cola_gpu, scipy]
labels = ["cola (CPU)", "cola (GPU)", "SciPy (CPU)", "GPyTorch (GPU)"]
colors = ["#2b8cbe", "#a6bddb", "#7fc97f", "#e34a33"]
# colors = [palette[0], palette[1]]

skip = 1
plt.figure(dpi=50, figsize=(14, 10))
for idx, result in enumerate(results):
    times = np.mean(result["times"][:, skip:], axis=1)
    sizes = result["sizes"]
    plt.plot(sizes, times, label=labels[idx], c=colors[idx], lw=6)
    plt.scatter(sizes, times, c=colors[idx], lw=8)
plt.xlabel("Dataset Size")
plt.ylabel("Runtime")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("runtimes.pdf")
plt.show()
