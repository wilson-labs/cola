from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
# from cola.experiment_utils import load_object

sns.set(style="whitegrid", font_scale=4.0)

# cola_cpu = load_object("./logs/bipoisson_cola_cpu.pkl")
# cola_gpu = load_object("./logs/bipoisson_cola_gpu.pkl")
# scipy = load_object("./logs/bipoisson_scipy.pkl")
cola_cpu = {}
cola_cpu["times"] = np.array([5, 15, 17, 20, 22])
cola_cpu["res"] = np.array([1.e-2, 1.e-3, 1e-4, 1e-5, 1e-6])
cola_gpu = {}
cola_gpu["times"] = np.array([5, 7, 8, 10, 12])
cola_gpu["res"] = np.array([1.e-2, 1.e-3, 1e-4, 1e-5, 1e-6])
scipy = {}
scipy["times"] = np.array([5, 10, 20, 30, 40])
scipy["res"] = np.array([1.e-2, 1.e-3, 1e-4, 1e-5, 1e-6])
results = [cola_cpu, cola_gpu, scipy]
labels = ["cola (CPU)", "cola (GPU)", "SciPy (CPU)"]
colors = ["#2b8cbe", "#a6bddb", "#7fc97f"]

plt.figure(dpi=50, figsize=(14, 10))
for idx, result in enumerate(results):
    times = result["times"]
    res = result["res"]
    plt.plot(times, res, label=labels[idx], c=colors[idx], lw=6)
    plt.scatter(times, res, c=colors[idx], lw=8)
plt.xlabel("Runtime")
plt.ylabel("Residual")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("bipoisson.pdf")
plt.show()
