from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
# from linops.experiment_utils import load_object

sns.set(style="whitegrid", font_scale=4.0)

# linops_cpu = load_object("./logs/neural_pdes_linops_cpu.pkl")
# linops_gpu = load_object("./logs/neural_pdes_linops_gpu.pkl")
# scipy = load_object("./logs/neural_pdes_scipy.pkl")
linops = {}
linops["times"] = np.array([0., 0.5, 1.0, 1.5, 2.0])
linops["res"] = np.array([3e-4, 1e-3, 5e-4, 6e-4, 3e-3])
scipy = {}
scipy["times"] = np.array([0., 0.5, 1.0, 1.5, 2.0])
scipy["res"] = np.array([4e-4, 2e-3, 4e-4, 5e-4, 4e-3])
results = [linops, scipy]
labels = ["LinOps", "Neural-IVP"]
colors = ["#2b8cbe", "#636363"]

plt.figure(dpi=50, figsize=(14, 10))
for idx, result in enumerate(results):
    times = result["times"]
    res = result["res"]
    plt.plot(times, res, label=labels[idx], c=colors[idx], lw=6)
    plt.scatter(times, res, c=colors[idx], lw=8)
plt.text(1.4, 3.7e-3, "539 sec")
plt.text(1.3, 2.3e-3, "470 sec")
# plt.annotate('', xy=(2.2, 3.7e-3), xytext=(1.5, 3.7e-3),
#              arrowprops=dict(facecolor='black', shrink=0.05))
plt.xlabel("Time Evolution")
plt.ylabel("PDE Residual")
plt.yscale("log")
plt.ylim([1e-4, 1e-2])
plt.legend()
plt.tight_layout()
plt.savefig("neural_pdes.pdf")
plt.show()
