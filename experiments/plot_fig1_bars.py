from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


sns.set(style="whitegrid", font_scale=4.0)
# plt.rcParams['lines.linewidth'] = 6
colors = {"LinOps": "#2b8cbe", "GPyTorch": "#e34a33", "sklearn": "#8856a7", "SciPy": "#7fc97f"}

gps_gpytorch = np.array([0.1, 0.1])
gps_linops = np.array([0.11, 0.12])

minimal_scipy = np.array([0.34, 0.5])
minimal_linops = np.array([0.15, 0.2])

spectral_sklearn = np.array([0.6, 0.7])
spectral_linops = np.array([0.15, 0.18])


start, put_label = 0, True
barWidth = 0.25
skip = 1
axis_lables = ["GPs", "Spectral\nClustering", "Minimal\nSurface"]
plt.figure(dpi=50, figsize=(14, 10))
br1 = [start]
br2 = [x + barWidth for x in br1]
label = "LinOps"
plt.bar(br2, np.mean(gps_linops[skip:]), color=colors[label],
        width=barWidth, edgecolor='black', label=label, lw=4)
label = "GPyTorch"
plt.bar(br1, np.mean(gps_gpytorch[skip:]), color=colors[label],
        width=barWidth, edgecolor='black', label=label, lw=4)

start += 2. * barWidth + barWidth / 2.
br1 = [start]
br2 = [x + barWidth for x in br1]
label = "sklearn"
plt.bar(br1, np.mean(spectral_sklearn[skip:]), color=colors[label],
        width=barWidth, edgecolor='black', label=label, lw=4)
label = "LinOps"
plt.bar(br2, np.mean(spectral_linops[skip:]), color=colors[label],
        width=barWidth, edgecolor='black', lw=4)

start += 2. * barWidth + barWidth / 2.
br1 = [start]
br2 = [x + barWidth for x in br1]
label = "SciPy"
plt.bar(br1, np.mean(minimal_scipy[skip:]), color=colors[label],
        width=barWidth, edgecolor='black', label=label, lw=4)
label = "LinOps"
plt.bar(br2, np.mean(minimal_linops[skip:]), color=colors[label],
        width=barWidth, edgecolor='black', lw=4)

plt.ylabel("Runtime")
plt.legend(loc="upper left")
plt.xticks([0.1, 0.7, 1.3], axis_lables)
plt.tight_layout()
plt.savefig("linops_vs.pdf")
plt.show()
