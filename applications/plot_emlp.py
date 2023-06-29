from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style="whitegrid", font_scale=4.0)
colors = ["#2b8cbe", "#7fc97f"]

sk1 = np.array([5e-1, 5e-1])
cola1 = np.array([4.5e-1, 4.5e-1])
sk2 = np.array([5e-0, 5e-0])
cola2 = np.array([4e-0, 4e-0])
sk3 = np.array([3e+1, 3e+1])
cola3 = np.array([7e-0, 7e-0])

start = 0
barWidth = 0.25
skip = 1
axis_lables = ["N=$10^2$", "N=$10^3$", "N=$10^4$"]
plt.figure(dpi=50, figsize=(14, 10))
br1 = [start]
br2 = [x + barWidth for x in br1]
plt.bar(br2, np.mean(cola1[skip:]), color=colors[0],
        width=barWidth, edgecolor='black', label="cola", lw=4)
plt.bar(br1, np.mean(sk1[skip:]), color=colors[-1],
        width=barWidth, edgecolor='black', label="EMLP (JAX)", lw=4)

start += 2. * barWidth + barWidth / 2.
br1 = [start]
br2 = [x + barWidth for x in br1]
plt.bar(br1, np.mean(sk2[skip:]), color=colors[-1],
        width=barWidth, edgecolor='black', lw=4)
plt.bar(br2, np.mean(cola2[skip:]), color=colors[0],
        width=barWidth, edgecolor='black', lw=4)

start += 2. * barWidth + barWidth / 2.
br1 = [start]
br2 = [x + barWidth for x in br1]
plt.bar(br1, np.mean(sk3[skip:]), color=colors[-1],
        width=barWidth, edgecolor='black', lw=4)
plt.bar(br2, np.mean(cola3[skip:]), color=colors[0],
        width=barWidth, edgecolor='black', lw=4)

plt.ylabel("Runtime")
plt.xlabel("Training Set Size")
plt.yscale("log")
plt.legend()
plt.xticks([0.1, 0.8, 1.4], axis_lables)
plt.tight_layout()
plt.savefig("emlp.pdf")
plt.show()
