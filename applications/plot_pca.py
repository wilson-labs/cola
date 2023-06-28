from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from linops.experiment_utils import load_object
from linops.experiment_utils import convert_results_to_df

input_path = "./logs/pca_20230516_2026.pkl"
results = load_object(input_path)
df = convert_results_to_df(results, var_name="system_size")

mask = (df["case"] == "sklearn") & (df["ds_size"] == 10)
sk1 = np.array(df[mask]["times"])
mask = (df["case"] == "linops") & (df["ds_size"] == 10)
linops1 = np.array(df[mask]["times"])

mask = (df["case"] == "sklearn") & (df["ds_size"] == 25)
sk2 = np.array(df[mask]["times"])
mask = (df["case"] == "linops") & (df["ds_size"] == 25)
linops2 = np.array(df[mask]["times"])

mask = (df["case"] == "sklearn") & (df["ds_size"] == 50)
sk3 = np.array(df[mask]["times"])
mask = (df["case"] == "linops") & (df["ds_size"] == 50)
linops3 = np.array(df[mask]["times"])

sns.set(style="whitegrid", font_scale=4.0)
colors = ["#2b8cbe", "#8856a7"]
skip = 0
start = 0
barWidth = 0.25
axis_lables = ["k=5", "k=10", "k=20"]
plt.figure(dpi=50, figsize=(14, 10))
br1 = [start]
br2 = [x + barWidth for x in br1]
plt.bar(br2, np.mean(linops1[skip:]), color=colors[0], width=barWidth, edgecolor='black',
        label="CoLA", lw=4)
plt.bar(br1, np.mean(sk1[skip:]), color=colors[-1], width=barWidth, edgecolor='black',
        label="sk", lw=4)

start += 2. * barWidth + barWidth / 2.
br1 = [start]
br2 = [x + barWidth for x in br1]
plt.bar(br1, np.mean(sk2[skip:]), color=colors[-1], width=barWidth, edgecolor='black', lw=4)
plt.bar(br2, np.mean(linops2[skip:]), color=colors[0], width=barWidth, edgecolor='black', lw=4)

start += 2. * barWidth + barWidth / 2.
br1 = [start]
br2 = [x + barWidth for x in br1]
plt.bar(br1, np.mean(sk3[skip:]), color=colors[-1], width=barWidth, edgecolor='black', lw=4)
plt.bar(br2, np.mean(linops3[skip:]), color=colors[0], width=barWidth, edgecolor='black', lw=4)

plt.ylabel("Runtime (sec)")
plt.xlabel("PCA Components")
plt.yscale("log")
plt.ylim([1e-1, 1e+1])
plt.legend()
plt.xticks([0.1, 0.7, 1.3], axis_lables)
plt.tight_layout()
plt.savefig("pca.pdf")
plt.show()
