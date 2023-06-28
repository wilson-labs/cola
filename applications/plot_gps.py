from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from linops.experiment_utils import load_object

sns.set(style="whitegrid", font_scale=4.0)

linops = load_object("./logs/gps_elevators_linops_20230516_2029.pkl")
df1 = pd.DataFrame([val for _, val in linops.items()])
df1["diff"] = df1["time"] - df1["time"].iloc[0]
linops = load_object("./logs/gps_kin40k_linops_20230516_2029.pkl")
df1p = pd.DataFrame([val for _, val in linops.items()])
df1p["diff"] = df1p["time"] - df1p["time"].iloc[0]
gp = load_object("./logs/gps_elevators_master_20230516_2029.pkl")
df2 = pd.DataFrame([val for _, val in gp.items()])
df2["diff"] = df2["time"] - df2["time"].iloc[0]
gp = load_object("./logs/gps_kin40k_master_20230516_2029.pkl")
df2p = pd.DataFrame([val for _, val in gp.items()])
df2p["diff"] = df2p["time"] - df2p["time"].iloc[0]

linops1 = {}
linops1["times"] = np.array(df1["diff"])
linops1["loss"] = np.array(df1["iter"])
# linops1["loss"] = np.array(df1["loss"])
gp1 = {}
gp1["times"] = np.array(df2["diff"])
gp1["loss"] = np.array(df2["iter"])
# gp1["loss"] = np.array(df2["loss"])

linops2 = {}
linops2["times"] = np.array(df1p["diff"])
linops2["loss"] = np.array(df1p["iter"])
gp2 = {}
gp2["times"] = np.array(df2p["diff"])
gp2["loss"] = np.array(df2p["iter"])

# results = [linops1, gp1]
# labels = ["LO (ele)", "GP (ele)"]
# colors = ["#2b8cbe", "#e34a33"]
results = [linops1, gp1, linops2, gp2]
labels = ["CoLA (ele)", "GP (ele)", "CoLA (kin)", "GP (kin)"]
colors = ["#2b8cbe", "#e34a33", "#a6bddb", "#fdbb84"]

plt.figure(dpi=50, figsize=(14, 10))
for idx, result in enumerate(results):
    times = result["times"]
    loss = result["loss"]
    plt.plot(times, loss, label=labels[idx], c=colors[idx], lw=6)
plt.xlabel("Runtime (sec)")
# plt.ylabel("Loss")
plt.ylabel("Epochs")
# plt.ylim([1.25, 1.4])
plt.xlim([0, 50])
plt.legend()
plt.tight_layout()
plt.savefig("gps.pdf")
plt.show()
