from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from linops.experiment_utils import load_object
from linops.experiment_utils import convert_results_to_df

input_path = "./logs/linear_regression_gpu_20230516_2147.pkl"
results = load_object(input_path)
df0 = convert_results_to_df(results["song"], var_name="system_size")
df0["dataset"] = "song"
mask = df0["case"] == "linops"
df0["case"][mask] = "linops_gpu"
input_path = "./logs/linear_regression_20230516_2116.pkl"
# input_path = "./logs/linear_regression.pkl"
results = load_object(input_path)
df1 = convert_results_to_df(results["protein"], var_name="system_size")
df1["dataset"] = "protein"
df2 = convert_results_to_df(results["song"], var_name="system_size")
df2["dataset"] = "song"
df = pd.concat((df1, df2, df0))
# datasets = ["song", "protein"]
datasets = ["song"]
keys = ["sklearn", "linops_gpu", "linops"]
an = {
    "sklearn": {
        "song": {
            "color": "#9e9ac8",
            "label": "sk"
        },
        "protein": {
            "color": "#54278f",
            "label": "sk (pro)"
        },
    },
    "linops": {
        "song": {
            "color": "#74a9cf",
            "label": "CoLA (CPU)"
        },
        "protein": {
            "color": "#045a8d",
            "label": "CoLA (pro)"
        },
    },
    "linops_gpu": {
        "song": {
            "color": "#045a8d",
            "label": "CoLA (GPU)"
        },
        "protein": {
            "color": "#045a8d",
            "label": "CoLA (pro)"
        },
    },
}

sns.set(style="whitegrid", font_scale=4.0)
plt.figure(dpi=50, figsize=(14, 10))
for key in keys:
    for ds in datasets:
        mask = (df["case"] == key) & (df["dataset"] == ds)
        sizes = df[mask]["ds_size"]
        idx = sizes.argsort()
        sizes = sizes.iloc[idx]
        times = df[mask]["times"].iloc[idx]
        plt.plot(sizes, times, label=an[key][ds]["label"], c=an[key][ds]["color"], lw=6)
        plt.scatter(sizes, times, c=an[key][ds]["color"], lw=8)
plt.xlabel("Dasetset Size")
plt.ylabel("Runtime (sec)")
plt.yscale("log")
plt.xscale("log")
# plt.ylim([1e-4, 1e-0])
plt.legend()
plt.tight_layout()
plt.savefig("linear_regression.pdf")
plt.show()
