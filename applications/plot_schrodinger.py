from matplotlib import pyplot as plt
import seaborn as sns
from cola.experiment_utils import load_object
from cola.experiment_utils import convert_results_to_df

input_path = "./logs/schrodinger_20230515_1630.pkl"
# input_path = "./logs/schrodinger.pkl"
results = load_object(input_path)
df = convert_results_to_df(results, var_name="grid_size")
keys = ["scipy", "cola"]
an = {
    "scipy": {
        "color": "#7fc97f",
        "label": "SciPy"
    },
    "cola": {
        "color": "#2b8cbe",
        "label": "CoLA"
    }
}

sns.set(style="whitegrid", font_scale=4.0)
plt.figure(dpi=50, figsize=(14, 10))
for key in keys:
    mask = df["case"] == key
    sizes = df[mask]["sizes"]
    idx = sizes.argsort()
    sizes = sizes.iloc[idx]
    times = df[mask]["times"].iloc[idx]
    plt.plot(sizes, times, label=an[key]["label"], c=an[key]["color"], lw=6)
    plt.scatter(sizes, times, c=an[key]["color"], lw=8)
plt.xlabel("Grid Size")
plt.ylabel("Runtime (sec)")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("schrodinger.pdf")
plt.show()
