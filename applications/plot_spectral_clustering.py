from matplotlib import pyplot as plt
import seaborn as sns
from linops.experiment_utils import load_object
from linops.experiment_utils import convert_results_to_df

# input_path = "./logs/spectral_20230512_1237.pkl"
input_path = "./logs/spectral.pkl"
results = load_object(input_path)
df = convert_results_to_df(results, var_name="edges")
# keys = ["amg", "linops"]
keys = ["amg", "arpack", "linops"]
an = {
    "arpack": {
        "color": "#9e9ac8",
        "label": "sk (L)"
    },
    "amg": {
        "color": "#54278f",
        "label": "sk (A)"
    },
    "linops": {
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
plt.xlabel("Edges")
plt.ylabel("Runtime (sec)")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("spectral_clustering.pdf")
plt.show()
