from matplotlib import pyplot as plt
import seaborn as sns
from cola.experiment_utils import load_object
from cola.experiment_utils import convert_results_to_df

input_path = "./logs/minimal_20230515_1504.pkl"
# input_path = "./logs/minimal.pkl"
results = load_object(input_path)
df = convert_results_to_df(results, var_name="grid_size")
keys = ["scipy", "cola", "jax"]
an = {
    "scipy": {
        # "color": "#006d2c",
        "color": "#d95f0e",
        "label": "SciPy"
    },
    "jax": {
        # "color": "#66c2a4",
        "color": "#fe9929",
        "label": "SciPy (JAX)"
    },
    "cola": {
        "color": "#2b8cbe",
        # "color": "#045a8d",
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
plt.savefig("minimal_surface.pdf")
plt.show()
