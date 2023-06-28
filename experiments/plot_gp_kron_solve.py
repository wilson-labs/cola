from matplotlib import pyplot as plt
import seaborn as sns
from linops.experiment_utils import load_object
from linops.experiment_utils import convert_results_to_df

input_path = "./logs/gp_kron_solve_20230509_1946.pkl"
results = load_object(input_path)
df = convert_results_to_df(results)
keys = ["dense", "iterative", "linops"]
an = {
    "dense": {
        "color": "#636363",
        "label": "Dense"
    },
    "iterative": {
        "color": "#de2d26",
        "label": "Iterative"
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
    sizes = df[mask]["sizes"] / 1e2
    idx = sizes.argsort()
    sizes = sizes.iloc[idx]
    times = df[mask]["times"].iloc[idx]
    plt.plot(sizes, times, label=an[key]["label"], c=an[key]["color"], lw=6)
    plt.scatter(sizes, times, c=an[key]["color"], lw=8)
plt.xlabel("Size ($10^2$)")
plt.ylabel("Runtime (sec)")
plt.yscale("log")
plt.ylim([1e-4, 1e-0])
plt.legend()
plt.tight_layout()
plt.savefig("gps_kron_solve.pdf")
plt.show()
