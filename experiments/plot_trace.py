from matplotlib import pyplot as plt
import seaborn as sns
from linops.experiment_utils import load_object
from linops.experiment_utils import convert_results_to_df

results = {}
input_path = "./logs/trace_opt_20230523_1831.pkl"
# input_path = "./logs/trace_opt.pkl"
results["cola"] = load_object(input_path)
input_path = "./logs/trace_loop_20230523_1830.pkl"
# input_path = "./logs/trace_loop.pkl"
results["iterative"] = load_object(input_path)
df = convert_results_to_df(results, var_name="memory", time_name="time", skip=0)
df.rename(columns={"ds_size": "case", "case": "iters", "sizes": "memory"}, inplace=True)
keys = ["iterative", "cola"]
an = {
    "iterative": {
        "color": "#7fc97f",
        "label": "iterative"
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
    iters = df[mask]["iters"]
    idx = iters.argsort()
    iters = iters.iloc[idx]
    times = df[mask]["times"].iloc[idx]
    plt.plot(iters, times, label=an[key]["label"], c=an[key]["color"], lw=6)
    plt.scatter(iters, times, c=an[key]["color"], lw=8)
plt.xlabel("CG iters")
plt.ylabel("Runtime (sec)")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("trace_runtime.pdf")
plt.show()

plt.figure(dpi=50, figsize=(14, 10))
for key in keys:
    mask = df["case"] == key
    iters = df[mask]["iters"]
    idx = iters.argsort()
    iters = iters.iloc[idx]
    memory = df[mask]["memory"].iloc[idx] / 1e6
    plt.plot(iters, memory, label=an[key]["label"], c=an[key]["color"], lw=6)
    plt.scatter(iters, memory, c=an[key]["color"], lw=8)
plt.xlabel("CG iters")
plt.ylabel("Memory (MB)")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("trace_memory.pdf")
plt.show()
