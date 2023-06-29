import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from cola.experiment_utils import load_object
from palettable.cmocean.sequential import Thermal_8

palette = Thermal_8.mpl_colors
sns.set(style="whitegrid", font_scale=2.0, palette=palette, rc={"lines.linewidth": 3.0})
barWidth = 0.25
color_dense = Thermal_8.mpl_colors[2]
color_dispatch = Thermal_8.mpl_colors[-2]
color_iterative = Thermal_8.mpl_colors[-1]
paths = [
    "./logs/kronecker_solve_double_cpu.pkl",
    "./logs/kronecker_solve_single_cpu.pkl",
    "./logs/kronecker_solve_single_gpu.pkl",
]
paths += [
    "./logs/prod_solve_double_cpu.pkl",
    "./logs/prod_solve_single_cpu.pkl",
    "./logs/prod_solve_single_gpu.pkl",
]
paths += [
    "./logs/block_solve_double_cpu.pkl",
    "./logs/block_solve_single_cpu.pkl",
    "./logs/block_solve_single_gpu.pkl",
]
axis_lables = [
    "Kron\nCPU D", "Kron\nCPU S", "Kron\nGPU S", "Prod\nCPU D", "Prod\nCPU S", "Prod\nGPU S",
    "Block\nCPU D", "Block\nCPU S", "Block\nGPU S"
]

start, put_label = 0, True
plt.figure(dpi=100, figsize=(16, 5))
for input_path in paths:
    results_kron = load_object(input_path)
    dense_rmse = np.mean(results_kron["dense"]["times"][1:])
    dispatch_rmse = np.mean(results_kron["dispatch"]["times"][1:])
    iterative_rmse = np.mean(results_kron["iterative"]["times"][1:])
    dense_error = np.std(results_kron["dense"]["times"][1:])
    dispatch_error = np.std(results_kron["dispatch"]["times"][1:])
    iterative_error = np.std(results_kron["iterative"]["times"][1:])

    br1 = [start]
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    start += 3. * barWidth + barWidth / 2.

    plt.bar(br1, dense_rmse, color=color_dense, width=barWidth, edgecolor='black',
            label='Dense' if put_label else '')
    plt.bar(br2, iterative_rmse, color=color_iterative, width=barWidth, edgecolor='black',
            label='Iterative' if put_label else '')
    plt.bar(br3, dispatch_rmse, color=color_dispatch, width=barWidth, edgecolor='black',
            label='Dispatch' if put_label else '')
    plt.errorbar(br1, y=dense_rmse, yerr=dense_error, fmt="o", color="black")
    plt.errorbar(br2, y=iterative_rmse, yerr=iterative_error, fmt="o", color="black")
    plt.errorbar(br3, y=dispatch_rmse, yerr=dispatch_error, fmt="o", color="black")

    put_label = False

ticks = [0.25, 1.12, 2., 2.9, 3.72, 4.6, 5.5, 6.4, 7.3]
plt.xticks(ticks, axis_lables)
plt.ylabel("Runtime")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("./logs/bar_comp.pdf")
plt.show()
