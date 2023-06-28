import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from linops.experiment_utils import load_object
from palettable.cmocean.sequential import Thermal_8

palette = Thermal_8.mpl_colors
sns.set(style="white", font_scale=2.0, palette=palette, rc={"lines.linewidth": 3.0})
# sns.set(style="whitegrid", font_scale=2.0, rc={"lines.linewidth": 3.0})
# sns.set_palette("Set1")

# input_path = "./logs/kronecker_solve.pkl"
# input_path = "./logs/kronecker_solve_double_cpu.pkl"
input_path = "./logs/kronecker_solve_single_cpu.pkl"
results = load_object(input_path)
color_dense = Thermal_8.mpl_colors[2]
color_dispatch = Thermal_8.mpl_colors[-2]
color_iterative = Thermal_8.mpl_colors[-1]

dense_rmse = np.mean(results["dense"]["times"][1:])
dispatch_rmse = np.mean(results["dispatch"]["times"][1:])
iterative_rmse = np.mean(results["iterative"]["times"][1:])
dense_error = np.std(results["dense"]["times"][1:])
dispatch_error = np.std(results["dispatch"]["times"][1:])
iterative_error = np.std(results["iterative"]["times"][1:])

barWidth = 0.25
br1 = np.arange(1)
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
plt.figure(dpi=100, figsize=(12, 8))
plt.title("Timing results for Kronecker Solve")
plt.bar(br1, dense_rmse, color=color_dense, width=barWidth, edgecolor='black',
        label='Dense')
plt.bar(br2, iterative_rmse, color=color_iterative, width=barWidth, edgecolor='black',
        label='Iterative')
plt.bar(br3, dispatch_rmse, color=color_dispatch, width=barWidth, edgecolor='black',
        label='Dispatch')
plt.errorbar(br1, y=dense_rmse, yerr=dense_error, fmt="o", color="black")
plt.errorbar(br2, y=iterative_rmse, yerr=iterative_error, fmt="o", color="black")
plt.errorbar(br3, y=dispatch_rmse, yerr=dispatch_error, fmt="o", color="black")
# plt.xticks([r + barWidth / 2. for r in range(len(single_rmse))], datasets)
plt.xticks([])
plt.ylabel("Seconds (log)", fontsize=35)
plt.tight_layout()
plt.yscale("log")
plt.legend()
# plt.legend(prop={'size': 35}, mode=None)
plt.show()
