import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from cola.experiment_utils import load_object


input_path = "./logs/nivp_solve_20230516_1934.pkl"
results = load_object(input_path)
keys = ["svrg", "cg"]
an = {
    "cg": {
        "color": "#de2d26",
        "label": "Iterative"
    },
    "svrg": {
        "color": "#2b8cbe",
        "label": "CoLA (SVRG)"
    }
}
mult = {"svrg": 2., "cg": 1.}

sns.set(style="whitegrid", font_scale=4.0)
plt.figure(dpi=50, figsize=(14, 10))
for key in keys:
    errors = np.array(results[key])
    its = (np.arange(len(errors))) * mult[key] + 1.0
    plt.plot(its, errors, label=an[key]["label"], c=an[key]["color"], lw=6)
plt.xlabel("MVMs")
plt.ylabel("Residual")
plt.yscale("log")
plt.xlim([0, 200])
plt.legend()
plt.tight_layout()
plt.savefig("nivp_solve.pdf")
plt.show()
