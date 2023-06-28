import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from linops.experiment_utils import load_object

input_path = "./logs/rff_solve_20230509_1631.pkl"
# input_path = "./logs/rff_solve.pkl"
results = load_object(input_path)
keys = ["svrg", "iterative"]
an = {
    "iterative": {
        "color": "#de2d26",
        "label": "Iterative"
    },
    "svrg": {
        "color": "#2b8cbe",
        "label": "CoLA (SVRG)"
    }
}
mult = {"svrg": 2., "iterative": 10.}

sns.set(style="whitegrid", font_scale=4.0)
plt.figure(dpi=50, figsize=(14, 10))
for key in keys:
    errors = results[key]["errors"]
    its = (np.arange(len(errors))) * mult[key] + 1.0
    plt.plot(its, errors, label=an[key]["label"], c=an[key]["color"], lw=6)
plt.xlabel("MVMs ($10^2$)")
plt.ylabel("Residual")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("rff_solve.pdf")
plt.show()
