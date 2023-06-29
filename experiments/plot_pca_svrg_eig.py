import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from cola.experiment_utils import load_object

input_path = "./logs/pca_svrg_eig_20230509_1454.pkl"
results = load_object(input_path)
keys = ["svrg", "power"]
an = {
    "power": {
        "color": "#de2d26",
        "label": "Iterative"
    },
    "svrg": {
        "color": "#2b8cbe",
        "label": "CoLA (SVRG)"
    }
}
mult = {"svrg": 2., "power": 43.}

sns.set(style="whitegrid", font_scale=4.0)
plt.figure(dpi=50, figsize=(14, 10))
for key in keys:
    errors = results[key]
    its = (np.arange(len(errors))) * mult[key] + 1.0
    plt.plot(its, errors, label=an[key]["label"], c=an[key]["color"], lw=6)
plt.xlabel("MVMs ($10^4$)")
plt.ylabel("Convergence Criteria")
plt.yscale("log")
plt.ylim([1e-6, 1e+2])
plt.legend()
plt.tight_layout()
plt.savefig("pca_svrg_eig.pdf")
plt.show()
