import time
import torch
from scipy.io import loadmat
from math import floor
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RFFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

tic = time.time()
torch.manual_seed(seed=21)
training_iter = 10
# rff_num = 10
# rff_num = 1_000
rff_num = 5_000

file_path = '/home/ubu/Downloads/elevators.mat'

data = torch.Tensor(loadmat(file_path)['data'])
X = data[:, :-1]
X = X - X.min(0)[0]
X = 2 * (X / X.max(0)[0]) - 1
y = data[:, -1]

train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()

test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RFFKernel(num_samples=rff_num))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


likelihood = GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    text = f"Iter {i + 1:3d}/{training_iter:d} - Loss: {loss.item():.3f}    "
    text += f"lengthscale: {model.covar_module.base_kernel.lengthscale.item():.3f}    "
    text += f"noise: {model.likelihood.noise.item():.3f}    "
    print(text)
    optimizer.step()

toc = time.time()
print(f"Experiment took: {toc - tic:.4f} sec")

model.eval()
likelihood.eval()

with torch.no_grad():
    preds = model(test_x)
    # preds = likelihood(model(test_x))
print(f"Test MAE: {torch.mean(torch.abs(preds.mean - test_y)):1.5e}")
