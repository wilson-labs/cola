import time
import torch
import gpytorch
from scipy.io import loadmat
from math import floor
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.kernels import AdditiveStructureKernel, GridInterpolationKernel
from gpytorch.distributions import MultivariateNormal

# file_path = '/datasets/uci/elevators/elevators.mat'
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

if torch.cuda.is_available():
    train_x, train_y = train_x.cuda(), train_y.cuda()
    test_x, test_y = test_x.cuda(), test_y.cuda()


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, num_proj=16, grid_size=128):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.register_parameter("proj_mat",
                                torch.nn.Parameter(torch.randn(train_x.size(-1), num_proj)))

        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            AdditiveStructureKernel(
                GridInterpolationKernel(RBFKernel(), grid_size=grid_size, num_dims=1),
                num_dims=num_proj))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x @ self.proj_mat)
        return MultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y)

if torch.cuda.is_available():
    model = model.cuda()

training_iterations = 50
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

# iterator = tqdm.tqdm(range(training_iterations), desc="Training")
tic = time.time()
iterator = range(training_iterations)
for i in iterator:
    optimizer.zero_grad()
    # For some reason I'm getting OOM erros with preconditioning
    with gpytorch.settings.max_preconditioner_size(0):
        t0 = time.time()
        output = model(train_x)
        loss = -mll(output, train_y)
        lo = output.lazy_covariance_matrix.evaluate_kernel()
        loss.backward()
        # iterator.set_postfix(loss=loss.item())
        print(f"Loss: {loss.item():1.5e}")
        optimizer.step()
        torch.cuda.empty_cache()
        t1 = time.time()
        print(f"Time: {t1 - t0:4.3f} sec")
toc = time.time()
print(f"Time: {toc - tic:4.3f} sec")

model.eval()
likelihood.eval()
with gpytorch.settings.max_preconditioner_size(
        0), torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds = model(test_x)

print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - test_y))))
