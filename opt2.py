import torch
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
import matplotlib.pyplot as plt


def styblinski_tang(x):
    return 0.5 * torch.sum(x ** 4 - 16 * x ** 2 + 5 * x, dim=-1)


global_optimum = -39.16599 * 25


class DropoutMixBO:
    def __init__(self, dim, active_dim, bounds, dropout_prob=0.1):
        self.dim = dim
        self.active_dim = active_dim
        self.bounds = bounds
        self.dropout_prob = dropout_prob
        self.X = torch.Tensor([])
        self.Y = torch.Tensor([])
        self.best_f = None
        self.best_x = None
        self.eval_history = []

    def optimize(self, n_iter):
        for _ in range(n_iter):
            active_dims = np.random.choice(self.dim, self.active_dim, replace=False)

            if len(self.X) > 0:
                train_X = self.X[:, active_dims]
                train_Y = self.Y.unsqueeze(-1)
                model = SingleTaskGP(train_X, train_Y)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_model(mll)

                EI = ExpectedImprovement(model, best_f=train_Y.max())
                bounds_active = torch.stack([self.bounds[0][active_dims], self.bounds[1][active_dims]])
                candidate, _ = optimize_acqf(
                    EI, bounds=bounds_active, q=1, num_restarts=5, raw_samples=20,
                )

                x_new = torch.zeros(self.dim)
                if np.random.random() < self.dropout_prob:
                    x_new = torch.rand(self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
                else:
                    x_new[active_dims] = candidate.squeeze()
                    if self.best_x is not None:
                        x_new[np.setdiff1d(range(self.dim), active_dims)] = self.best_x[
                            np.setdiff1d(range(self.dim), active_dims)]
                    else:
                        x_new[np.setdiff1d(range(self.dim), active_dims)] = torch.rand(self.dim - self.active_dim) * (
                                    self.bounds[1] - self.bounds[0]) + self.bounds[0]
            else:
                x_new = torch.rand(self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

            y_new = styblinski_tang(x_new.unsqueeze(0))

            self.X = torch.cat([self.X, x_new.unsqueeze(0)])
            self.Y = torch.cat([self.Y, y_new])

            if self.best_f is None or y_new < self.best_f:
                self.best_f = y_new
                self.best_x = x_new

            self.eval_history.append(self.best_f.item())

        return self.best_x, self.best_f


class REMBO:
    def __init__(self, high_dim, low_dim, bounds):
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.bounds = bounds
        self.A = torch.randn(high_dim, low_dim)
        self.X = torch.Tensor([])
        self.Y = torch.Tensor([])
        self.best_f = None
        self.best_x = None
        self.eval_history = []

    def optimize(self, n_iter):
        for _ in range(n_iter):
            if len(self.X) > 0:
                train_X = self.X
                train_Y = self.Y.unsqueeze(-1)
                model = SingleTaskGP(train_X, train_Y)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_model(mll)

                EI = ExpectedImprovement(model, best_f=train_Y.max())
                bounds_low = torch.stack([torch.ones(self.low_dim) * -1, torch.ones(self.low_dim)])
                candidate, _ = optimize_acqf(
                    EI, bounds=bounds_low, q=1, num_restarts=5, raw_samples=20,
                )

                x_low = candidate.squeeze()
            else:
                x_low = torch.rand(self.low_dim) * 2 - 1

            x_high = torch.matmul(self.A, x_low)
            x_high = torch.clamp(x_high, self.bounds[0], self.bounds[1])

            y_new = styblinski_tang(x_high.unsqueeze(0))

            self.X = torch.cat([self.X, x_low.unsqueeze(0)])
            self.Y = torch.cat([self.Y, y_new])

            if self.best_f is None or y_new < self.best_f:
                self.best_f = y_new
                self.best_x = x_high

            self.eval_history.append(self.best_f.item())

        return self.best_x, self.best_f


class LINEBO:
    def __init__(self, dim, bounds):
        self.dim = dim
        self.bounds = bounds
        self.X = torch.Tensor([])
        self.Y = torch.Tensor([])
        self.best_f = None
        self.best_x = None
        self.eval_history = []

    def optimize(self, n_iter):
        for _ in range(n_iter):
            if len(self.X) > 0:
                train_X = self.X
                train_Y = self.Y.unsqueeze(-1)
                model = SingleTaskGP(train_X, train_Y)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_model(mll)

                EI = ExpectedImprovement(model, best_f=train_Y.max())
                candidate, _ = optimize_acqf(
                    EI, bounds=self.bounds, q=1, num_restarts=5, raw_samples=20,
                )

                x_new = candidate.squeeze()
            else:
                x_new = torch.rand(self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

            y_new = styblinski_tang(x_new.unsqueeze(0))

            self.X = torch.cat([self.X, x_new.unsqueeze(0)])
            self.Y = torch.cat([self.Y, y_new])

            if self.best_f is None or y_new < self.best_f:
                self.best_f = y_new
                self.best_x = x_new

            self.eval_history.append(self.best_f.item())

        return self.best_x, self.best_f


# 最適化の実行
dim = 25
active_dim = 5
bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
n_iter = 100

dropout_bo = DropoutMixBO(dim, active_dim, bounds)
rembo = REMBO(dim, active_dim, bounds)
linebo = LINEBO(dim, bounds)

dropout_best_x, dropout_best_f = dropout_bo.optimize(n_iter)
rembo_best_x, rembo_best_f = rembo.optimize(n_iter)
linebo_best_x, linebo_best_f = linebo.optimize(n_iter)

# 結果のプロット
plt.figure(figsize=(12, 8))
plt.plot(range(1, n_iter + 1), dropout_bo.eval_history, label='Dropout-Mix BO')
plt.plot(range(1, n_iter + 1), rembo.eval_history, label='REMBO')
plt.plot(range(1, n_iter + 1), linebo.eval_history, label='LINEBO')
plt.axhline(y=global_optimum, color='r', linestyle='--', label='Global Optimum')
plt.xlabel('Iteration')
plt.ylabel('Evaluation Value')
plt.title('Comparison of Optimization Algorithms')
plt.legend()
plt.yscale('symlog')
plt.grid(True)
plt.show()

print(f"Dropout-Mix BO best value: {dropout_best_f.item()}")
print(f"REMBO best value: {rembo_best_f.item()}")
print(f"LINEBO best value: {linebo_best_f.item()}")