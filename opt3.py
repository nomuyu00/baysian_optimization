import torch
from botorch import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
import numpy as np
import matplotlib.pyplot as plt


def styblinski_tang(x):
    return 0.5 * torch.sum(x ** 4 - 16 * x ** 2 + 5 * x, dim=-1)


def rosenbrock(x):
    return torch.sum(100.0 * (x[..., 1:] - x[..., :-1]**2)**2 + (1 - x[..., :-1])**2, dim=-1)


global_optimum = 0.0


def generate_initial_points(n_initial, dim, bounds):
    return torch.rand(n_initial, dim) * (bounds[1] - bounds[0]) + bounds[0]


def create_model(train_X, train_Y):
    kernel = ScaleKernel(RBFKernel(ard_num_dims=train_X.shape[-1]))
    model = SingleTaskGP(train_X, train_Y, covar_module=kernel)
    return model


class DropoutMixBO:
    def __init__(self, dim, active_dim, bounds, n_initial, dropout_prob=0.1):
        self.dim = dim
        self.active_dim = active_dim
        self.bounds = bounds
        self.dropout_prob = dropout_prob
        self.X = generate_initial_points(n_initial, dim, bounds)
        self.Y = rosenbrock(self.X)
        self.best_f = self.Y.min().item()
        self.best_x = self.X[self.Y.argmin()]
        self.eval_history = [self.best_f] * n_initial

    def optimize(self, n_iter):
        for _ in range(n_iter):
            # 全次元からランダムにactive_dim個選ぶ
            active_dims = np.random.choice(self.dim, self.active_dim, replace=False)

            train_X = self.X[:, active_dims]
            train_Y = self.Y.unsqueeze(-1)
            model = create_model(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            UCB = UpperConfidenceBound(model, beta=0.1)
            bounds_active = torch.stack([self.bounds[0][active_dims], self.bounds[1][active_dims]])
            candidate, _ = optimize_acqf(
                UCB, bounds=bounds_active, q=1, num_restarts=5, raw_samples=20,
            )

            x_new = torch.zeros(self.dim)
            if np.random.random() < self.dropout_prob:
                x_new[active_dims] = candidate.squeeze()
                inactive_dims = np.setdiff1d(range(self.dim), active_dims)
                x_new[inactive_dims] = (torch.rand(len(inactive_dims))
                                        * (self.bounds[1][inactive_dims] - self.bounds[0][inactive_dims])
                                        + self.bounds[0][inactive_dims])
            else:
                x_new[active_dims] = candidate.squeeze()
                x_new[np.setdiff1d(range(self.dim), active_dims)] = self.best_x[
                    np.setdiff1d(range(self.dim), active_dims)]

            y_new = rosenbrock(x_new.unsqueeze(0))

            self.X = torch.cat([self.X, x_new.unsqueeze(0)])
            self.Y = torch.cat([self.Y, y_new])

            if y_new < self.best_f:
                self.best_f = y_new.item()
                self.best_x = x_new

            self.eval_history.append(self.best_f)

        return self.best_x, self.best_f


class REMBO:
    def __init__(self, high_dim, low_dim, bounds, n_initial):
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.bounds = bounds
        self.A = torch.randn(high_dim, low_dim)
        self.X = torch.randn(n_initial, low_dim) * 2 - 1
        self.X_high = torch.clamp(torch.matmul(self.X, self.A.t()), bounds[0], bounds[1])
        self.Y = rosenbrock(self.X_high)
        self.best_f = self.Y.min().item()
        self.best_x = self.X_high[self.Y.argmin()]
        self.eval_history = [self.best_f] * n_initial

    def optimize(self, n_iter):
        for _ in range(n_iter):
            train_X = self.X
            train_Y = self.Y.unsqueeze(-1)
            model = create_model(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            UCB = UpperConfidenceBound(model, beta=0.1)
            bounds_low = torch.stack([torch.ones(self.low_dim) * -1, torch.ones(self.low_dim)])
            candidate, _ = optimize_acqf(
                UCB, bounds=bounds_low, q=1, num_restarts=5, raw_samples=20,
            )

            x_low = candidate.squeeze()
            x_high = torch.clamp(torch.matmul(x_low.unsqueeze(0), self.A.t()), self.bounds[0], self.bounds[1]).squeeze()

            y_new = rosenbrock(x_high.unsqueeze(0))

            self.X = torch.cat([self.X, x_low.unsqueeze(0)])
            self.X_high = torch.cat([self.X_high, x_high.unsqueeze(0)])
            self.Y = torch.cat([self.Y, y_new])

            if y_new < self.best_f:
                self.best_f = y_new.item()
                self.best_x = x_high

            self.eval_history.append(self.best_f)

        return self.best_x, self.best_f


class LINEBO:
    def __init__(self, dim, bounds, n_initial):
        self.dim = dim
        self.bounds = bounds
        self.X = generate_initial_points(n_initial, dim, bounds)
        self.Y = rosenbrock(self.X)
        self.best_f = self.Y.min().item()
        self.best_x = self.X[self.Y.argmin()]
        self.eval_history = [self.best_f] * n_initial

    def optimize(self, n_iter):
        for _ in range(n_iter):
            train_X = self.X
            train_Y = self.Y.unsqueeze(-1)
            model = create_model(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            UCB = UpperConfidenceBound(model, beta=0.1)
            candidate, _ = optimize_acqf(
                UCB, bounds=self.bounds, q=1, num_restarts=5, raw_samples=20,
            )

            x_new = candidate.squeeze()
            y_new = rosenbrock(x_new.unsqueeze(0))

            self.X = torch.cat([self.X, x_new.unsqueeze(0)])
            self.Y = torch.cat([self.Y, y_new])

            if y_new < self.best_f:
                self.best_f = y_new.item()
                self.best_x = x_new

            self.eval_history.append(self.best_f)

        return self.best_x, self.best_f


# 最適化の実行
dim = 25
active_dim = 5
bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
n_initial = 200
n_iter = 100

dropout_bo = DropoutMixBO(dim, active_dim, bounds, n_initial)
rembo = REMBO(dim, active_dim, bounds, n_initial)
linebo = LINEBO(dim, bounds, n_initial)

dropout_best_x, dropout_best_f = dropout_bo.optimize(n_iter)
rembo_best_x, rembo_best_f = rembo.optimize(n_iter)
linebo_best_x, linebo_best_f = linebo.optimize(n_iter)

# 結果のプロット
plt.figure(figsize=(12, 8))
plt.plot(range(1, n_initial + n_iter + 1), dropout_bo.eval_history, label='Dropout-Mix BO')
plt.plot(range(1, n_initial + n_iter + 1), rembo.eval_history, label='REMBO')
plt.plot(range(1, n_initial + n_iter + 1), linebo.eval_history, label='LINEBO')
plt.axhline(y=global_optimum, color='r', linestyle='--', label='Global Optimum')
plt.xlabel('Iteration')
plt.ylabel('Best Function Value')
plt.title('Comparison of Optimization Algorithms')
plt.legend()
plt.yscale('symlog')
plt.grid(True)
plt.show()

print(f"Dropout-Mix BO best value: {dropout_best_f}")
print(f"REMBO best value: {rembo_best_f}")
print(f"LINEBO best value: {linebo_best_f}")