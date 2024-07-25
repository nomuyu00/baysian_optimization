import torch
from botorch import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
import numpy as np
import matplotlib.pyplot as plt


def rosenbrock(x):
    return torch.sum(100.0 * (x[..., 1:] - x[..., :-1]**2)**2 + (1 - x[..., :-1])**2, dim=-1)

# rosenbrockを使う場合
global_optimum = 0.0


def generate_initial_points(n_initial, dim, bounds):
    return torch.rand(n_initial, dim) * (bounds[1] - bounds[0]) + bounds[0]


def create_model(train_X, train_Y):
    kernel = ScaleKernel(RBFKernel(ard_num_dims=train_X.shape[-1]))
    model = SingleTaskGP(train_X, train_Y, covar_module=kernel)
    return model


class DropoutMixBO_Bandit:
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

        # バンディットアルゴリズム用の変数
        self.dim_counts = np.zeros(dim)
        self.dim_rewards = np.zeros(dim)
        self.total_pulls = 0

    def select_active_dims(self):
        ucb_scores = self.dim_rewards / (self.dim_counts + 1e-5) + np.sqrt(
            2 * np.log(self.total_pulls + 1) / (self.dim_counts + 1e-5))
        return np.argsort(ucb_scores)[-self.active_dim:]

    def update_bandit(self, selected_dims, reward):
        self.dim_counts[selected_dims] += 1
        self.dim_rewards[selected_dims] += reward
        self.total_pulls += 1

    def optimize(self, n_iter):
        for _ in range(n_iter):
            active_dims = self.select_active_dims()

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

            improvement = max(self.best_f - y_new.item(), 0)
            self.update_bandit(active_dims, improvement)

            if y_new < self.best_f:
                self.best_f = y_new.item()
                self.best_x = x_new

            self.eval_history.append(self.best_f)

        return self.best_x, self.best_f



dim = 25
active_dim = 5
bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
n_initial = 200
n_iter = 100

dropout_bandit = DropoutMixBO_Bandit(dim, active_dim, bounds, n_initial)

dropout_bandit_best_x, dropout_bandit_best_f = dropout_bandit.optimize(n_iter)


plt.figure(figsize=(12, 8))
plt.plot(range(1, n_initial + n_iter + 1), dropout_bandit.eval_history, label='Dropout-Bandit BO')
plt.axhline(y=global_optimum, color='r', linestyle='--', label='Global Optimum')
plt.xlabel('Iteration')
plt.ylabel('Best Function Value')
plt.title('Comparison of Optimization Algorithms　for Rosenbrock Function')
plt.legend()
plt.yscale('symlog')
plt.grid(True)
plt.show()