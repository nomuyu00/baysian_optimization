import random
import torch
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch import constraints
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
import numpy as np
import pandas as pd
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ランダムシードの設定
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def styblinski_tang(x):
    indices = [2, 3, 5, 7, 9]
    x_selected = x[..., indices]
    return 0.5 * torch.sum(x_selected ** 4 - 16 * x_selected ** 2 + 5 * x_selected, dim=-1)


# Styblinski-Tang関数の最適値
global_optimum = -39.16599 * 5


def generate_initial_points(n_initial, dim, bounds):
    return torch.rand(n_initial, dim) * (bounds[1] - bounds[0]) + bounds[0]


def create_model(train_X, train_Y):
    kernel = ScaleKernel(RBFKernel(ard_num_dims=train_X.shape[-1]))
    model = SingleTaskGP(train_X, train_Y, covar_module=kernel)
    # ノイズ制約の設定
    model.likelihood.noise_covar.register_constraint("raw_noise", constraints.GreaterThan(1e-5))
    return model


class ECI_BO_Bandit:
    def __init__(self, X, objective_function, bounds, n_initial, n_max, dim, active_dim, gamma=0.99):
        self.objective_function = objective_function
        self.bounds = bounds  # Should be a list of tensors: [lower_bounds, upper_bounds]
        self.n_initial = n_initial
        self.n_max = n_max
        self.dim = dim
        self.active_dim = active_dim
        self.X = X
        self.Y = None
        self.best_value = None
        self.best_point = None
        self.model = None
        self.gamma = 1

        # Bandit algorithm parameters
        self.dimension_counts = [1] * self.dim  # Number of times each dimension was selected
        self.dimension_rewards = [0.0] * self.dim  # Cumulative rewards for each dimension
        self.squared_reward = [0.0] * self.dim  # Cumulative squared rewards for each dimension

        self.eval_history = [self.best_value] * n_initial
        self.arm_selection_history = []

    def update_model(self):
        kernel = ScaleKernel(RBFKernel(ard_num_dims=self.X.shape[-1]), noise_constraint=1e-5)
        self.model = SingleTaskGP(self.X, self.Y, covar_module=kernel)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

    def normalize_rewards(self, rewards):
        if isinstance(rewards, (int, float)):
            return rewards  # 単一の値の場合はそのまま返す
        min_reward = min(rewards)
        max_reward = max(rewards)
        if min_reward == max_reward:
            return [1.0] * len(rewards)
        return [(r - min_reward) / (max_reward - min_reward) for r in rewards]

    def initialize(self):
        self.Y = self.objective_function(self.X).unsqueeze(-1)
        self.best_value = self.Y.min().item()
        self.best_point = self.X[self.Y.argmin()]
        self.update_model()
        self.eval_history = [self.best_value] * self.n_initial

        # Calculate initial ECI values and normalize them
        eci_values = self.calculate_eci()
        self.dimension_rewards = self.normalize_rewards(eci_values)
        self.squared_reward = [r ** 2 for r in self.dimension_rewards]

    def select_dimension(self, total_iterations):
        # UCB calculation
        ucb_values = []
        for i in range(self.dim):
            if self.dimension_counts[i] == 0:
                ucb_values.append(float('inf'))  # Ensure unselected dimensions are chosen first
            else:
                average_reward = self.dimension_rewards[i] / self.dimension_counts[i]
                var = max(0, self.squared_reward[i] / self.dimension_counts[i] - average_reward ** 2)
                confidence = math.sqrt(
                    2 * var * math.log(total_iterations + 1) / self.dimension_counts[i]) + 3 * math.log(
                    total_iterations + 1) / self.dimension_counts[i]
                ucb = average_reward + confidence
                ucb_values.append(ucb)

        # Sort dimensions by UCB values
        sorted_indices = sorted(range(self.dim), key=lambda k: ucb_values[k], reverse=True)

        # Select dimensions based on UCB values
        selected_dims = []
        while len(selected_dims) < self.active_dim:
            remaining = self.active_dim - len(selected_dims)
            selected_dims.extend(sorted_indices[:remaining])
            sorted_indices = sorted_indices[remaining:] + sorted_indices[:remaining]

        return selected_dims

    def calculate_eci(self):
        eci_values = []
        for i in range(self.dim):
            ei = ExpectedImprovement(self.model, best_f=self.best_value, maximize=False)

            def eci_func(x):
                # x has shape [batch_size, q=1, d=1]
                batch_size = x.shape[0]
                full_x = self.best_point.clone().unsqueeze(0).expand(batch_size, -1).clone()
                full_x[:, i] = x.view(-1)
                full_x = full_x.unsqueeze(1)  # shape [batch_size, q=1, dim]
                return ei(full_x)

            bound = torch.tensor([[self.bounds[0][i]], [self.bounds[1][i]]], device=self.X.device)
            candidate, value = optimize_acqf(
                eci_func, bound, q=1, num_restarts=10, raw_samples=100,
            )
            eci_values.append(value.item())
        return eci_values

    def optimize(self):
        self.initialize()
        n = self.n_initial
        total_iterations = 1  # For UCB calculation

        while n < self.n_max:
            self.update_model()
            print(f"Iteration {n}, reward: {self.dimension_rewards}")

            # Select dimensions to optimize using the bandit algorithm
            selected_dims = self.select_dimension(total_iterations)
            self.arm_selection_history.append(selected_dims)
            print(f"Selected dimensions: {selected_dims}")

            # Optimize over the selected dimensions
            ei = ExpectedImprovement(self.model, best_f=self.best_value, maximize=False)

            def eci_func(x):
                batch_size = x.shape[0]
                full_x = self.best_point.clone().unsqueeze(0).expand(batch_size, -1).clone()
                full_x[:, selected_dims] = x.view(batch_size, -1)
                full_x = full_x.unsqueeze(1)
                return ei(full_x)

            # Bounds for the selected dimensions
            bounds = torch.stack([
                torch.tensor([self.bounds[0][i] for i in selected_dims]),
                torch.tensor([self.bounds[1][i] for i in selected_dims])
            ], dim=0).to(self.X.device)

            # Optimize the acquisition function over the selected dimensions
            candidate, acq_value = optimize_acqf(
                eci_func,
                bounds,
                q=1,
                num_restarts=10,
                raw_samples=100,
            )

            # Construct the new point
            new_x = self.best_point.clone()
            new_x[selected_dims] = candidate.squeeze()
            new_y = self.objective_function(new_x.unsqueeze(0)).unsqueeze(-1)

            # Update data
            self.X = torch.cat([self.X, new_x.unsqueeze(0)])
            self.Y = torch.cat([self.Y, new_y])

            improvement = max(0, self.best_value - new_y.item())

            # Update rewards and counts for selected dimensions
            for dim in selected_dims:
                self.dimension_rewards[dim] = self.gamma * self.dimension_rewards[dim] + improvement
                self.squared_reward[dim] = self.gamma * self.squared_reward[dim] + improvement ** 2
                self.dimension_counts[dim] += 1

            # Update best value and point if improvement is found
            if new_y.item() < self.best_value:
                self.best_value = new_y.item()
                self.best_point = new_x

            self.eval_history.append(self.best_value)

            n += 1
            total_iterations += 1
            print(f"Iteration {n}, Best value: {self.best_value}")

        return self.best_point, self.best_value


# パラメータの設定
dim = 12
active_dim = 5
bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
n_initial = 20
n_iter = 50

X = generate_initial_points(n_initial, dim, bounds)

eci_bo_bandit_history = []
arm_selection_history = []

# インスタンスの作成と最適化の実行
eci_bo_bandit = ECI_BO_Bandit(X, styblinski_tang, bounds, n_initial, n_iter, dim, active_dim)
best_x, best_f = eci_bo_bandit.optimize()
eci_bo_bandit_history.append(eci_bo_bandit.eval_history)
arm_selection_history.append(eci_bo_bandit.arm_selection_history)


