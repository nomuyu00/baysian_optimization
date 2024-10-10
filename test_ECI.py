import torch
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
from botorch.acquisition import ExpectedImprovement
from botorch.utils.transforms import standardize, normalize
import math
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.double)

def styblinski_tang(x):
    indices = [0, 1, 2, 3, 4]
    x_selected = x[..., indices]
    return 0.5 * torch.sum(x_selected ** 4 - 16 * x_selected ** 2 + 5 * x_selected, dim=-1)

global_optimum = -39.16599 * 5

def generate_initial_points(n_initial, dim, bounds):
    return torch.rand(n_initial, dim) * (bounds[1] - bounds[0]) + bounds[0]

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel
import math
import numpy as np


class ECI_BO_Bandit:
    def __init__(self, X, objective_function, bounds, n_initial, n_max, dim):
        self.objective_function = objective_function
        self.bounds = bounds  # Should be a list of tensors: [lower_bounds, upper_bounds]
        self.n_initial = n_initial
        self.n_max = n_max
        self.dim = dim
        self.X = X
        self.Y = None
        self.best_value = None
        self.best_point = None
        self.model = None

        # Bandit algorithm parameters
        self.dimension_counts = [0] * self.dim  # Number of times each dimension was selected
        self.dimension_rewards = [0.0] * self.dim  # Cumulative rewards for each dimension

        self.eval_history = [self.best_value] * n_initial
        self.arm_selection_history = []

    def update_model(self):
        kernel = ScaleKernel(RBFKernel(ard_num_dims=self.X.shape[-1]), noise_constraint=1e-5)
        self.model = SingleTaskGP(self.X, self.Y, covar_module=kernel)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

    def normalize_rewards(self, rewards):
        if isinstance(rewards, (int, float)):
            return 1  # 単一の値の場合はそのまま返す
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

    def select_dimension(self, total_iterations):
        # UCB calculation
        ucb_values = []
        for i in range(self.dim):
            if self.dimension_counts[i] == 0:
                # Ensure each dimension is selected at least once
                return i
            average_reward = self.dimension_rewards[i] / self.dimension_counts[i]
            confidence = math.sqrt(2 * math.log(total_iterations) / self.dimension_counts[i])
            ucb = average_reward + confidence
            ucb_values.append(ucb)
        # Select dimension with highest UCB
        return ucb_values.index(max(ucb_values))

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

            # Select a dimension to optimize using the bandit algorithm
            selected_dim = self.select_dimension(total_iterations)
            self.arm_selection_history.append(selected_dim)

            # Optimize over the selected dimension
            ei = ExpectedImprovement(self.model, best_f=self.best_value, maximize=False)

            def eci_func(x):
                batch_size = x.shape[0]
                full_x = self.best_point.clone().unsqueeze(0).expand(batch_size, -1).clone()
                full_x[:, selected_dim] = x.view(-1)
                full_x = full_x.unsqueeze(1)
                return ei(full_x)

            # Bounds for the selected dimension
            bounds = torch.tensor([
                [self.bounds[0][selected_dim]],
                [self.bounds[1][selected_dim]]
            ], device=self.X.device)

            # Optimize the acquisition function over the selected dimension
            candidate, acq_value = optimize_acqf(
                eci_func,
                bounds,
                q=1,
                num_restarts=10,
                raw_samples=100,
            )

            # Construct the new point
            new_x = self.best_point.clone()
            new_x[selected_dim] = candidate.squeeze()
            new_y = self.objective_function(new_x.unsqueeze(0)).unsqueeze(-1)

            # Update data
            self.X = torch.cat([self.X, new_x.unsqueeze(0)])
            self.Y = torch.cat([self.Y, new_y])

            improvement = max(0, self.best_value - new_y.item())

            # Calculate reward (improvement)

            self.dimension_rewards = torch.tensor(self.dimension_rewards)
            self.dimension_rewards[selected_dim] += self.normalize_rewards(improvement)

            # Update best value and point if improvement is found
            if new_y.item() < self.best_value:
                self.best_value = new_y.item()
                self.best_point = new_x

            self.eval_history.append(self.best_value)

            # Update bandit statistics
            self.dimension_counts = torch.tensor(self.dimension_counts)
            self.dimension_counts[selected_dim] += 1

            n += 1
            total_iterations += 1
            print(n)

        return self.best_point, self.best_value


# Parameters
dim = 10
bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
n_initial = 200
n_iter = 210
n_runs = 1

X = generate_initial_points(n_initial, dim, bounds)

# 複数回の実行結果を格納するリスト
all_histories = []

# Run the optimization

for run in range(n_runs):
    eci_bo = ECI_BO_Bandit(X, styblinski_tang, bounds, n_initial, n_iter, dim)
    best_x, best_f = eci_bo.optimize()
    all_histories.append(eci_bo.eval_history)


# 全実行の平均を計算
average_history = np.mean(all_histories, axis=0)

plt.figure(figsize=(12, 8))
plt.plot(eci_bo.eval_history, label="ECI_BO")
#plt.plot(full_bandit.eval_history, label="fullBandit")
plt.axhline(global_optimum, color="black", linestyle="--", label="Global optimum")
plt.xlabel("Iteration")
plt.ylabel("Best value")
plt.title("Optimization results")
plt.legend()
plt.grid(True)
plt.show()


# 追加：全ての実行結果を薄い色でプロット
plt.figure(figsize=(12, 8))
for history in all_histories:
    plt.plot(history, alpha=0.2, color='blue')
plt.plot(average_history, label="Average ECI-BO", color='red', linewidth=2)
plt.axhline(global_optimum, color="black", linestyle="--", label="Global optimum")
plt.xlabel("Iteration")
plt.ylabel("Best Value")
plt.title(f"All Optimization Results ({n_runs} runs)")
plt.legend()
plt.grid(True)
plt.show()

print(eci_bo.arm_selection_history)
