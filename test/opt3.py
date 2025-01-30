import torch
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def styblinski_tang(x):
    return 0.5 * torch.sum(x ** 4 - 16 * x ** 2 + 5 * x, dim=-1)


def generate_initial_points(n_initial, dim, bounds):
    lower, upper = bounds
    return torch.rand(n_initial, dim) * (upper - lower) + lower


def create_model(train_X, train_Y):
    kernel = ScaleKernel(RBFKernel(ard_num_dims=train_X.shape[-1]))
    model = SingleTaskGP(train_X, train_Y, covar_module=kernel)
    return model


def compute_kernel_complexity_metrics(model, X):
    with torch.no_grad():
        K = model.covar_module(X).evaluate().cpu().numpy()

    I = np.eye(K.shape[0])
    frobenius_norm = np.linalg.norm(K - I, 'fro')
    condition_number = np.linalg.cond(K)
    eigenvalues = np.linalg.eigvalsh(K)
    trace_norm = np.sum(np.abs(eigenvalues))
    normalized_eigenvalues = eigenvalues / np.sum(eigenvalues)
    entropy = -np.sum(normalized_eigenvalues * np.log(normalized_eigenvalues + 1e-10))

    return {
        'frobenius_norm': frobenius_norm,
        'condition_number': condition_number,
        'eigenvalue_range': (np.min(eigenvalues), np.max(eigenvalues)),
        'trace_norm': trace_norm,
        'entropy': entropy
    }


class BasicBO:
    def __init__(self, dim, bounds, n_initial, obj_function):
        self.dim = dim
        self.bounds = bounds
        self.obj_function = obj_function
        self.X = generate_initial_points(n_initial, dim, bounds).double()
        self.Y = self.obj_function(self.X).double().unsqueeze(-1)
        self.best_f = self.Y.min().item()
        self.best_x = self.X[self.Y.argmin()]
        self.eval_history = [self.best_f] * n_initial
        self.kernel_complexity_history = []

    def optimize(self, n_iter):
        for _ in range(n_iter):
            model = create_model(self.X, self.Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            metrics = compute_kernel_complexity_metrics(model, self.X)
            self.kernel_complexity_history.append(metrics)

            EI = ExpectedImprovement(model, best_f=self.best_f, maximize=False)

            candidate, _ = optimize_acqf(
                EI, bounds=self.bounds, q=1, num_restarts=10, raw_samples=100,
            )

            x_new = candidate.squeeze()
            y_new = self.obj_function(x_new.unsqueeze(0)).double().unsqueeze(-1)

            self.X = torch.cat([self.X, x_new.unsqueeze(0)])
            self.Y = torch.cat([self.Y, y_new])

            if y_new < self.best_f:
                self.best_f = y_new.item()
                self.best_x = x_new

            self.eval_history.append(self.best_f)

        return self.best_x, self.best_f


def run_optimization_for_dimensions(dims, bounds, n_initial, n_iter, obj_function):
    results = []
    for dim in dims:
        print(f"Running optimization for dimension {dim}")
        bounds_dim = torch.stack([torch.tensor([bounds[0]] * dim), torch.tensor([bounds[1]] * dim)]).double()
        optimizer = BasicBO(dim, bounds_dim, n_initial, obj_function)
        best_x, best_f = optimizer.optimize(n_iter)
        results.append({
            'dim': dim,
            'best_f': best_f,
            'history': optimizer.kernel_complexity_history
        })
    return results


def plot_multi_dim_kernel_complexity(results):
    metrics = ['frobenius_norm', 'condition_number', 'trace_norm', 'entropy']
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Kernel Complexity Metrics Over Iterations for Different Dimensions')

    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        for result in results:
            dim = result['dim']
            values = [m[metric] for m in result['history']]
            sns.lineplot(x=range(len(values)), y=values, ax=ax, label=f'Dim {dim}')
        ax.set_title(metric.replace('_', ' ').capitalize())
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_multi_dim_eigenvalue_range(results):
    fig, ax = plt.subplots(figsize=(15, 10))
    for result in results:
        dim = result['dim']
        min_eigenvalues = [m['eigenvalue_range'][0] for m in result['history']]
        max_eigenvalues = [m['eigenvalue_range'][1] for m in result['history']]
        ax.fill_between(range(len(min_eigenvalues)), min_eigenvalues, max_eigenvalues, alpha=0.3, label=f'Dim {dim}')
        ax.plot(range(len(min_eigenvalues)), min_eigenvalues, label=f'Min Eigenvalue (Dim {dim})')
        ax.plot(range(len(max_eigenvalues)), max_eigenvalues, label=f'Max Eigenvalue (Dim {dim})')
    ax.set_title('Eigenvalue Range Over Iterations for Different Dimensions')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Eigenvalue')
    ax.legend()
    plt.show()


# メイン実行部分
dims = [5, 10, 15, 20]
bounds = [-5.0, 5.0]
n_initial = 20
n_iter = 50

results = run_optimization_for_dimensions(dims, bounds, n_initial, n_iter, styblinski_tang)

# グラフの表示
plot_multi_dim_kernel_complexity(results)
plot_multi_dim_eigenvalue_range(results)