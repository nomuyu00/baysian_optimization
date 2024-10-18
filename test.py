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


class DropoutMixBO_BC:
    def __init__(self, dim, active_dim, bounds, n_initial, obj_function, dropout_prob=0.0, epsilon=0.1,
                 temperature=1e-3, reset_interval=1000, learning_rate=0.005, initial_beta=2.0, annealing_rate=1000):
        # クラスの初期化
        self.dim = dim
        self.active_dim = active_dim
        self.bounds = bounds
        self.dropout_prob = dropout_prob
        self.obj_function = obj_function
        self.epsilon = epsilon
        self.temperature = temperature
        self.reset_interval = reset_interval
        self.iteration = 0
        self.learning_rate = learning_rate
        self.initial_beta = initial_beta
        self.annealing_rate = annealing_rate

        # 初期点の生成と評価
        initial_X = generate_initial_points(n_initial, dim, bounds)
        initial_Y = obj_function(initial_X)

        self.X = initial_X.double()
        self.Y = initial_Y.double()

        self.best_f = self.Y.min().item()
        self.best_x = self.X[self.Y.argmin()]
        self.eval_history = [self.best_f] * n_initial
        self.improvement_history = []

        self.arm_rewards = np.zeros(dim)
        self.arm_counts = np.zeros(dim)
        self.total_pulls = 0
        self.dim_sensitivity = np.zeros(dim)

        self.arm_selection_history = []

    def select_active_dims(self):
        # 活性化する次元を選択
        self.iteration += 1

        # UCBスコアに基づいて選択
        ucb_scores = self.calculate_ucb_scores()

        # ソフトマックスの適用
        probabilities = softmax(ucb_scores / self.temperature)
        probabilities = np.nan_to_num(probabilities, nan=1.0 / self.dim, posinf=1.0, neginf=0.0)
        probabilities = np.clip(probabilities, 1e-10, 1)
        probabilities /= probabilities.sum()

        selected_arms = np.random.choice(self.dim, self.active_dim, replace=False, p=probabilities)

        # 選択された次元を記録
        arm_selection = np.zeros(self.dim)
        arm_selection[selected_arms] = 1
        self.arm_selection_history.append(arm_selection)

        return selected_arms

    def calculate_ucb_scores(self):
        # UCBスコアを計算
        exploration_term = np.sqrt(2 * np.log(self.total_pulls + 1) / (self.arm_counts + 1e-5))
        exploitation_term = self.arm_rewards / (self.arm_counts + 1e-5)

        # アニーリングによるβの調整
        beta = self.initial_beta * np.exp(-self.iteration / self.annealing_rate)

        ucb_scores = exploitation_term + beta * exploration_term
        return ucb_scores

    def calculate_dimension_sensitivity(self, new_x, new_y):
        # new_y を Tensor に変換
        if not isinstance(new_y, torch.Tensor):
            new_y = torch.tensor([new_y], dtype=torch.double)
        # 新しいデータポイントを追加
        X_new = torch.cat([self.X, new_x.unsqueeze(0)], dim=0)
        Y_new = torch.cat([self.Y, new_y])

        self.X = X_new
        self.Y = Y_new

        # NumPy配列に変換
        X_np = self.X.cpu().numpy()
        Y_np = self.Y.cpu().numpy()

        sensitivities = np.zeros(self.dim)
        for i in range(self.dim):
            sorted_indices = np.argsort(X_np[:, i])
            sorted_x = X_np[sorted_indices, i]
            sorted_y = Y_np[sorted_indices]
            dx = np.diff(sorted_x)
            dy = np.diff(sorted_y)
            nonzero_dx = dx != 0
            diffs = np.zeros_like(dx)
            diffs[nonzero_dx] = dy[nonzero_dx] / dx[nonzero_dx]
            sensitivities[i] = np.mean(np.abs(diffs))

        total_sensitivity = np.sum(sensitivities) + 1e-10
        new_sensitivity = sensitivities / total_sensitivity

        # 指数移動平均で感度を更新
        alpha = 0.1
        self.dim_sensitivity = alpha * new_sensitivity + (1 - alpha) * self.dim_sensitivity

    def update_bandit(self, selected_dims, y_new, y_pred):
        # バンディットの更新
        improvement = np.exp(-((y_pred - y_new) ** 2))
        self.improvement_history.append(improvement)

        self.total_pulls += 1
        for dim in selected_dims:
            self.arm_counts[dim] += 1
            arm_contribution = improvement * self.dim_sensitivity[dim] / (
                    sum(self.dim_sensitivity[selected_dims]) + 1e-10)
            self.arm_rewards[dim] += arm_contribution

    def optimize(self, n_iter):
        # 最適化のメインループ
        for _ in range(n_iter):
            # 活性化する次元を選択
            active_dims = self.select_active_dims()

            # モデルの学習データを準備
            train_X = self.X[:, active_dims]
            train_Y = self.Y.unsqueeze(-1)

            # ガウス過程モデルの作成とフィッティング
            model = create_model(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            # Expected Improvement (EI) 獲得関数の定義
            EI = ExpectedImprovement(model, best_f=self.best_f, maximize=False)
            bounds_active = torch.stack([self.bounds[0][active_dims], self.bounds[1][active_dims]]).double()

            # 獲得関数の最適化
            candidate, _ = optimize_acqf(
                EI, bounds=bounds_active, q=1, num_restarts=10, raw_samples=100,
                options={"maxiter": 200, "batch_limit": 5}
            )

            # 新しい候補点を構築
            x_new = torch.zeros(self.dim, dtype=torch.double)
            x_new[active_dims] = candidate.squeeze()
            x_new[np.setdiff1d(range(self.dim), active_dims)] = self.best_x[
                np.setdiff1d(range(self.dim), active_dims)]

            # ガウス過程モデルによる予測値の計算
            with torch.no_grad():
                y_pred = model(x_new[active_dims].unsqueeze(0)).mean.item()

            # 目的関数の評価
            y_new = self.obj_function(x_new.unsqueeze(0))
            if isinstance(y_new, torch.Tensor):
                y_new = y_new.item()

            # 感度の更新
            self.calculate_dimension_sensitivity(x_new, y_new)

            # バンディットの更新
            self.update_bandit(active_dims, y_new, y_pred)

            # 最良値の更新
            if y_new < self.best_f:  # item() を取り除く
                self.best_f = y_new
                self.best_x = x_new

            # 評価履歴の更新
            self.eval_history.append(self.best_f)

        # 次元選択の履歴をDataFrameに変換
        self.arm_selection_df = pd.DataFrame(self.arm_selection_history,
                                             columns=[f'Arm_{i}' for i in range(self.dim)])
        self.arm_selection_df.index.name = 'Iteration'

        return self.best_x, self.best_f

    # 結果の保存と可視化
    def save_arm_selection_history(self, filename):
        self.arm_selection_df.to_csv(filename)

    def plot_dim_sensitivity(self):
        plt.figure(figsize=(12, 6))
        plt.bar(range(self.dim), self.dim_sensitivity)
        plt.xlabel('Dimension')
        plt.ylabel('Sensitivity')
        plt.title('Dimension Sensitivity')
        plt.xticks(range(self.dim))
        plt.grid(True)
        plt.show()


# パラメータの設定
dim = 10
active_dim = 5
bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
n_initial = 20
n_iter = 50

# インスタンスの作成と最適化の実行
dropout_bandit_bc_ucb = DropoutMixBO_BC(dim, active_dim, bounds, n_initial, styblinski_tang, dropout_prob=0.0)

dropout_bandit_bc_ucb_best_x, dropout_bandit_bc_ucb_best_f = dropout_bandit_bc_ucb.optimize(n_iter)
dropout_bandit_bc_ucb.save_arm_selection_history('dropout_bandit_bc_ucb_arm_selection_binary.csv')
