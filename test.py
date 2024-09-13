import torch
from botorch import fit_gpytorch_model
from botorch.acquisition import UpperConfidenceBound
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
import numpy as np
import pandas as pd
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
from botorch.models import SaasFullyBayesianSingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch import constraints


def styblinski_tang(x):
    indices = [2, 3, 5, 7, 9]
    x_selected = x[..., indices]
    return 0.5 * torch.sum(x_selected ** 4 - 16 * x_selected ** 2 + 5 * x_selected, dim=-1)

# styblinski_tang関数の最適解
global_optimum = -39.16599 * 5


def generate_initial_points(n_initial, dim, bounds):
    return torch.rand(n_initial, dim) * (bounds[1] - bounds[0]) + bounds[0]


def create_model(train_X, train_Y):
    kernel = ScaleKernel(RBFKernel(ard_num_dims=train_X.shape[-1], noise_constraint=1e-5))
    model = SingleTaskGP(train_X, train_Y, covar_module=kernel)
    return model


class DropoutMixBO_BC_UCB:
    def __init__(self, dim, active_dim, bounds, n_initial, obj_function, dropout_prob=0.1, epsilon=0.05,
                 temperature=1e-3, reset_interval=1000, learning_rate=0.005, initial_beta=0.1, annealing_rate=1000):
        # クラスの初期化。たくさんのパラメータを設定するよ
        self.dim = dim  # 全体の次元数
        self.active_dim = active_dim  # 活性化する次元数
        self.bounds = bounds  # 各次元の探索範囲
        self.dropout_prob = dropout_prob  # ドロップアウトの確率
        self.obj_function = obj_function  # 最適化したい目的関数
        self.epsilon = epsilon  # ε-greedy法のパラメータ
        self.temperature = temperature  # ソフトマックスの温度パラメータ
        self.reset_interval = reset_interval  # リセット間隔
        self.iteration = 0  # イテレーション回数
        self.learning_rate = learning_rate  # 学習率
        self.initial_beta = initial_beta  # UCBのβの初期値
        self.annealing_rate = annealing_rate  # アニーリングレート

        initial_X = self.generate_initial_points(n_initial, dim, bounds)  # 初期点を生成
        initial_Y = obj_function(initial_X)  # 初期点での目的関数値を計算

        self.X = torch.tensor(initial_X, dtype=torch.double)  # データ点
        self.Y = torch.tensor(initial_Y, dtype=torch.double)  # 目的関数値

        self.best_f = self.Y.min().item()  # 現在の最良の目的関数値
        self.best_x = self.X[self.Y.argmin()]  # 現在の最良の解
        self.eval_history = [self.best_f] * n_initial  # 評価履歴
        self.improvement_history = []  # 改善履歴

        self.arm_rewards = np.zeros(dim)  # 各次元（腕）の報酬
        self.arm_counts = np.zeros(dim)  # 各次元（腕）の選択回数
        self.total_pulls = 0  # 総選択回数
        self.dim_importance = np.ones(dim) / dim  # 各次元の重要度
        self.dim_sensitivity = np.zeros(dim)  # 各次元の感度

        self.arm_selection_history = []  # 次元選択の履歴
        self.sigma_history = []  # 予測標準偏差の履歴

    def generate_initial_points(self, n_initial, dim, bounds):
        # 初期点をランダムに生成する関数
        return torch.rand(n_initial, dim) * (bounds[1] - bounds[0]) + bounds[0]

    def select_active_dims(self):
        # 活性化する次元を選択する関数
        self.iteration += 1

        if np.random.random() < max(self.epsilon * np.exp(-self.iteration / 10000), 0.01):
            # ε-greedy法：ランダムに選択
            selected_arms = np.random.choice(self.dim, self.active_dim, replace=False)
        else:
            # UCBスコアに基づいて選択
            ucb_scores = self.calculate_ucb_scores()

            # UCBスコアの正規化
            ucb_scores = np.clip(ucb_scores, -10, 10)  # 極端な値をクリップ
            ucb_scores = ucb_scores - np.min(ucb_scores)  # 最小値を0にシフト
            max_score = np.max(ucb_scores)
            if max_score > 0:
                ucb_scores = ucb_scores / max_score  # 最大値を1に正規化

            # ソフトマックスの適用
            probabilities = softmax(ucb_scores / self.temperature)

            # NaNやinfの処理
            probabilities = np.nan_to_num(probabilities, nan=1.0 / self.dim, posinf=1.0, neginf=0.0)

            # 確率の正規化
            probabilities = np.clip(probabilities, 1e-10, 1)
            probabilities /= probabilities.sum()

            selected_arms = np.random.choice(self.dim, self.active_dim, replace=False, p=probabilities)

        # 選択された次元を記録
        arm_selection = np.zeros(self.dim)
        arm_selection[selected_arms] = 1
        self.arm_selection_history.append(arm_selection)

        # 定期的にリセット
        if self.iteration % self.reset_interval == 0:
            self.arm_rewards *= 0.5
            self.arm_counts *= 0.5

        return selected_arms

    def calculate_ucb_scores(self):
        # UCBスコアを計算する関数
        exploration_term = np.sqrt(2 * np.log(self.total_pulls + 1) / (self.arm_counts + 1e-5))
        exploitation_term = self.arm_rewards / (self.arm_counts + 1e-5)

        # アニーリングスケジュールの導入
        beta = self.initial_beta * np.exp(-self.iteration / self.annealing_rate)

        ucb_scores = exploitation_term + beta * exploration_term
        return ucb_scores * self.dim_importance

    def calculate_dimension_sensitivity(self, new_x, new_y, max_points=200):
        # new_y が2次元の場合は1次元に変換
        if new_y.dim() == 2:
            new_y = new_y.squeeze()

        # new_y が0次元（スカラー）の場合は1次元に変換
        if new_y.dim() == 0:
            new_y = new_y.unsqueeze(0)

        # 新しいデータポイントを追加
        X_new = torch.cat([self.X, new_x.unsqueeze(0)], dim=0)
        Y_new = torch.cat([self.Y, new_y])

        # Y の値が小さい順にソート（最小化問題を仮定）
        _, indices = torch.sort(Y_new)

        # 上位 max_points 個のデータポイントを保持
        if len(indices) > max_points:
            keep_indices = indices[:max_points]
            self.X = X_new[keep_indices]
            self.Y = Y_new[keep_indices]
        else:
            self.X = X_new
            self.Y = Y_new

        sensitivities = np.zeros(self.dim)
        for i in range(self.dim):
            sorted_indices = np.argsort(self.X[:, i])
            sorted_y = self.Y[sorted_indices]
            diffs = np.diff(sorted_y) / np.diff(self.X[sorted_indices, i])
            sensitivities[i] = np.mean(np.abs(diffs))

        total_sensitivity = np.sum(sensitivities) + 1e-10
        new_sensitivity = sensitivities / total_sensitivity

        # 指数移動平均を使用して感度を更新
        alpha = 0.1  # 平滑化係数
        self.dim_sensitivity = alpha * new_sensitivity + (1 - alpha) * self.dim_sensitivity

    def update_bandit(self, selected_dims, y_new):
        # バンディットアルゴリズムの更新
        improvement = max(0, self.best_f - y_new)
        relative_improvement = improvement / (abs(self.best_f) + 1e-8)

        self.total_pulls += 1
        for dim in selected_dims:
            self.arm_counts[dim] += 1
            arm_contribution = relative_improvement * self.dim_sensitivity[dim] / (
                        sum(self.dim_sensitivity[selected_dims]) + 1e-10)
            self.arm_rewards[dim] += arm_contribution

        importance_update = self.dim_sensitivity / (np.sum(self.dim_sensitivity) + 1e-10)
        self.dim_importance = (1 - self.learning_rate) * self.dim_importance + self.learning_rate * importance_update
        self.dim_importance = np.clip(self.dim_importance, 1e-10, 1)
        self.dim_importance /= np.sum(self.dim_importance)

        # 報酬の正規化を追加
        self.arm_rewards = (self.arm_rewards - np.mean(self.arm_rewards)) / (np.std(self.arm_rewards) + 1e-8)

    def optimize(self, n_iter):
        # 最適化のメイン関数。指定された回数だけ繰り返して最適な解を探す
        for _ in range(n_iter):
            # 活性化する次元を選択。これで探索空間を絞り込む
            active_dims = self.select_active_dims()

            # 選択された次元のデータだけを使ってモデルを学習する
            train_X = self.X[:, active_dims]
            train_Y = self.Y.unsqueeze(-1)

            # ガウス過程回帰モデルを作成。
            model = create_model(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            # Upper Confidence Bound (UCB) 獲得関数を定義。
            # これは探索（未知の領域）と活用（既知の良い領域）のバランスを取る
            UCB = UpperConfidenceBound(model, beta=0.1)
            bounds_active = torch.stack([self.bounds[0][active_dims], self.bounds[1][active_dims]]).double()

            # UCB獲得関数を最適化して、次に評価すべき候補点を見つける
            candidate, _ = optimize_acqf(
                UCB, bounds=bounds_active, q=1, num_restarts=10, raw_samples=100,
                options={"maxiter": 200, "batch_limit": 5}
            )

            # 新しい候補点を生成。選択されなかった次元は最良の既知の点の値を使う
            x_new = torch.zeros(self.dim, dtype=torch.double)
            x_new[active_dims] = candidate.squeeze()
            x_new[np.setdiff1d(range(self.dim), active_dims)] = self.best_x[
                np.setdiff1d(range(self.dim), active_dims)]

            # 新しい候補点での目的関数の値を計算
            y_new = self.obj_function(x_new.unsqueeze(0))

            # 各次元の感度を計算。これで重要な次元を特定できる
            self.calculate_dimension_sensitivity(x_new, y_new)

            # モデルを使って新しい点での予測標準偏差を計算。
            # これは不確実性の指標として使える
            with torch.no_grad():
                pred = model(x_new[active_dims].unsqueeze(0))
            sigma = pred.stddev.item()
            self.sigma_history.append(sigma)

            # 改善量を計算して、バンディットアルゴリズムを更新
            # これで各次元の重要度を学習
            improvement = max(0, self.best_f - y_new.item())
            self.update_bandit(active_dims, y_new.item())
            self.improvement_history.append(improvement)

            # もし新しい点が今までの最良値より良ければ、最良解を更新
            if y_new.item() < self.best_f:
                self.best_f = y_new.item()
                self.best_x = x_new

            # 評価履歴に現在の最良値を追加
            self.eval_history.append(self.best_f)

        # 次元選択の履歴をDataFrameに変換。これで後で分析しやすくなるんだ
        self.arm_selection_df = pd.DataFrame(self.arm_selection_history,
                                             columns=[f'Arm_{i}' for i in range(self.dim)])
        self.arm_selection_df.index.name = 'Iteration'

        # 最適化が終わったら、最良の解と最良の目的関数値を返すよ
        return self.best_x, self.best_f

    # 以下、結果の可視化や保存のための関数たち
    def save_arm_selection_history(self, filename):
        self.arm_selection_df.to_csv(filename)

    def plot_sigma_history(self):
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(self.sigma_history) + 1), self.sigma_history)
        plt.xlabel('Iteration')
        plt.ylabel('Predicted Standard Deviation')
        plt.title('Predicted Standard Deviation vs Iteration')
        plt.grid(True)
        plt.show()

    def plot_dim_importance(self):
        plt.figure(figsize=(12, 6))
        plt.bar(range(self.dim), self.dim_importance)
        plt.xlabel('Dimension')
        plt.ylabel('Importance')
        plt.title('Dimension Importance')
        plt.xticks(range(self.dim))
        plt.grid(True)
        plt.show()

    def plot_dim_sensitivity(self):
        plt.figure(figsize=(12, 6))
        plt.bar(range(self.dim), self.dim_sensitivity)
        plt.xlabel('Dimension')
        plt.ylabel('Sensitivity')
        plt.title('Dimension Sensitivity')
        plt.xticks(range(self.dim))
        plt.grid(True)
        plt.show()



dim = 10
active_dim = 5
bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
n_initial = 200
n_iter = 5

dropout_bandit_bc_ucb = DropoutMixBO_BC_UCB(dim, active_dim, bounds, n_initial, styblinski_tang, dropout_prob=0.0)

dropout_bandit_bc_ucb_best_x, dropout_bandit_bc_ucb_best_f = dropout_bandit_bc_ucb.optimize(n_iter)
dropout_bandit_bc_ucb.save_arm_selection_history('dropout_bandit_bc_ucb_arm_selection_binary.csv')

