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
from botorch.acquisition import ExpectedImprovement
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


class DropoutMixEST1BO:
    def __init__(self, dim, active_dim, bounds, n_initial, obj_function, dropout_prob=0.1, delta=0.1):
        self.dim = dim  # 全次元数
        self.active_dim = active_dim  # 活性化する次元数
        self.bounds = bounds  # 各次元の探索範囲
        self.obj_function = obj_function  # 最適化したい目的関数
        self.dropout_prob = dropout_prob  # ドロップアウトの確率
        self.delta = delta  # 信頼度
        self.X = generate_initial_points(n_initial, dim, bounds).float()  # 初期点を生成
        self.Y = self.obj_function(self.X).float()  # 初期点での目的関数の値を計算
        self.best_f = self.Y.min().item()  # 現在の最良の目的関数値
        self.best_x = self.X[self.Y.argmin()]  # 現在の最良の解
        self.eval_history = [self.best_f] * n_initial  # 評価履歴を初期化
        self.improvement_history = [] # 報酬の履歴を初期化
        self.iteration = 0  # 現在のイテレーション

        # CSARアルゴリズムの初期化
        self.N = list(range(self.dim))  # 全次元の集合
        self.accepted_dims = []  # 受理された次元
        self.rejected_dims = []  # 除外された次元
        self.remaining_dims = self.N.copy()  # 残りの次元
        self.theta_hat = np.zeros(self.dim)  # 各次元の推定報酬
        self.phase = 1  # 現在のフェーズ
        self.epsilon_t = 0.5  # 初期の精度レベル
        self.delta_t = (6 / np.pi ** 2) * self.delta  # 初期の信頼度レベル

    def EST1(self, N_t, k, epsilon_t, delta_t):
        n = len(N_t)
        m = int((2 / epsilon_t ** 2) * np.log(2 * n / delta_t))  # サンプル数を計算
        # N_tをサイズ2kの部分集合に分割
        num_subsets = int(np.ceil(n / (2 * k)))
        subsets = []
        for i in range(num_subsets):
            subset = N_t[i * 2 * k:(i + 1) * 2 * k]
            if len(subset) < 2 * k:
                # 次元が足りない場合は繰り返しで埋める
                subset += subset[:(2 * k - len(subset))]
            subsets.append(subset)

        # 推定報酬と出現回数を初期化
        theta_hat = np.zeros(n)
        counts = np.zeros(n)

        # 各部分集合について推定を行う
        for subset in subsets:
            # サイズ2kのハダマード行列を作成
            H = self.create_hadamard(2 * k)
            Z_hat = np.zeros(2 * k)

            # subset内の次元をN_tのインデックスにマッピング
            subset_indices = [N_t.index(dim) for dim in subset]

            # ハダマード行列に従って部分集合をサンプリング
            for i in range(2 * k):
                h_row = H[i]
                if i == 0:
                    pos_dims = subset[:k]
                    neg_dims = subset[k:2 * k]
                else:
                    pos_dims = [subset[j] for j in range(2 * k) if h_row[j] == 1]
                    neg_dims = [subset[j] for j in range(2 * k) if h_row[j] == -1]

                pos_samples = []
                neg_samples = []

                for l in range(m):
                    # 正の次元の報酬をサンプリング
                    pos_sample = self.predict_without_x(pos_dims)
                    pos_samples.append(pos_sample)
                    # 負の次元の報酬をサンプリング
                    neg_sample = self.predict_without_x(neg_dims)
                    neg_samples.append(neg_sample)

                # 正の次元と負の次元の報酬の平均を計算
                mu_pos = np.mean(pos_samples) if len(pos_samples) > 0 else 0
                mu_neg = np.mean(neg_samples) if len(neg_samples) > 0 else 0

                if i == 0:
                    Z_hat[i] = mu_pos + mu_neg
                else:
                    Z_hat[i] = mu_pos - mu_neg

            # ハダマード行列を用いて次元ごとの報酬を推定
            theta_subset = (1 / (2 * k)) * H.T @ Z_hat

            # 推定された報酬をtheta_hatに反映し、出現回数を更新
            for idx, dim_idx in enumerate(subset_indices):
                theta_hat[dim_idx] += theta_subset[idx]
                counts[dim_idx] += 1

        # 各次元の推定報酬の平均を計算
        theta_hat = theta_hat / counts

        return theta_hat

    def create_hadamard(self, n):
        # サイズnのハダマード行列を作成（nは2の累乗）
        assert n & (n - 1) == 0, "ハダマード行列のサイズは2の累乗である必要があります"
        H = np.array([[1]])
        while H.shape[0] < n:
            H = np.block([[H, H], [H, -H]])
        return H

    def predict(self, X, active_dims):
        # GPモデルを使用して予測を行う
        train_X = self.X[:, active_dims].float()
        train_Y = self.Y.unsqueeze(-1).float()
        model = create_model(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        model.eval()
        self.iteration += 1
        print(self.iteration)
        with torch.no_grad():
            y_pred = model(X.float()).mean.item()
        return y_pred

    def predict_without_x(self, active_dims):
        # GPモデルを使用して予測を行う
        train_X = self.X[:, active_dims].float()
        train_Y = self.Y.unsqueeze(-1).float()
        model = create_model(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)
        model.eval()
        self.iteration += 1
        print(self.iteration)

        # Expected Improvement (EI) 獲得関数の定義
        EI = ExpectedImprovement(model, best_f=self.best_f, maximize=False)
        bounds_active = torch.stack([self.bounds[0][active_dims], self.bounds[1][active_dims]]).float()

        # 獲得関数の最適化
        candidate, _ = optimize_acqf(
            EI, bounds=bounds_active, q=1, num_restarts=10, raw_samples=100,
            options={"maxiter": 200, "batch_limit": 5}
        )

        # 新しい候補点を構築
        x_new = torch.zeros(self.dim, dtype=torch.float32)
        x_new[active_dims] = candidate.squeeze()
        x_new[np.setdiff1d(range(self.dim), active_dims)] = self.best_x[
            np.setdiff1d(range(self.dim), active_dims)]

        with torch.no_grad():
            y_pred = model(x_new[active_dims].unsqueeze(0).float()).mean.item()
        return y_pred

        # 目的関数の評価
        y_new = self.obj_function(x_new.unsqueeze(0))
        if isinstance(y_new, torch.Tensor):
            y_new = y_new.item()

        improvement = np.exp(-((y_pred - y_new) ** 2))
        self.improvement_history.append(improvement)

        # 活性化次元の推定報酬を更新
        for dim in active_dims:
            self.theta_hat[dim] = (self.theta_hat[dim] + improvement) / 2  # 平均を取る

        # データセットに新しい点を追加
        self.X = torch.cat([self.X, x_new.unsqueeze(0)])
        self.Y = torch.cat([self.Y, y_new])

        # 最良の解を更新
        if y_new < self.best_f:
            self.best_f = y_new.item()
            self.best_x = x_new

        self.eval_history.append(self.best_f)

        return improvement

    def optimize(self, n_iter):
        while self.iteration < n_iter:
            # CSARアルゴリズムの実行
            while len(self.remaining_dims) + len(self.accepted_dims) > self.active_dim:
                # 推定アルゴリズムEST1を使用して報酬を推定
                theta_hat_t = self.EST1(self.remaining_dims, self.active_dim, self.epsilon_t, self.delta_t)
                # 推定報酬に基づいて次元をソート
                sorted_indices = np.argsort(-theta_hat_t)
                sorted_dims = [self.remaining_dims[i] for i in sorted_indices]

                # 受理および除外する次元を決定
                theta_k = theta_hat_t[sorted_indices[self.active_dim - 1]]
                theta_k_plus_1 = theta_hat_t[sorted_indices[self.active_dim]] if len(sorted_dims) > self.active_dim else -np.inf

                A = [sorted_dims[i] for i in range(len(sorted_dims)) if theta_hat_t[sorted_indices[i]] - theta_k_plus_1 > 2 * self.epsilon_t]
                R = [sorted_dims[i] for i in range(len(sorted_dims)) if theta_k - theta_hat_t[sorted_indices[i]] > 2 * self.epsilon_t]

                self.accepted_dims.extend(A)
                self.rejected_dims.extend(R)
                self.remaining_dims = [dim for dim in self.remaining_dims if dim not in A and dim not in R]

                # 精度と信頼度を更新
                self.phase += 1
                self.epsilon_t /= 2
                self.delta_t = (6 / (np.pi ** 2)) * (self.delta / (self.phase ** 2))

                # 必要な次元数が揃ったらループを抜ける
                if len(self.accepted_dims) >= self.active_dim:
                    break

            # 活性化次元を決定
            if len(self.accepted_dims) >= self.active_dim:
                active_dims = self.accepted_dims[:self.active_dim]
            else:
                active_dims = self.accepted_dims + self.remaining_dims[:(self.active_dim - len(self.accepted_dims))]

            active_dims = active_dims[:self.active_dim]  # 必要に応じて調整

            # 報酬を計算
            improvement = self.predict_without_x(active_dims)

            # 次のイテレーションのためにリセット
            self.remaining_dims = self.N.copy()
            self.accepted_dims = []
            self.rejected_dims = []
            self.phase = 1
            self.epsilon_t = 0.5
            self.delta_t = (6 / np.pi ** 2) * self.delta

        return self.best_x, self.best_f


dim = 10
active_dim = 2
bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
n_initial = 200
n_iter = 5

dropout_bandit_est1 = DropoutMixEST1BO(dim, active_dim, bounds, n_initial, styblinski_tang, dropout_prob=0.0)

dropout_bandit_est1_best_x, dropout_bandit_est1_best_f = dropout_bandit_est1.optimize(n_iter)

