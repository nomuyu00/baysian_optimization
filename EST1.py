import numpy as np
from scipy.linalg import hadamard


class Environment:
    def __init__(self, n):
        self.true_rewards = np.random.rand(n)

    def sample(self, arms):
        return np.sum(np.random.normal(self.true_rewards[arms], 0.1))


def next_power_of_2(n):
    return 2 ** int(np.ceil(np.log2(n)))


def est1(env, N, k, epsilon, delta):
    n = len(N)
    m = int(2 / (epsilon ** 2) * np.log(2 * n / delta))

    hadamard_size = next_power_of_2(2 * k)
    H = hadamard(hadamard_size)

    all_estimates = np.zeros(n)

    for l in range(0, n, 2 * k):
        group = N[l:l + 2 * k]
        Z = np.zeros(hadamard_size)
        group_size = len(group)

        for i in range(hadamard_size):
            if i == 0:
                S_pos = group[:k]
                S_neg = group[k:2 * k]
            else:
                S_pos = [group[j] for j in range(group_size) if j < hadamard_size and H[i, j] > 0]
                S_neg = [group[j] for j in range(group_size) if j < hadamard_size and H[i, j] < 0]

            mu_pos = np.mean([env.sample(S_pos) for _ in range(m)]) if len(S_pos) > 0 else 0
            mu_neg = np.mean([env.sample(S_neg) for _ in range(m)]) if len(S_neg) > 0 else 0

            Z[i] = mu_pos + mu_neg if i == 0 else mu_pos - mu_neg

        theta_hat = np.dot(H, Z) / hadamard_size
        all_estimates[l:l + group_size] = theta_hat[:group_size]

    return all_estimates


def csar(env, n, k, delta):
    N = np.arange(n)
    A = np.array([], dtype=int)
    epsilon = 1 / 2
    delta_1 = 6 * delta / (np.pi ** 2)
    t = 1

    while len(N) + len(A) > k:
        theta_hat = est1(env, N, k, epsilon, delta_1)

        sorted_indices = np.argsort(theta_hat)[::-1]
        theta_k = theta_hat[sorted_indices[min(k - 1, len(sorted_indices) - 1)]]
        theta_k_plus_1 = theta_hat[sorted_indices[min(k, len(sorted_indices) - 1)]]

        A_new = N[theta_hat - theta_k_plus_1 > 2 * epsilon]
        R = N[theta_k - theta_hat > 2 * epsilon]

        A = np.concatenate((A, A_new))
        N = N[~np.isin(N, np.concatenate((A_new, R)))]

        epsilon /= 2
        delta_1 = 6 * delta / (np.pi ** 2 * t ** 2)
        t += 1

    return np.concatenate((A, N[:k - len(A)]))


# 使用例
n = 20  # アームの総数
k = 5  # 選択するアーム数
delta = 0.05  # 信頼度パラメータ

env = Environment(n)
selected_arms = csar(env, n, k, delta)

print("Selected arms:", selected_arms)
print("True rewards of selected arms:", env.true_rewards[selected_arms])
print("Average reward of selected arms:", np.mean(env.true_rewards[selected_arms]))
print("Maximum possible average reward:", np.mean(np.sort(env.true_rewards)[-k:]))