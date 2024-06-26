import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import differential_evolution
from sklearn.neighbors import KNeighborsRegressor


def create_objective_function():
    N = 10000  # サンプル数
    D = 10  # 次元
    kernel = RBF(length_scale=0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=42)

    # ランダムにサンプルポイントを生成
    X = np.random.uniform(-1, 1, size=(N, D))

    # ランダムな値を生成
    y = gpr.sample_y(X, random_state=42).ravel()

    # KNN回帰器を使用して近似関数を作成
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X, y)

    def objective(X):
        return knn.predict(X)

    return objective


objective_function = create_objective_function()


def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)

    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def dropout_optimization(objective, bounds, n_iter, d, p):
    D = len(bounds)
    X = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(50, D))
    y = objective(X)

    best_x = X[np.argmax(y)]
    best_y = np.max(y)

    ei_values = []
    obj_values = []

    for i in range(n_iter):
        active_dims = np.random.choice(D, d, replace=False)

        kernel = ConstantKernel() * RBF(length_scale=np.ones(d))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6)
        gpr.fit(X[:, active_dims], y)

        def negative_ei(x):
            return -expected_improvement(x.reshape(1, -1), X[:, active_dims], y, gpr)

        x_new = np.zeros(D)
        res = differential_evolution(negative_ei, [(bounds[i, 0], bounds[i, 1]) for i in active_dims])
        x_new[active_dims] = res.x

        if np.random.random() < p:
            x_new[np.setdiff1d(np.arange(D), active_dims)] = np.random.uniform(
                bounds[np.setdiff1d(np.arange(D), active_dims), 0], bounds[np.setdiff1d(np.arange(D), active_dims), 1])
        else:
            x_new[np.setdiff1d(np.arange(D), active_dims)] = best_x[np.setdiff1d(np.arange(D), active_dims)]

        y_new = objective(x_new.reshape(1, -1))

        X = np.vstack((X, x_new))
        y = np.append(y, y_new)

        if y_new > best_y:
            best_x = x_new
            best_y = y_new

        ei_values.append(-negative_ei(best_x[active_dims]))
        obj_values.append(best_y)

    return best_x, best_y, X, y, ei_values, obj_values


def main():
    D = 10
    bounds = np.array([[-1, 1]] * D)
    n_iter = 500
    d = 3
    p = 0.1

    best_x, best_y, X, y, ei_values, obj_values = dropout_optimization(objective_function, bounds, n_iter, d, p)

    print(f"Best solution: {best_x}")
    print(f"Best objective value: {best_y}")

    # 結果の可視化（2次元の射影）
    plt.figure(figsize=(12, 10))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.colorbar(label='Objective value')
    plt.scatter(best_x[0], best_x[1], c='r', marker='*', s=200, label='Best solution (projection)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Dropout Optimization Results (2D projection)')
    plt.legend()
    plt.show()

    # 獲得関数と目的関数の収束の様子
    plt.figure(figsize=(12, 6))
    plt.plot(range(n_iter), ei_values, label='Expected Improvement')
    plt.plot(range(n_iter), obj_values, label='Best Objective Value')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Convergence of Expected Improvement and Objective Function')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()