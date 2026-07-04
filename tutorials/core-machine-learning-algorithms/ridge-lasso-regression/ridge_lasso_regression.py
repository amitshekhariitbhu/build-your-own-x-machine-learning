import numpy as np

class Ridge:
    # L2-regularized linear regression via closed-form normal equation.
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Center data so the intercept is not penalized.
        X_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - X_mean, y - y_mean

        # w = (X^T X + alpha*I)^-1 X^T y
        n_features = Xc.shape[1]
        A = Xc.T @ Xc + self.alpha * np.eye(n_features)
        self.weights = np.linalg.inv(A) @ Xc.T @ yc
        self.bias = y_mean - X_mean @ self.weights

    def predict(self, X):
        return X @ self.weights + self.bias

class Lasso:
    # L1-regularized linear regression via coordinate descent + soft-thresholding.
    def __init__(self, alpha=1.0, n_iters=1000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    @staticmethod
    def _soft_threshold(rho, lam):
        # Shrink rho toward zero by lam; snap small values to exactly 0.
        return np.sign(rho) * max(abs(rho) - lam, 0.0)

    def fit(self, X, y):
        X_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - X_mean, y - y_mean
        n_samples, n_features = Xc.shape
        w = np.zeros(n_features)

        # Cyclically optimize one coordinate at a time.
        for _ in range(self.n_iters):
            for j in range(n_features):
                residual = yc - Xc @ w + Xc[:, j] * w[j]
                rho = Xc[:, j] @ residual
                z = Xc[:, j] @ Xc[:, j]
                w[j] = self._soft_threshold(rho, self.alpha * n_samples) / z

        self.weights = w
        self.bias = y_mean - X_mean @ self.weights

    def predict(self, X):
        return X @ self.weights + self.bias

# Example usage
if __name__ == "__main__":
    np.random.seed(0)

    # Synthetic linear data: feature 2 is pure noise (true coef = 0).
    n = 200
    X = np.random.randn(n, 3)
    true_w = np.array([3.0, -2.0, 0.0])
    y = X @ true_w + 5.0 + 0.1 * np.random.randn(n)

    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.1)
    ridge.fit(X, y)
    lasso.fit(X, y)

    def r2(model):
        pred = model.predict(X)
        return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)

    print("True coefficients: ", true_w)
    print("Ridge coefficients:", np.round(ridge.weights, 3), " R^2:", round(r2(ridge), 4))
    print("Lasso coefficients:", np.round(lasso.weights, 3), " R^2:", round(r2(lasso), 4))
    print("Irrelevant feature -> Ridge coef:", round(ridge.weights[2], 4),
          "| Lasso coef:", round(lasso.weights[2], 4))
    print("Lasso zeroed the irrelevant feature:", abs(lasso.weights[2]) < 1e-6)
