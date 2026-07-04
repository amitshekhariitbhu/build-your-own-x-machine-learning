import numpy as np

class ElasticNet:
    # Linear regression with combined L1 + L2 penalty solved by coordinate descent.
    # Objective: (1/2n)||y - Xw - b||^2 + alpha*l1_ratio*|w| + 0.5*alpha*(1-l1_ratio)*w^2
    def __init__(self, alpha=1.0, l1_ratio=0.5, n_iters=1000, tol=1e-6):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_iters = n_iters
        self.tol = tol
        self.weights = None
        self.bias = 0.0

    @staticmethod
    def _soft_threshold(rho, lam):
        # Soft-thresholding operator (proximal step for the L1 term)
        return np.sign(rho) * max(abs(rho) - lam, 0.0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = np.mean(y)

        l1 = self.alpha * self.l1_ratio            # L1 strength
        l2 = self.alpha * (1.0 - self.l1_ratio)    # L2 strength
        col_sq = np.sum(X ** 2, axis=0) / n_samples  # per-feature normalizer

        for _ in range(self.n_iters):
            w_old = self.weights.copy()
            # Update bias as residual mean (intercept is unpenalized)
            self.bias = np.mean(y - X @ self.weights)
            for j in range(n_features):
                # Partial residual excluding feature j
                r_j = y - self.bias - X @ self.weights + X[:, j] * self.weights[j]
                rho = np.dot(X[:, j], r_j) / n_samples
                # Soft-threshold the L1 part, shrink by the L2 part
                self.weights[j] = self._soft_threshold(rho, l1) / (col_sq[j] + l2)
            if np.max(np.abs(self.weights - w_old)) < self.tol:
                break

    def predict(self, X):
        return X @ self.weights + self.bias

# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    # Synthetic data: only 3 of 10 features are truly informative (sparse signal)
    n_samples, n_features = 200, 10
    X = np.random.randn(n_samples, n_features)
    true_w = np.array([5.0, 0.0, -3.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    y = X @ true_w + 1.5 + 0.5 * np.random.randn(n_samples)

    model = ElasticNet(alpha=0.1, l1_ratio=0.5, n_iters=1000)
    model.fit(X, y)

    preds = model.predict(X)
    mse = np.mean((y - preds) ** 2)
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    np.set_printoptions(precision=3, suppress=True)
    print("True weights:     ", true_w)
    print("Learned weights:  ", model.weights)
    print("Learned bias:      %.3f (true 1.5)" % model.bias)
    print("Nonzero coeffs:    %d (true 3)" % np.sum(np.abs(model.weights) > 1e-4))
    print("MSE:               %.4f" % mse)
    print("R^2:               %.4f" % r2)
