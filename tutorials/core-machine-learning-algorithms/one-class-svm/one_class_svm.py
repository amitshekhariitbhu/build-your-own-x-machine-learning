import numpy as np

class OneClassSVM:
    """One-Class SVM (Scholkopf) for novelty/anomaly detection.
    RBF random features + sub-gradient descent learn a hyperplane w.phi(x) - rho
    that separates the data from the origin. nu upper-bounds the outlier fraction.
    predict: +1 inlier if w.phi(x) - rho >= 0 else -1 (outlier)."""

    def __init__(self, nu=0.1, gamma=0.3, n_components=200,
                 learning_rate=0.05, n_epochs=800):
        self.nu = nu                      # ~ upper bound on the outlier fraction
        self.gamma = gamma                # RBF kernel width
        self.n_components = n_components  # number of random Fourier features
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def _features(self, X):
        # Random Fourier features approximating the RBF kernel exp(-gamma||x-y||^2)
        return np.sqrt(2.0 / self.n_components) * np.cos(X @ self.W + self.b)

    def fit(self, X):
        n, d = X.shape
        # Sample the random Fourier basis: omega ~ N(0, 2*gamma), b ~ U(0, 2pi)
        self.W = np.random.randn(d, self.n_components) * np.sqrt(2 * self.gamma)
        self.b = np.random.uniform(0, 2 * np.pi, self.n_components)
        Z = self._features(X)

        # Minimize 1/2||w||^2 - rho + 1/(nu*n) * sum(max(0, rho - w.z))
        self.w = np.zeros(self.n_components)
        self.rho = 0.0
        for _ in range(self.n_epochs):
            scores = Z @ self.w
            viol = scores < self.rho                      # points inside the margin
            grad_w = self.w - Z[viol].sum(axis=0) / (self.nu * n)
            grad_rho = -1.0 + viol.sum() / (self.nu * n)
            self.w -= self.learning_rate * grad_w
            self.rho -= self.learning_rate * grad_rho
        return self

    def decision_function(self, X):
        # Signed distance to the hyperplane; >= 0 means inlier
        return self._features(X) @ self.w - self.rho

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    # Train only on "normal" data: a tight Gaussian cluster
    X_train = np.random.randn(200, 2) * 0.5 + np.array([5.0, 5.0])

    # Test inliers (same distribution) + outliers scattered far on a ring
    X_inliers = np.random.randn(20, 2) * 0.5 + np.array([5.0, 5.0])
    angles = np.random.uniform(0, 2 * np.pi, 15)
    X_outliers = np.array([5.0, 5.0]) + 5.0 * np.c_[np.cos(angles), np.sin(angles)]

    model = OneClassSVM(nu=0.1, gamma=0.3, n_components=200,
                        learning_rate=0.05, n_epochs=800)
    model.fit(X_train)

    pred_in = model.predict(X_inliers)
    pred_out = model.predict(X_outliers)

    print("Learned rho:", round(model.rho, 4))
    print("Inlier decision scores (first 5):",
          np.round(model.decision_function(X_inliers)[:5], 3))
    print("Outlier decision scores (first 5):",
          np.round(model.decision_function(X_outliers)[:5], 3))
    print("Inlier predictions (want +1):", pred_in)
    print("Outlier predictions (want -1):", pred_out)

    # Correctness signal: how many inliers kept (+1) and outliers caught (-1)
    inlier_acc = np.mean(pred_in == 1)
    outlier_acc = np.mean(pred_out == -1)
    y_true = np.r_[np.ones(len(pred_in)), -np.ones(len(pred_out))]
    y_pred = np.r_[pred_in, pred_out]
    print("Inlier retention:", round(inlier_acc, 3))
    print("Outlier detection rate:", round(outlier_acc, 3))
    print("Overall accuracy:", round(np.mean(y_true == y_pred), 3))
