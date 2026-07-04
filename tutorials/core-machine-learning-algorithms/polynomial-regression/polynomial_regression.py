import numpy as np


class PolynomialRegression:
    """Polynomial regression via polynomial-basis expansion + normal equation."""

    def __init__(self, degree=2):
        self.degree = degree
        self.coef_ = None  # weights, first entry is the bias

    def _expand(self, X):
        # X: (n_samples, n_features) -> [1, x, x^2, ..., x^degree] per feature
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        cols = [np.ones((X.shape[0], 1))]  # bias term
        for d in range(1, self.degree + 1):
            cols.append(X ** d)
        return np.hstack(cols)

    def fit(self, X, y):
        Phi = self._expand(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        # normal equation: w = (PhiT Phi)^-1 PhiT y  (pinv for stability)
        self.coef_ = np.linalg.pinv(Phi.T @ Phi) @ Phi.T @ y
        return self

    def predict(self, X):
        return self._expand(X) @ self.coef_

    def score(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        pred = self.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot  # R^2


if __name__ == "__main__":
    np.random.seed(0)

    # true cubic: y = 2 - 3x + 0.5x^2 + 1.5x^3 + noise
    n = 80
    x = np.linspace(-3, 3, n)
    true_coef = np.array([2.0, -3.0, 0.5, 1.5])
    y = true_coef[0] + true_coef[1] * x + true_coef[2] * x ** 2 + true_coef[3] * x ** 3
    y = y + np.random.normal(0, 2.0, size=n)

    model = PolynomialRegression(degree=3).fit(x, y)
    pred = model.predict(x)
    mse = np.mean((y - pred) ** 2)

    print("true  coefficients:", np.round(true_coef, 3))
    print("learned coefficients:", np.round(model.coef_, 3))
    print("MSE:", round(float(mse), 4))
    print("R^2:", round(float(model.score(x, y)), 4))
