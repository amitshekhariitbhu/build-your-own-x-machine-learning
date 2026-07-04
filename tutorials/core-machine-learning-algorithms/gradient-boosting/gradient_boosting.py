import numpy as np

# A shallow CART regression tree: the weak learner for boosting.
class RegressionTree:
    def __init__(self, max_depth=3, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples

    def fit(self, X, y):
        self.root = self._grow(X, y, depth=0)
        return self

    def _best_split(self, X, y):
        # Pick the split minimizing total squared error of the two children.
        best = (None, None, np.inf)  # feature, threshold, sse
        for f in range(X.shape[1]):
            for t in np.unique(X[:, f]):
                left = X[:, f] <= t
                if left.sum() == 0 or left.sum() == len(y):
                    continue
                yl, yr = y[left], y[~left]
                sse = np.sum((yl - yl.mean()) ** 2) + np.sum((yr - yr.mean()) ** 2)
                if sse < best[2]:
                    best = (f, t, sse)
        return best

    def _grow(self, X, y, depth):
        # Leaf value is the mean target (minimizes squared loss locally).
        if depth >= self.max_depth or len(y) < self.min_samples or np.all(y == y[0]):
            return float(y.mean())
        f, t, _ = self._best_split(X, y)
        if f is None:
            return float(y.mean())
        left = X[:, f] <= t
        return (f, t, self._grow(X[left], y[left], depth + 1),
                self._grow(X[~left], y[~left], depth + 1))

    def _predict_one(self, x, node):
        if not isinstance(node, tuple):
            return node
        f, t, l, r = node
        return self._predict_one(x, l if x[f] <= t else r)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])


# Gradient boosting for squared loss: fit trees to the residuals (neg. gradient).
class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(self, X, y):
        self.init_ = float(y.mean())    # start from the mean prediction
        self.trees = []
        pred = np.full(len(y), self.init_)
        for _ in range(self.n_estimators):
            residual = y - pred         # neg. gradient of 0.5*(y-pred)^2
            tree = RegressionTree(max_depth=self.max_depth).fit(X, residual)
            pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X):
        pred = np.full(X.shape[0], self.init_)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred


if __name__ == "__main__":
    np.random.seed(0)
    # Noisy nonlinear 1D target: y = sin(2x) + 0.5x + noise.
    X = np.linspace(-3, 3, 120).reshape(-1, 1)
    y = np.sin(2 * X[:, 0]) + 0.5 * X[:, 0] + 0.3 * np.random.randn(120)

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X, y)

    round0_mse = np.mean((y - y.mean()) ** 2)          # MSE of the mean baseline
    final_mse = np.mean((y - model.predict(X)) ** 2)   # MSE after boosting

    print("Samples           :", len(y))
    print("Round 0 MSE (mean): {:.4f}".format(round0_mse))
    print("Final MSE         : {:.4f}".format(final_mse))
    print("MSE reduction     : {:.2%}".format(1 - final_mse / round0_mse))
    print("Improved          :", final_mse < round0_mse)
