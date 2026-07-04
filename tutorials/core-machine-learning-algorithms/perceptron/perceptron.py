import numpy as np


class Perceptron:
    """Rosenblatt perceptron with unit-step activation, labels {0, 1}."""

    def __init__(self, lr=0.1, epochs=50):
        self.lr = lr
        self.epochs = epochs

    def _step(self, z):
        return np.where(z >= 0.0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        for _ in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                yhat = self._step(np.dot(xi, self.w) + self.b)
                update = self.lr * (yi - yhat)
                self.w += update * xi
                self.b += update
                errors += int(update != 0.0)
            if errors == 0:  # converged: no misclassifications this epoch
                break
        return self

    def predict(self, X):
        return self._step(X @ self.w + self.b)


if __name__ == "__main__":
    np.random.seed(0)
    # Two linearly separable Gaussian blobs.
    n = 50
    X0 = np.random.randn(n, 2) + np.array([-2.0, -2.0])
    X1 = np.random.randn(n, 2) + np.array([2.0, 2.0])
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n), np.ones(n)]).astype(int)

    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]

    clf = Perceptron(lr=0.1, epochs=50).fit(X, y)
    preds = clf.predict(X)
    acc = np.mean(preds == y)

    print("weights:", clf.w)
    print("bias:", clf.b)
    print("accuracy:", acc)
