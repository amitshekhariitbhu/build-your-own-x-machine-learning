import numpy as np


class DecisionStump:
    # A one-level decision tree: split on one feature at one threshold.
    def __init__(self):
        self.feature = 0
        self.threshold = 0.0
        self.polarity = 1
        self.alpha = 0.0

    def predict(self, X):
        col = X[:, self.feature]
        pred = np.ones(X.shape[0])
        # Points on the "wrong" side of the threshold get -1.
        pred[self.polarity * col < self.polarity * self.threshold] = -1
        return pred


class AdaBoost:
    # Discrete AdaBoost (SAMME) with decision stumps, labels {-1, +1}.
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.stumps = []

    def fit(self, X, y):
        n, m = X.shape
        w = np.full(n, 1.0 / n)  # uniform sample weights
        self.stumps = []
        for _ in range(self.n_estimators):
            stump = self._best_stump(X, y, w)
            pred = stump.predict(X)
            err = np.sum(w[pred != y])
            err = np.clip(err, 1e-10, 1 - 1e-10)
            stump.alpha = 0.5 * np.log((1 - err) / err)
            # Up-weight misclassified, down-weight correct, then renormalize.
            w *= np.exp(-stump.alpha * y * pred)
            w /= np.sum(w)
            self.stumps.append(stump)
        return self

    def _best_stump(self, X, y, w):
        # Exhaustively search feature/threshold/polarity for min weighted error.
        n, m = X.shape
        best, best_err = DecisionStump(), np.inf
        for feat in range(m):
            for thr in np.unique(X[:, feat]):
                for pol in (1, -1):
                    pred = np.ones(n)
                    pred[pol * X[:, feat] < pol * thr] = -1
                    err = np.sum(w[pred != y])
                    if err < best_err:
                        best_err = err
                        best.feature, best.threshold, best.polarity = feat, thr, pol
        return best

    def predict(self, X):
        agg = sum(s.alpha * s.predict(X) for s in self.stumps)
        return np.sign(agg)


if __name__ == "__main__":
    np.random.seed(0)
    # Two Gaussian blobs, labels in {-1, +1}.
    n = 100
    X0 = np.random.randn(n, 2) + np.array([-2, -2])
    X1 = np.random.randn(n, 2) + np.array([2, 2])
    X = np.vstack([X0, X1])
    y = np.concatenate([-np.ones(n), np.ones(n)])
    # Shuffle and split.
    idx = np.random.permutation(2 * n)
    X, y = X[idx], y[idx]
    Xtr, ytr, Xte, yte = X[:150], y[:150], X[150:], y[150:]

    model = AdaBoost(n_estimators=20).fit(Xtr, ytr)
    acc = np.mean(model.predict(Xte) == yte)
    print(f"Trained {len(model.stumps)} stumps")
    print(f"Test accuracy: {acc:.3f}")
