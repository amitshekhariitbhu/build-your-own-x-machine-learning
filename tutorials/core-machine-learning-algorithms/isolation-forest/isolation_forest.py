import numpy as np


def _c(n):
    # Average path length of an unsuccessful search in a BST of n points.
    if n <= 1:
        return 0.0                                  # single/empty node: no extra depth
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


# One isolation tree: recursively split on a random feature at a random value.
# Anomalies get isolated fast, so they end up with short path lengths.
class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit

    def fit(self, X, depth=0):
        n = len(X)
        # Stop: too deep or too few points -> leaf holding remaining size.
        if depth >= self.height_limit or n <= 1:
            self.leaf_size = n
            self.left = None
            return self
        self.leaf_size = None
        f = np.random.randint(X.shape[1])          # random feature
        lo, hi = X[:, f].min(), X[:, f].max()
        if lo == hi:                                # constant column -> leaf
            self.leaf_size = n
            self.left = None
            return self
        self.feature = f
        self.split = np.random.uniform(lo, hi)     # random split value
        mask = X[:, f] < self.split
        self.left = IsolationTree(self.height_limit).fit(X[mask], depth + 1)
        self.right = IsolationTree(self.height_limit).fit(X[~mask], depth + 1)
        return self

    def path_length(self, x, depth=0):
        # Depth reached + expected extra length for the unsplit leaf points.
        if self.leaf_size is not None:
            return depth + _c(self.leaf_size)
        branch = self.left if x[self.feature] < self.split else self.right
        return branch.path_length(x, depth + 1)


# Ensemble of isolation trees; anomaly score s = 2^(-E(h(x)) / c(n)).
class IsolationForest:
    def __init__(self, n_trees=100, sample_size=256):
        self.n_trees = n_trees
        self.sample_size = sample_size

    def fit(self, X):
        n = min(self.sample_size, len(X))
        self.c = _c(n)
        limit = int(np.ceil(np.log2(n)))           # tree height limit
        self.trees = []
        for _ in range(self.n_trees):
            idx = np.random.choice(len(X), n, replace=False)  # subsample
            self.trees.append(IsolationTree(limit).fit(X[idx]))
        return self

    def score(self, X):
        # Mean path length across trees -> normalized anomaly score in (0, 1).
        h = np.array([[t.path_length(x) for t in self.trees] for x in X])
        return 2.0 ** (-h.mean(axis=1) / self.c)

    def predict(self, X, threshold=0.5):
        # 1 = anomaly (high score), 0 = normal.
        return (self.score(X) > threshold).astype(int)


if __name__ == "__main__":
    np.random.seed(0)
    # Dense normal cluster (200 pts) plus a few injected far-away outliers.
    normal = np.random.randn(200, 2) * 0.5 + np.array([5.0, 5.0])
    outliers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    X = np.vstack([normal, outliers])
    y_true = np.array([0] * len(normal) + [1] * len(outliers))

    forest = IsolationForest(n_trees=100, sample_size=128).fit(X)
    scores = forest.score(X)
    preds = forest.predict(X, threshold=0.7)

    print("Points                :", len(X), "(200 normal + 4 outliers)")
    print("Mean score normal     : {:.3f}".format(scores[y_true == 0].mean()))
    print("Mean score outliers   : {:.3f}".format(scores[y_true == 1].mean()))
    print("Injected outliers     :", outliers.tolist())
    print("Outlier scores        :", np.round(scores[y_true == 1], 3).tolist())
    print("Flagged anomalies     :", X[preds == 1].tolist())
    print("Detection accuracy    : {:.3f}".format(np.mean(preds == y_true)))
    print("Outliers > normals    :", scores[y_true == 1].min() > scores[y_true == 0].max())
