import numpy as np

# A small Gini-based decision tree used as the base learner.
class DecisionTree:
    def __init__(self, max_depth=8, min_samples=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_features = n_features  # size of random feature subset per split

    def fit(self, X, y):
        self.root = self._grow(X, y, depth=0)
        return self

    @staticmethod
    def _gini(y):
        # Gini impurity of a label array.
        counts = np.bincount(y)
        p = counts / len(y)
        return 1.0 - np.sum(p ** 2)

    def _best_split(self, X, y, feats):
        best = (None, None, 1e18)  # feature, threshold, weighted impurity
        for f in feats:
            for t in np.unique(X[:, f]):
                left = X[:, f] <= t
                if left.sum() == 0 or left.sum() == len(y):
                    continue
                g = (left.sum() * self._gini(y[left]) +
                     (~left).sum() * self._gini(y[~left])) / len(y)
                if g < best[2]:
                    best = (f, t, g)
        return best

    def _grow(self, X, y, depth):
        # Leaf if pure, too deep, or too few samples: store majority label.
        if (depth >= self.max_depth or len(y) < self.min_samples or
                len(np.unique(y)) == 1):
            return np.bincount(y).argmax()
        n = self.n_features or X.shape[1]
        feats = np.random.choice(X.shape[1], n, replace=False)
        f, t, _ = self._best_split(X, y, feats)
        if f is None:
            return np.bincount(y).argmax()
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


# Bagging ensemble: many trees on bootstrap samples + random feature subsets.
class RandomForest:
    def __init__(self, n_trees=25, max_depth=8, min_samples=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples

    def fit(self, X, y):
        n_features = max(1, int(np.sqrt(X.shape[1])))  # sqrt-sized subset
        self.trees = []
        for _ in range(self.n_trees):
            idx = np.random.choice(len(y), len(y), replace=True)  # bootstrap rows
            tree = DecisionTree(self.max_depth, self.min_samples, n_features)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict(self, X):
        # Majority vote across all trees.
        preds = np.array([t.predict(X) for t in self.trees])
        return np.array([np.bincount(col).argmax() for col in preds.T])


if __name__ == "__main__":
    np.random.seed(0)
    # 3 overlapping Gaussian blobs (signal) + noise features so a single deep
    # tree overfits and bagging's variance reduction becomes visible.
    means = [(0, 0), (5, 0), (0, 5)]
    signal = np.vstack([np.random.randn(100, 2) * 1.8 + m for m in means])
    noise = np.random.randn(300, 6)  # 6 irrelevant features
    X = np.hstack([signal, noise])
    y = np.repeat([0, 1, 2], 100)
    # Shuffle and split 70/30.
    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]
    cut = int(0.7 * len(y))
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

    forest = RandomForest(n_trees=25, max_depth=8).fit(Xtr, ytr)
    single = DecisionTree(max_depth=8).fit(Xtr, ytr)

    forest_acc = np.mean(forest.predict(Xte) == yte)
    single_acc = np.mean(single.predict(Xte) == yte)

    print("Test samples          :", len(yte))
    print("Single tree accuracy  : {:.3f}".format(single_acc))
    print("Random forest accuracy: {:.3f}".format(forest_acc))
    print("Forest >= single tree :", forest_acc >= single_acc)
