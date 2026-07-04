import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Feature index used for the split
        self.threshold = threshold  # Threshold value for the split
        self.left = left            # Left child (samples <= threshold)
        self.right = right          # Right child (samples > threshold)
        self.value = value          # Majority class if this is a leaf

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini(self, y):
        # Gini impurity = 1 - sum(p_k^2)
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        best_gain, best_feat, best_thr = 0.0, None, None
        parent_gini = self._gini(y)
        n_samples = len(y)
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thr in thresholds:
                left = y[X[:, feat] <= thr]
                right = y[X[:, feat] > thr]
                if len(left) == 0 or len(right) == 0:
                    continue
                # Weighted child impurity, then the reduction (gain)
                child = (len(left) * self._gini(left) + len(right) * self._gini(right)) / n_samples
                gain = parent_gini - child
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, feat, thr
        return best_feat, best_thr

    def _majority(self, y):
        return np.bincount(y).argmax()

    def _build(self, X, y, depth):
        # Stopping rules: pure node, max depth, or too few samples to split
        if (len(np.unique(y)) == 1 or depth >= self.max_depth
                or len(y) < self.min_samples_split):
            return Node(value=self._majority(y))
        feat, thr = self._best_split(X, y)
        if feat is None:  # No split improved impurity -> leaf
            return Node(value=self._majority(y))
        mask = X[:, feat] <= thr
        left = self._build(X[mask], y[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return Node(feature=feat, threshold=thr, left=left, right=right)

    def fit(self, X, y):
        self.root = self._build(X, y, 0)

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])


# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    # 3-class synthetic data: three Gaussian blobs in 2D
    centers = np.array([[0, 0], [4, 4], [0, 5]])
    X = np.vstack([c + np.random.randn(50, 2) for c in centers])
    y = np.repeat([0, 1, 2], 50)

    model = DecisionTree(max_depth=5, min_samples_split=2)
    model.fit(X, y)
    preds = model.predict(X)

    accuracy = np.mean(preds == y)
    print("Number of samples:", len(y))
    print("Classes:", np.unique(y))
    print("Training accuracy: {:.2%}".format(accuracy))
