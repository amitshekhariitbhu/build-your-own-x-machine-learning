import numpy as np

# Bagging (Bootstrap AGGregatING) ensemble, from scratch.
# Base learner: a hand-written CART decision tree (Gini impurity). Deep trees are
# low-bias / high-variance -> they overfit label noise. Bagging trains each tree on
# a bootstrap resample (draw n rows WITH replacement) and averages their votes, which
# cancels the per-tree variance and lifts test accuracy. As a free bonus, each tree
# leaves out ~37% of rows (the "out-of-bag" set), giving a validation-free accuracy
# estimate. Everything below (tree growth, bootstrap, voting, OOB) is manual numpy.


class DecisionTree:
    """CART classifier: greedy Gini-minimizing axis-aligned splits."""

    def __init__(self, max_depth=10, min_samples=4):
        self.max_depth, self.min_samples = max_depth, min_samples

    def fit(self, X, y):
        self.n_classes = int(y.max()) + 1
        self.root = self._grow(X, y, 0)
        return self

    def _best_split(self, X, y):
        """Vectorized best (feature, threshold): prefix-sum Gini over sorted values."""
        n, d = X.shape
        Y = np.eye(self.n_classes)[y]          # one-hot labels -> cumulative class counts
        best_gini, best = np.inf, (None, None)
        for f in range(d):
            order = np.argsort(X[:, f], kind="stable")
            xs, ys = X[order, f], Y[order]
            left = np.cumsum(ys, axis=0)        # class counts of rows <= split, all positions
            right = left[-1] - left
            ln = np.arange(1, n + 1)            # #rows on each side
            rn = n - ln
            gl = 1.0 - (left ** 2).sum(1) / np.maximum(ln, 1) ** 2
            gr = 1.0 - (right ** 2).sum(1) / np.maximum(rn, 1) ** 2
            w = (ln * gl + rn * gr) / n          # weighted child impurity per split point
            w[:-1][xs[:-1] == xs[1:]] = np.inf   # can't split between equal values
            w[-1] = np.inf                       # last position = no split
            i = int(np.argmin(w))
            if w[i] < best_gini:
                best_gini, best = w[i], (f, (xs[i] + xs[i + 1]) / 2.0)
        return best

    def _grow(self, X, y, depth):
        node = {"pred": int(np.argmax(np.bincount(y, minlength=self.n_classes)))}
        if depth >= self.max_depth or len(y) < self.min_samples or len(np.unique(y)) == 1:
            return node
        f, thr = self._best_split(X, y)
        if f is None:
            return node
        mask = X[:, f] <= thr
        if mask.all() or not mask.any():
            return node
        node.update(feat=f, thr=thr,
                    left=self._grow(X[mask], y[mask], depth + 1),
                    right=self._grow(X[~mask], y[~mask], depth + 1))
        return node

    def predict(self, X):
        out = np.empty(len(X), dtype=int)
        for i, row in enumerate(X):
            node = self.root
            while "feat" in node:
                node = node["left"] if row[node["feat"]] <= node["thr"] else node["right"]
            out[i] = node["pred"]
        return out


class BaggingClassifier:
    """Ensemble of trees, each fit on a bootstrap sample; predict by majority vote."""

    def __init__(self, n_estimators=40, max_depth=10, min_samples=4, seed=0):
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.min_samples, self.seed = min_samples, seed

    def fit(self, X, y):
        n = len(y)
        self.n_classes = int(y.max()) + 1
        rng = np.random.RandomState(self.seed)
        self.trees, self.oob = [], []
        for _ in range(self.n_estimators):
            idx = rng.randint(0, n, n)          # bootstrap: n draws with replacement
            self.trees.append(DecisionTree(self.max_depth, self.min_samples).fit(X[idx], y[idx]))
            m = np.ones(n, bool)
            m[idx] = False                       # rows never drawn = out-of-bag for this tree
            self.oob.append(m)
        self._X, self._y = X, y
        return self

    def _vote(self, per_tree):
        """per_tree: (B, n) predictions -> majority-vote class per column."""
        votes = np.zeros((per_tree.shape[1], self.n_classes))
        for b in range(per_tree.shape[0]):
            votes[np.arange(per_tree.shape[1]), per_tree[b]] += 1
        return votes.argmax(1)

    def predict(self, X):
        return self._vote(np.stack([t.predict(X) for t in self.trees]))

    def oob_score(self):
        """Accuracy using, for each train row, only the trees that did NOT see it."""
        n = len(self._y)
        votes = np.zeros((n, self.n_classes))
        for t, m in zip(self.trees, self.oob):
            if m.any():
                p = t.predict(self._X[m])
                votes[np.where(m)[0], p] += 1
        seen = votes.sum(1) > 0                  # rows OOB for at least one tree
        return float(np.mean(votes[seen].argmax(1) == self._y[seen]))


if __name__ == "__main__":
    np.random.seed(0)

    # PLANTED STRUCTURE: a curved "inside-the-sphere" boundary on features 0-2
    # (y = 1 iff x0^2+x1^2+x2^2 below its median => balanced classes) plus 3 pure-noise
    # distractor features. A tree can only approximate the smooth boundary with a deep
    # staircase, and 12% flipped labels make that deep tree high-variance -- exactly the
    # setting where averaging bootstrap-resampled trees (bagging) cuts variance and wins.
    R2_MED = 3.0                                 # ~ median of a chi-square(3) => balanced split
    def make_data(n):
        X = np.random.randn(n, 6)
        y = (X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2 < R2_MED).astype(int)
        flip = np.random.rand(n) < 0.12
        return X, (y ^ flip).astype(int)

    X_tr, y_tr = make_data(400)
    X_te, y_te = make_data(3000)                 # large held-out set = true generalization

    majority = max(np.mean(y_te == 0), np.mean(y_te == 1))   # predict-the-majority baseline

    single = DecisionTree(max_depth=10).fit(X_tr, y_tr)      # one high-variance tree
    single_acc = float(np.mean(single.predict(X_te) == y_te))

    bag = BaggingClassifier(n_estimators=40, max_depth=10).fit(X_tr, y_tr)
    bag_acc = float(np.mean(bag.predict(X_te) == y_te))
    oob_acc = bag.oob_score()

    print("Task: sphere boundary (3 informative + 3 noise features), 12% label noise (Bayes acc ~ 0.88)")
    print("Train / test sizes           : {} / {}".format(len(y_tr), len(y_te)))
    print("-" * 60)
    print("Majority-class baseline acc  : {:.3f}".format(majority))
    print("Single decision tree   acc   : {:.3f}".format(single_acc))
    print("Bagging ({:d} trees)     acc   : {:.3f}".format(bag.n_estimators, bag_acc))
    print("Out-of-bag estimate    acc   : {:.3f}   (validation-free, tracks test acc)".format(oob_acc))
    print("Variance reduction: bag beats single tree by {:+.3f}".format(bag_acc - single_acc))

    assert bag_acc > majority + 0.15, "bagging must clearly beat the majority baseline"
    assert bag_acc > single_acc, "bagging must beat the single high-variance tree"
    assert abs(oob_acc - bag_acc) < 0.06, "OOB estimate should track held-out accuracy"
    print("PASS: bagging beat the majority baseline AND the single tree; OOB tracked test acc.")
