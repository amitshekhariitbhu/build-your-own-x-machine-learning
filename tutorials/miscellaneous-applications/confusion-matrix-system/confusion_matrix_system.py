import numpy as np


class ConfusionMatrixSystem:
    """Build a confusion matrix and derive classification metrics from scratch."""

    def __init__(self, n_classes=None):
        self.n_classes = n_classes

    def fit(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        k = self.n_classes or int(max(y_true.max(), y_pred.max()) + 1)
        self.n_classes = k

        # Tally: rows = true class, cols = predicted class (vectorized).
        M = np.zeros((k, k), dtype=int)
        np.add.at(M, (y_true, y_pred), 1)
        self.matrix_ = M
        self._derive()
        return self

    def _derive(self):
        M = self.matrix_.astype(float)
        total = M.sum()

        # Per-class counts read straight off the matrix.
        tp = np.diag(M)                 # correct predictions for the class
        fp = M.sum(axis=0) - tp         # predicted as class, but truly other
        fn = M.sum(axis=1) - tp         # truly class, but predicted other
        tn = total - tp - fp - fn
        support = M.sum(axis=1)         # number of true samples per class

        # Guarded ratios (zero when the denominator is empty).
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        denom = precision + recall
        f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

        self.tp_, self.fp_, self.fn_, self.tn_ = tp, fp, fn, tn
        self.precision_, self.recall_, self.f1_, self.support_ = precision, recall, f1, support
        self.accuracy_ = tp.sum() / total

        # Averages: macro = unweighted, weighted = by support, micro == accuracy.
        w = support / support.sum()
        self.macro_f1_ = f1.mean()
        self.weighted_f1_ = (f1 * w).sum()
        self.micro_f1_ = self.accuracy_

    def report(self):
        print("Confusion matrix (rows=true, cols=pred):")
        print(self.matrix_)
        print("\nclass  precision  recall     f1  support")
        for c in range(self.n_classes):
            print("{:>5} {:>10.3f} {:>7.3f} {:>6.3f} {:>8}".format(
                c, self.precision_[c], self.recall_[c], self.f1_[c], int(self.support_[c])))
        print("\naccuracy (micro-F1): {:.4f}".format(self.accuracy_))
        print("macro-F1:            {:.4f}".format(self.macro_f1_))
        print("weighted-F1:         {:.4f}".format(self.weighted_f1_))


class NearestCentroid:
    """Tiny from-scratch classifier used to generate realistic predictions."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroids_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X):
        # Assign each point to the nearest class centroid (squared Euclidean).
        d = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return self.classes_[d.argmin(axis=1)]


if __name__ == "__main__":
    np.random.seed(0)

    # --- 1) Exact hand-verifiable check on a tiny labelled example ---------
    y_true = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2, 2, 0, 2])
    expected = np.array([[2, 1, 0], [0, 1, 1], [1, 0, 3]])   # worked out by hand
    cm = ConfusionMatrixSystem().fit(y_true, y_pred)
    ok = np.array_equal(cm.matrix_, expected)
    print("Hand-check confusion matrix matches expected:", ok)
    print("Hand-check accuracy 6/9:", np.isclose(cm.accuracy_, 6 / 9), "\n")
    assert ok and np.isclose(cm.accuracy_, 6 / 9)

    # --- 2) Planted 3-class data: imbalanced Gaussian blobs ---------------
    means = np.array([[0.0, 0.0], [4.0, 4.0], [-4.0, 3.0]])
    sizes = [300, 200, 100]                       # class 0 is the majority
    X = np.vstack([np.random.randn(n, 2) * 1.3 + means[c] for c, n in enumerate(sizes)])
    y = np.concatenate([np.full(n, c) for c, n in enumerate(sizes)])

    # Shuffle, then hold out 40% for evaluation.
    perm = np.random.permutation(len(y))
    X, y = X[perm], y[perm]
    split = int(0.6 * len(y))
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]

    y_hat = NearestCentroid().fit(Xtr, ytr).predict(Xte)

    cm = ConfusionMatrixSystem(n_classes=3).fit(yte, y_hat)
    cm.report()

    # Baseline: always predict the majority (most frequent training) class.
    majority = np.bincount(ytr).argmax()
    baseline_acc = (yte == majority).mean()
    print("\nmajority-class baseline accuracy: {:.4f}".format(baseline_acc))
    print("classifier accuracy:              {:.4f}".format(cm.accuracy_))
    print("beats baseline:", cm.accuracy_ > baseline_acc + 0.2)
