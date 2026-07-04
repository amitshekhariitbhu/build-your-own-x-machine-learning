import numpy as np


class BreastCancerClassifier:
    """Logistic-regression tumor classifier (benign=0 / malignant=1) from scratch.

    Trained by full-batch gradient descent on standardized features; predicts
    malignancy probability via the sigmoid and thresholds at 0.5.
    """

    def __init__(self, lr=0.1, n_iter=800, l2=1e-3):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2  # ridge penalty to keep weights sane
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    def _standardize(self, X, fit=False):
        if fit:
            self.mu = X.mean(axis=0)
            self.sigma = X.std(axis=0) + 1e-8
        return (X - self.mu) / self.sigma

    def fit(self, X, y):
        X = self._standardize(np.asarray(X, dtype=float), fit=True)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iter):
            p = self._sigmoid(X @ self.w + self.b)  # predicted probabilities
            err = p - y
            grad_w = X.T @ err / n + self.l2 * self.w  # cross-entropy + ridge grad
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        X = self._standardize(np.asarray(X, dtype=float))
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def make_tumor_data(n=600, seed=0):
    """Synthetic diagnostic features with two planted, overlapping classes."""
    rng = np.random.RandomState(seed)
    n_mal = n // 2
    n_ben = n - n_mal
    # feature order: radius, texture, perimeter, area, concavity, symmetry
    # malignant tumors are larger / more irregular; benign are smaller / smoother
    mal_mean = np.array([17.5, 21.0, 115.0, 970.0, 0.16, 0.21])
    ben_mean = np.array([12.5, 18.0, 80.0, 480.0, 0.06, 0.17])
    cov = np.array([3.0, 4.0, 20.0, 180.0, 0.04, 0.03])  # per-feature std
    mal = rng.normal(mal_mean, cov, size=(n_mal, 6))
    ben = rng.normal(ben_mean, cov, size=(n_ben, 6))
    X = np.vstack([mal, ben])
    y = np.concatenate([np.ones(n_mal), np.zeros(n_ben)])
    perm = rng.permutation(n)
    return X[perm], y[perm].astype(int)


def metrics(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return acc, prec, rec, f1


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_tumor_data(n=600, seed=0)

    # held-out split: 70% train / 30% test
    n_train = int(0.7 * len(y))
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]

    clf = BreastCancerClassifier(lr=0.1, n_iter=800, l2=1e-3).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc, prec, rec, f1 = metrics(yte, pred)

    # majority-class baseline
    majority = int(round(ytr.mean()))
    base_pred = np.full_like(yte, majority)
    base_acc, _, _, base_f1 = metrics(yte, base_pred)

    print("=== Breast Cancer Classification (from scratch) ===")
    print(f"test samples          : {len(yte)}")
    print(f"malignant in test     : {int(yte.sum())} / {len(yte)}")
    print(f"baseline (majority)   : acc={base_acc:.3f}  f1={base_f1:.3f}")
    print(f"logistic regression   : acc={acc:.3f}  f1={f1:.3f}")
    print(f"  precision={prec:.3f}  recall={rec:.3f}")
    print(f"improvement over base : +{(acc - base_acc) * 100:.1f} acc points")
    assert acc > base_acc + 0.15, "model should clearly beat the majority baseline"
    print("PASS: classifier beats majority baseline")
