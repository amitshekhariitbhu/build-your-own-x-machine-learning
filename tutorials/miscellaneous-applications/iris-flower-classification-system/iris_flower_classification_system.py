import numpy as np


class IrisClassifier:
    """Softmax (multinomial logistic) regression for the 3 iris species, from scratch.

    Standardizes features, trains a K-way softmax by full-batch gradient descent on
    the cross-entropy loss, and predicts the argmax class probability.
    """

    def __init__(self, lr=0.2, n_iter=600, l2=1e-3):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2  # ridge penalty to keep weights sane
        self.W = None
        self.b = None
        self.mu = None
        self.sigma = None
        self.n_classes = None

    def _standardize(self, X, fit=False):
        if fit:
            self.mu = X.mean(axis=0)
            self.sigma = X.std(axis=0) + 1e-8
        return (X - self.mu) / self.sigma

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)  # stabilize
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        X = self._standardize(np.asarray(X, dtype=float), fit=True)
        y = np.asarray(y, dtype=int).reshape(-1)
        n, d = X.shape
        self.n_classes = int(y.max()) + 1
        Y = np.eye(self.n_classes)[y]  # one-hot targets
        self.W = np.zeros((d, self.n_classes))
        self.b = np.zeros(self.n_classes)
        for _ in range(self.n_iter):
            P = self._softmax(X @ self.W + self.b)  # class probabilities
            err = P - Y
            grad_W = X.T @ err / n + self.l2 * self.W  # cross-entropy + ridge grad
            grad_b = err.mean(axis=0)
            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        X = self._standardize(np.asarray(X, dtype=float))
        return self._softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


SPECIES = ["setosa", "versicolor", "virginica"]


def make_iris_data(n_per=60, seed=0):
    """Synthetic iris measurements with three planted, partially-overlapping species.

    Feature order: sepal length, sepal width, petal length, petal width (cm).
    Petal dimensions grow strongly from setosa -> versicolor -> virginica, which is
    the real latent structure that separates the classes.
    """
    rng = np.random.RandomState(seed)
    means = np.array([
        [5.0, 3.4, 1.5, 0.25],   # setosa: tiny petals
        [5.9, 2.8, 4.3, 1.35],   # versicolor: medium petals
        [6.6, 3.0, 5.6, 2.05],   # virginica: large petals
    ])
    stds = np.array([0.35, 0.32, 0.30, 0.22])  # per-feature spread
    X = np.vstack([rng.normal(m, stds, size=(n_per, 4)) for m in means])
    y = np.repeat(np.arange(3), n_per)
    perm = rng.permutation(len(y))
    return X[perm], y[perm].astype(int)


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true, y_pred, k):
    f1s = []
    for c in range(k):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_iris_data(n_per=60, seed=0)

    # held-out split: 70% train / 30% test (stratified via shuffled repeats)
    n_train = int(0.7 * len(y))
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]

    clf = IrisClassifier(lr=0.2, n_iter=600, l2=1e-3).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy(yte, pred)
    f1 = macro_f1(yte, pred, clf.n_classes)

    # majority-class baseline
    majority = int(np.bincount(ytr).argmax())
    base_pred = np.full_like(yte, majority)
    base_acc = accuracy(yte, base_pred)
    base_f1 = macro_f1(yte, base_pred, clf.n_classes)

    print("=== Iris Flower Classification (softmax, from scratch) ===")
    print(f"classes               : {', '.join(SPECIES)}")
    print(f"train / test samples  : {len(ytr)} / {len(yte)}")
    print(f"baseline (majority)   : acc={base_acc:.3f}  macro-f1={base_f1:.3f}")
    print(f"softmax regression    : acc={acc:.3f}  macro-f1={f1:.3f}")
    print("per-class test counts :", {SPECIES[c]: int(np.sum(yte == c)) for c in range(3)})
    print(f"improvement over base : +{(acc - base_acc) * 100:.1f} acc points")
    assert acc > base_acc + 0.30, "model should clearly beat the majority baseline"
    print("PASS: classifier beats majority baseline")
