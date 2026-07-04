import numpy as np


class LogisticRegression:
    """Linear base learner: batch gradient descent, exposes class-1 probability."""

    def __init__(self, lr=0.3, epochs=500):
        self.lr, self.epochs = lr, epochs

    def _design(self, X):
        return np.hstack([np.ones((len(X), 1)), X])   # prepend bias column

    def fit(self, X, y):
        Xb = self._design(X)
        self.w = np.zeros(Xb.shape[1])
        for _ in range(self.epochs):
            p = self.predict_proba(X)
            self.w -= self.lr * (Xb.T @ (p - y) / len(y))
        return self

    def predict_proba(self, X):
        z = np.clip(self._design(X) @ self.w, -30, 30)   # clip to avoid exp overflow
        return 1.0 / (1.0 + np.exp(-z))


class GaussianNB:
    """Naive Bayes base learner: per-class Gaussians, assumes feature independence."""

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mu, self.var, self.prior = [], [], []
        for c in self.classes:
            Xc = X[y == c]
            self.mu.append(Xc.mean(0))
            self.var.append(Xc.var(0) + 1e-6)            # floor variance for stability
            self.prior.append(len(Xc) / len(X))
        self.mu, self.var, self.prior = map(np.array, (self.mu, self.var, self.prior))
        return self

    def predict_proba(self, X):
        # log p(x|c) + log prior, per class, then softmax over the 2 classes.
        logp = []
        for k in range(len(self.classes)):
            ll = -0.5 * (np.log(2 * np.pi * self.var[k]) + (X - self.mu[k]) ** 2 / self.var[k])
            logp.append(ll.sum(1) + np.log(self.prior[k]))
        logp = np.array(logp).T
        logp -= logp.max(1, keepdims=True)
        p = np.exp(logp)
        return (p / p.sum(1, keepdims=True))[:, 1]


class KNN:
    """Non-linear base learner: class-1 probability = fraction of k nearest neighbours."""

    def __init__(self, k=15):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = X, y
        return self

    def predict_proba(self, X):
        d = ((X[:, None, :] - self.X[None, :, :]) ** 2).sum(2)   # squared euclidean
        nn = np.argsort(d, axis=1)[:, :self.k]                   # k nearest train indices
        return self.y[nn].mean(1)


class FeatureView:
    """Restrict a base learner to a subset of columns, giving each learner a partial
    VIEW of the data so their prediction errors decorrelate (which is what lets the
    meta-learner fuse them into something better than any single view)."""

    def __init__(self, base, cols):
        self.base, self.cols = base, cols

    def fit(self, X, y):
        self.base.fit(X[:, self.cols], y)
        return self

    def predict_proba(self, X):
        return self.base.predict_proba(X[:, self.cols])


def kfold(n, k, seed=0):
    idx = np.random.RandomState(seed).permutation(n)
    return [np.sort(f) for f in np.array_split(idx, k)]


class StackingClassifier:
    """Stack base learners: a meta-learner is trained on their OUT-OF-FOLD predictions,
    so it learns how to weight/combine each base without leaking labels."""

    def __init__(self, make_bases, make_meta, k=5):
        self.make_bases, self.make_meta, self.k = make_bases, make_meta, k

    def fit(self, X, y):
        self.bases = self.make_bases()
        n = len(y)
        # Out-of-fold predictions form clean meta-features (each row predicted by a model
        # that never saw it during training).
        meta = np.zeros((n, len(self.bases)))
        for test_idx in kfold(n, self.k):
            mask = np.ones(n, dtype=bool)
            mask[test_idx] = False
            for j, base in enumerate(self.make_bases()):
                base.fit(X[mask], y[mask])
                meta[test_idx, j] = base.predict_proba(X[test_idx])
        # Refit each base on ALL data for test-time use, then train the meta-learner.
        for base in self.bases:
            base.fit(X, y)
        self.meta = self.make_meta().fit(meta, y)
        return self

    def _base_matrix(self, X):
        return np.column_stack([b.predict_proba(X) for b in self.bases])

    def predict_proba(self, X):
        return self.meta.predict_proba(self._base_matrix(X))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


if __name__ == "__main__":
    np.random.seed(0)

    # PLANTED STRUCTURE: the label mixes a NON-LINEAR interaction (x0*x1, invisible to a
    # linear model) with two LINEAR signals living in other columns. We hand each base a
    # different 2-feature VIEW, so no single base sees the whole picture -> each is only
    # ~0.65-0.73 accurate, with decorrelated errors the meta-learner can exploit.
    def make_data(n):
        X = np.random.randn(n, 6)
        logit = (3.0 * X[:, 0] * X[:, 1]                  # non-linear: only KNN can model it
                 + 1.7 * X[:, 2] + 1.7 * X[:, 3]          # linear signals for the LR views
                 - 1.6 * X[:, 4]
                 + 0.5 * np.random.randn(n))
        y = (logit > 0).astype(int)
        return X, y

    Xtr, ytr = make_data(500)
    Xte, yte = make_data(4000)                            # large held-out set = ground truth

    # Diverse base learners (three model families), each on a different partial view.
    make_bases = lambda: [FeatureView(KNN(k=25), [0, 1]),             # non-linear view
                          FeatureView(GaussianNB(), [2, 3]),          # linear view A
                          FeatureView(LogisticRegression(), [3, 4])]  # linear view B
    names = ["KNN[0,1]", "GaussNB[2,3]", "LR[3,4]"]

    # Baseline 1: predict the majority class.
    majority = max(yte.mean(), 1 - yte.mean())

    # Baseline 2: the BEST single base learner (what stacking must beat).
    base_acc = {}
    for name, base in zip(names, make_bases()):
        base.fit(Xtr, ytr)
        base_acc[name] = float(np.mean(base.predict_proba(Xte).round() == yte))

    stack = StackingClassifier(make_bases, lambda: LogisticRegression(), k=5).fit(Xtr, ytr)
    stack_acc = float(np.mean(stack.predict(Xte) == yte))
    best_base = max(base_acc.values())

    print("Held-out test size            : {}".format(len(yte)))
    print("Majority-class baseline acc   : {:.3f}".format(majority))
    print("-" * 46)
    for name in base_acc:
        print("Base learner {:12s} acc  : {:.3f}".format(name, base_acc[name]))
    print("Best single base learner acc  : {:.3f}".format(best_base))
    print("-" * 46)
    print("STACKED ensemble accuracy     : {:.3f}".format(stack_acc))
    print("Lift over best single base    : {:+.3f}".format(stack_acc - best_base))
    print("Lift over majority baseline   : {:+.3f}".format(stack_acc - majority))

    assert stack_acc > majority + 0.15, "stack must clearly beat majority baseline"
    assert stack_acc > best_base + 0.01, "stack must beat the best single base learner"
    print("PASS: stacking combined diverse base learners and beat both the majority"
          " baseline and every individual base learner on held-out data.")
