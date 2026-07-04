import numpy as np


class HypothyroidPredictor:
    """Logistic-regression hypothyroidism classifier trained from scratch.

    Standardizes thyroid-panel features, learns weights + bias via full-batch
    gradient descent on the binary cross-entropy loss (with optional L2), and
    predicts the probability / label of hypothyroidism.
    """

    def __init__(self, lr=0.3, n_iter=3000, l2=1e-3):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _standardize(self, X):
        return (X - self.mu) / self.sigma

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        # scaling stats from the training set only
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xs = self._standardize(X)
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iter):
            p = self._sigmoid(Xs @ self.w + self.b)  # predicted probabilities
            err = p - y
            grad_w = Xs.T @ err / n + self.l2 * self.w
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        Xs = self._standardize(np.asarray(X, dtype=float))
        return self._sigmoid(Xs @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def make_thyroid_data(n=900):
    """Synthetic thyroid-panel records with a planted hypothyroid signal.

    Features (clinically inspired): TSH, T3, TT4, T4U, FTI, age.
    Hypothyroidism is driven mainly by HIGH TSH and LOW T4/FTI, so the
    true logit rewards high TSH and penalizes T4-related hormones.
    """
    age = np.random.normal(50, 15, n)
    tsh = np.abs(np.random.normal(3.5, 4.0, n))   # elevated in hypothyroidism
    t3 = np.random.normal(2.0, 0.6, n)            # low in hypothyroidism
    tt4 = np.random.normal(110, 30, n)            # low in hypothyroidism
    t4u = np.random.normal(1.0, 0.2, n)
    fti = np.random.normal(110, 30, n)            # low in hypothyroidism
    X = np.column_stack([age, tsh, t3, tt4, t4u, fti])

    # planted logit: high TSH + low hormones => hypothyroid
    z = (0.9 * (tsh - 3.5) - 0.6 * (t3 - 2.0)
         - 0.04 * (tt4 - 110) - 0.03 * (fti - 110) + 0.01 * (age - 50) - 1.0)
    prob = 1.0 / (1.0 + np.exp(-z))
    y = (prob + np.random.normal(0, 0.12, n) > 0.5).astype(int)
    return X, y


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_thyroid_data(900)

    # held-out train/test split
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    model = HypothyroidPredictor(lr=0.3, n_iter=3000, l2=1e-3).fit(Xtr, ytr)
    pred = model.predict(Xte)

    acc = np.mean(pred == yte)
    # majority-class baseline
    majority = int(round(ytr.mean()))
    base_acc = np.mean(yte == majority)

    # precision / recall / F1 on the positive (hypothyroid) class
    tp = np.sum((pred == 1) & (yte == 1))
    fp = np.sum((pred == 1) & (yte == 0))
    fn = np.sum((pred == 0) & (yte == 1))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print("test samples          :", len(yte))
    print("positive rate (test)  :", round(float(yte.mean()), 3))
    print("majority baseline acc :", round(float(base_acc), 3))
    print("model accuracy        :", round(float(acc), 3))
    print("precision / recall    :", round(float(precision), 3), "/", round(float(recall), 3))
    print("F1 score              :", round(float(f1), 3))
    print("feature weights       :", np.round(model.w, 3))
    print("  (age,TSH,T3,TT4,T4U,FTI)")
    print("BEATS baseline        :", bool(acc > base_acc))
