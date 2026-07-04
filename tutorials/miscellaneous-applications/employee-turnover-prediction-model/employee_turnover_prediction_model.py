import numpy as np


class TurnoverPredictor:
    """Logistic-regression employee-attrition classifier trained from scratch.

    Standardizes HR features, learns weights + bias by full-batch gradient
    descent on the L2-regularized binary cross-entropy loss, and predicts the
    probability / label of an employee leaving. No ML libraries -- just numpy.
    """

    def __init__(self, lr=0.2, n_iter=3000, l2=1e-3):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2  # L2 regularization strength
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
        # scaling stats from training data only (no test leakage)
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


def make_hr_data(n=1000):
    """Synthetic employee records with a planted attrition signal.

    Features (HR-inspired):
      satisfaction, last_evaluation, monthly_hours, tenure_years,
      salary_level (0=low..2=high), promotion_last_5yrs, distance_km.
    A true linear-logit rule decides who leaves, plus label noise so the
    problem is not perfectly separable.
    """
    satisfaction = np.clip(np.random.normal(0.62, 0.24, n), 0, 1)
    last_eval = np.clip(np.random.normal(0.72, 0.17, n), 0, 1)
    monthly_hours = np.random.normal(200, 48, n)
    tenure = np.random.gamma(2.0, 1.6, n)                # years at company
    salary_level = np.random.randint(0, 3, n)            # 0 low, 1 med, 2 high
    promoted = np.random.binomial(1, 0.12, n)            # promoted in last 5 yrs
    distance = np.abs(np.random.normal(11, 8, n))        # commute distance km
    X = np.column_stack([satisfaction, last_eval, monthly_hours, tenure,
                         salary_level, promoted, distance])

    # planted logit: leaving rises with low satisfaction, extreme hours,
    # longer commute; falls with higher salary and a recent promotion.
    z = (-2.6 * (satisfaction - 0.62) + 0.9 * (last_eval - 0.72)
         + 0.012 * (monthly_hours - 200) + 0.10 * (tenure - 3.2)
         - 0.85 * (salary_level - 1) - 1.1 * promoted
         + 0.045 * (distance - 11))
    prob = 1.0 / (1.0 + np.exp(-z))
    y = (prob + np.random.normal(0, 0.14, n) > 0.5).astype(int)
    return X, y


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_hr_data(1000)

    # held-out train/test split
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    model = TurnoverPredictor(lr=0.2, n_iter=3000, l2=1e-3).fit(Xtr, ytr)
    pred = model.predict(Xte)

    acc = np.mean(pred == yte)
    # majority-class baseline (predict the most common training label)
    majority = int(round(ytr.mean()))
    base_acc = np.mean(yte == majority)

    # precision / recall / F1 on the positive (leaver) class
    tp = np.sum((pred == 1) & (yte == 1))
    fp = np.sum((pred == 1) & (yte == 0))
    fn = np.sum((pred == 0) & (yte == 1))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    feats = ["satisfaction", "last_eval", "monthly_hours", "tenure",
             "salary_level", "promoted", "distance"]

    print("test samples          :", len(yte))
    print("attrition rate (test) :", round(float(yte.mean()), 3))
    print("majority baseline acc :", round(float(base_acc), 3))
    print("model accuracy        :", round(float(acc), 3))
    print("precision / recall    :", round(float(precision), 3), "/", round(float(recall), 3))
    print("F1 score              :", round(float(f1), 3))
    print("feature weights       :", dict(zip(feats, np.round(model.w, 2))))
    print("BEATS baseline        :", bool(acc > base_acc))
