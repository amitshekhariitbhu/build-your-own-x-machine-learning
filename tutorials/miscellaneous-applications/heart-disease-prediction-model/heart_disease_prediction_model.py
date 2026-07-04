import numpy as np


class HeartDiseasePredictor:
    """Logistic-regression heart-disease classifier trained from scratch.

    Standardizes clinical features, learns weights + bias by full-batch gradient
    descent on the L2-regularized binary cross-entropy loss, and predicts the
    probability / label of heart disease. No ML libraries -- just numpy math.
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
        # scaling stats computed on training data only (no test leakage)
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


def make_heart_data(n=900):
    """Synthetic patient records with a planted heart-disease signal.

    Features (clinically inspired):
      age, sex, chest_pain, resting_bp, cholesterol, max_heart_rate, oldpeak.
    A true linear-logit rule decides disease, plus label noise so it is not
    perfectly separable.
    """
    age = np.random.normal(54, 9, n)
    sex = np.random.binomial(1, 0.68, n)                 # 1 = male
    chest_pain = np.random.randint(0, 4, n)              # 0..3, higher = worse
    resting_bp = np.random.normal(131, 17, n)
    cholesterol = np.random.normal(246, 52, n)
    max_hr = np.random.normal(150, 23, n)                # peak heart rate achieved
    oldpeak = np.abs(np.random.normal(1.0, 1.0, n))      # ST depression
    X = np.column_stack([age, sex, chest_pain, resting_bp,
                         cholesterol, max_hr, oldpeak])

    # planted logit: risk rises with age, male sex, chest pain, bp, cholesterol,
    # ST depression, and FALLS with a higher achievable max heart rate.
    z = (0.045 * (age - 54) + 0.60 * sex + 0.55 * (chest_pain - 1.5)
         + 0.020 * (resting_bp - 131) + 0.010 * (cholesterol - 246)
         - 0.030 * (max_hr - 150) + 0.70 * (oldpeak - 1.0))
    prob = 1.0 / (1.0 + np.exp(-z))
    y = (prob + np.random.normal(0, 0.15, n) > 0.5).astype(int)
    return X, y


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_heart_data(900)

    # held-out train/test split
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    model = HeartDiseasePredictor(lr=0.2, n_iter=3000, l2=1e-3).fit(Xtr, ytr)
    pred = model.predict(Xte)

    acc = np.mean(pred == yte)
    # majority-class baseline (predict the most common training label)
    majority = int(round(ytr.mean()))
    base_acc = np.mean(yte == majority)

    # precision / recall / F1 on the positive (disease) class
    tp = np.sum((pred == 1) & (yte == 1))
    fp = np.sum((pred == 1) & (yte == 0))
    fn = np.sum((pred == 0) & (yte == 1))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    feats = ["age", "sex", "chest_pain", "resting_bp",
             "cholesterol", "max_hr", "oldpeak"]

    print("test samples          :", len(yte))
    print("positive rate (test)  :", round(float(yte.mean()), 3))
    print("majority baseline acc :", round(float(base_acc), 3))
    print("model accuracy        :", round(float(acc), 3))
    print("precision / recall    :", round(float(precision), 3), "/", round(float(recall), 3))
    print("F1 score              :", round(float(f1), 3))
    print("feature weights       :", dict(zip(feats, np.round(model.w, 2))))
    print("BEATS baseline        :", bool(acc > base_acc))
