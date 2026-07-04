import numpy as np


class DiabetesPredictor:
    """Logistic-regression diabetes classifier trained from scratch by gradient descent.

    Standardizes features, learns weights + bias via full-batch gradient descent on
    the binary cross-entropy loss, and predicts the probability / label of diabetes.
    """

    def __init__(self, lr=0.1, n_iter=2000, l2=0.0):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2  # optional L2 regularization strength
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
        # feature scaling stats from the training set only
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


def make_diabetes_data(n=800):
    """Synthetic patient records with a planted diabetes signal.

    Features (clinically inspired): glucose, bmi, age, blood_pressure, insulin.
    A true linear-logit rule (mainly glucose + bmi) decides diabetes, plus noise.
    """
    glucose = np.random.normal(120, 30, n)
    bmi = np.random.normal(32, 7, n)
    age = np.random.normal(45, 12, n)
    blood_pressure = np.random.normal(72, 12, n)
    insulin = np.random.normal(80, 40, n)
    X = np.column_stack([glucose, bmi, age, blood_pressure, insulin])

    # planted logit: diabetes driven by high glucose, high bmi, older age
    z = (0.05 * (glucose - 120) + 0.12 * (bmi - 32) + 0.03 * (age - 45)
         + 0.01 * (blood_pressure - 72))
    prob = 1.0 / (1.0 + np.exp(-z))
    y = (prob + np.random.normal(0, 0.15, n) > 0.5).astype(int)
    return X, y


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_diabetes_data(800)

    # held-out train/test split
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    model = DiabetesPredictor(lr=0.2, n_iter=3000, l2=1e-3).fit(Xtr, ytr)
    pred = model.predict(Xte)

    acc = np.mean(pred == yte)
    # majority-class baseline
    majority = int(round(ytr.mean()))
    base_acc = np.mean(yte == majority)

    # precision / recall / F1 on the positive (diabetic) class
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
    print("learned weights       :", np.round(model.w, 3))
    print("BEATS baseline        :", bool(acc > base_acc))
