import numpy as np


class BloodDonationPredictor:
    """Logistic-regression blood-donation classifier trained from scratch.

    Uses the classic RFMT donor features (Recency, Frequency, Monetary, Time),
    standardizes them, and learns weights + bias by full-batch gradient descent
    on the L2-regularized binary cross-entropy loss. Predicts the probability /
    label that a donor will give blood in the target campaign. No ML libraries.
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


def make_donor_data(n=900):
    """Synthetic donor records with a planted donation signal (RFMT features).

    Features (transfusion-center inspired):
      recency  : months since last donation (lower = more likely to give again)
      frequency: total number of past donations (higher = loyal donor)
      monetary : total blood volume donated in c.c. (~250 c.c. per donation)
      time     : months since first donation (donor lifetime)
    A true linear-logit rule decides who donates in the campaign, plus label
    noise so the problem is not perfectly separable.
    """
    frequency = np.random.poisson(5, n) + 1               # donations so far
    monetary = frequency * 250.0                          # 250 c.c. each
    recency = np.abs(np.random.normal(10, 8, n))          # months since last
    time = recency + np.random.gamma(2.0, 9.0, n)         # donor lifetime months
    X = np.column_stack([recency, frequency, monetary, time])

    # planted logit: donating again rises with recent + frequent giving,
    # falls as recency grows; monetary tracks frequency so weight is shared.
    z = (-0.18 * (recency - 10) + 0.42 * (frequency - 6)
         + 0.0006 * (monetary - 1500) - 0.02 * (time - 25))
    prob = 1.0 / (1.0 + np.exp(-z))
    y = (prob + np.random.normal(0, 0.13, n) > 0.5).astype(int)
    return X, y


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_donor_data(900)

    # held-out train/test split
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    model = BloodDonationPredictor(lr=0.2, n_iter=3000, l2=1e-3).fit(Xtr, ytr)
    pred = model.predict(Xte)

    acc = np.mean(pred == yte)
    # majority-class baseline (predict the most common training label)
    majority = int(round(ytr.mean()))
    base_acc = np.mean(yte == majority)

    # precision / recall / F1 on the positive (will-donate) class
    tp = np.sum((pred == 1) & (yte == 1))
    fp = np.sum((pred == 1) & (yte == 0))
    fn = np.sum((pred == 0) & (yte == 1))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    feats = ["recency", "frequency", "monetary", "time"]

    print("test samples          :", len(yte))
    print("donation rate (test)  :", round(float(yte.mean()), 3))
    print("majority baseline acc :", round(float(base_acc), 3))
    print("model accuracy        :", round(float(acc), 3))
    print("precision / recall    :", round(float(precision), 3), "/", round(float(recall), 3))
    print("F1 score              :", round(float(f1), 3))
    print("feature weights       :", dict(zip(feats, np.round(model.w, 3))))
    print("BEATS baseline        :", bool(acc > base_acc))
