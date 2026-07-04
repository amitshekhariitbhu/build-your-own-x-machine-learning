import numpy as np

# Detect Parkinson's disease from sustained-vowel VOICE biomarkers.
# Parkinson's degrades vocal-fold control, so the recorded voice shows more
# cycle-to-cycle pitch variation (jitter), amplitude variation (shimmer),
# pitch-period entropy (PPE) and less harmonic structure (lower HNR). We plant
# exactly that structure into synthetic data and train a from-scratch,
# standardized logistic-regression classifier to recover the diagnosis.

FEATURES = ["MDVP_Fo", "Jitter", "Shimmer", "HNR", "RPDE", "DFA", "spread1", "PPE"]


def make_voice_data(n=600, seed=0):
    # Each row is one voice recording. Healthy (0) and Parkinson's (1) speakers
    # draw features from Gaussians whose means differ along the biomarkers that
    # medicine links to the disease; overlap keeps the task non-trivial.
    rng = np.random.RandomState(seed)
    # class means for the 8 features (healthy vs parkinson's)
    mu_healthy = np.array([180.0, 0.004, 0.020, 24.0, 0.42, 0.72, -6.5, 0.12])
    mu_park    = np.array([150.0, 0.009, 0.045, 18.0, 0.55, 0.78, -5.2, 0.22])
    sigma      = np.array([22.0, 0.0028, 0.014, 4.0, 0.07, 0.05, 0.9, 0.06])

    # ~55% positive, matching the UCI Parkinson's class skew.
    y = (rng.rand(n) < 0.55).astype(int)
    means = np.where(y[:, None] == 1, mu_park, mu_healthy)
    X = means + rng.randn(n, len(FEATURES)) * sigma
    X[:, 1:3] = np.abs(X[:, 1:3])                      # jitter/shimmer >= 0
    return X, y


class StandardScaler:
    """Zero-mean/unit-variance scaling; fit on train, apply to any split."""

    def fit(self, X):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-9
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd


class LogisticRegression:
    """L2-regularized logistic regression via full-batch gradient descent.

    p(parkinson's | x) = sigmoid(x . w + b). The gradient of the mean binary
    cross-entropy w.r.t. the logit is (p - y), giving a compact update rule.
    """

    def __init__(self, lr=0.2, epochs=400, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = self._proba(X)
            err = p - y                                # dBCE/dlogit
            self.w -= self.lr * (X.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def _proba(self, X):
        z = X @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict_proba(self, X):
        return self._proba(X)

    def predict(self, X):
        return (self._proba(X) >= 0.5).astype(int)


def roc_auc(y_true, scores):
    # AUC = P(score_pos > score_neg) via the Mann-Whitney rank statistic.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = y_true == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def prf1(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_voice_data(n=600, seed=0)

    # Held-out split: fit the detector on 70%, judge it on the untouched 30%.
    idx = np.random.permutation(len(X))
    split = int(0.7 * len(X))
    tr, te = idx[:split], idx[split:]
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    model = LogisticRegression(lr=0.2, epochs=400, l2=1e-3).fit(Xtr_s, ytr)

    p_te = model.predict_proba(Xte_s)
    pred = (p_te >= 0.5).astype(int)
    acc = np.mean(pred == yte)
    prec, rec, f1 = prf1(yte, pred)
    auc = roc_auc(yte, p_te)

    # Baseline: always predict the majority class (no voice information used).
    majority = int(ytr.mean() >= 0.5)
    base_pred = np.full_like(yte, majority)
    base_acc = np.mean(base_pred == yte)
    _, _, base_f1 = prf1(yte, base_pred)

    print("Recordings: %d   Train: %d   Test: %d   Features: %d"
          % (len(X), len(tr), len(te), len(FEATURES)))
    print("Prevalence (Parkinson's): %.3f" % y.mean())
    print("-" * 60)
    print("Detector   acc: %.4f   F1: %.4f   AUC: %.4f" % (acc, f1, auc))
    print("Baseline   acc: %.4f   F1: %.4f   AUC: %.4f" % (base_acc, base_f1, 0.5))
    print("-" * 60)
    order = np.argsort(np.abs(model.w))[::-1][:4]
    print("Most predictive voice biomarkers:",
          [FEATURES[j] for j in order])
    print("-" * 60)
    print("Detector beats baseline: %s"
          % (acc > base_acc and f1 > base_f1 and auc > 0.5))
