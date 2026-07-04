import numpy as np

class LogisticRegression:
    """Binary logistic regression from scratch via full-batch gradient descent."""
    def __init__(self, lr=0.1, epochs=600, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = self._sigmoid(X @ self.w + self.b)      # predicted churn probs
            g = p - y                                    # gradient of log-loss wrt logit
            self.w -= self.lr * (X.T @ g / n + self.l2 * self.w)
            self.b -= self.lr * g.mean()
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X, thresh=0.5):
        return (self.predict_proba(X) >= thresh).astype(int)


def churn_features(series):
    """Turn each customer's monthly-activity time series into behavior features:
    level, recent vs early usage, linear trend slope, volatility, decay signals."""
    T = series.shape[1]
    t = np.arange(T)
    tc = t - t.mean()
    slope = (series @ tc) / (tc @ tc)                    # per-customer OLS trend slope
    first3 = series[:, :3].mean(1)
    last3 = series[:, -3:].mean(1)
    eps = 1e-6
    feats = np.column_stack([
        series.mean(1),                                  # average activity
        first3,                                          # early activity
        last3,                                           # recent activity
        slope,                                           # usage trend (key churn signal)
        series.std(1),                                   # volatility
        last3 / (first3 + eps),                          # recent/early ratio
        series.min(1),                                   # worst month
        (series < 0.5 * first3[:, None]).mean(1),        # fraction of low-usage months
    ])
    return feats


def make_data(n=700, T=12, churn_rate=0.35, seed=0):
    """Synthetic customer base: each customer has T months of usage. Churners have a
    declining usage trend; loyal customers stay flat/rising. Label = did they churn."""
    rng = np.random.RandomState(seed)
    y = (rng.rand(n) < churn_rate).astype(int)           # planted churn membership
    base = rng.uniform(20, 60, n)                         # baseline monthly usage
    amp = rng.uniform(0, 5, n)                            # seasonal amplitude
    sigma = rng.uniform(2, 5, n)                          # per-customer noise level
    # Loyal: flat/positive slope; churner: negative slope (usage decays away).
    slope = np.where(y == 1, -rng.uniform(0.4, 1.2, n), rng.uniform(-0.1, 0.4, n))
    t = np.arange(T)
    season = np.sin(2 * np.pi * t / 6.0)                  # shared seasonal pattern
    noise = rng.randn(n, T) * sigma[:, None]
    X = base[:, None] + slope[:, None] * t[None, :] + amp[:, None] * season[None, :] + noise
    X = np.clip(X, 0, None)
    # Inject label noise so the problem is not perfectly separable.
    flip = rng.rand(n) < 0.05
    y = np.where(flip, 1 - y, y)
    return X, y


def standardize(train, test):
    mu, sd = train.mean(0), train.std(0) + 1e-9
    return (train - mu) / sd, (test - mu) / sd


def auc_score(y, p):
    # Mann-Whitney rank statistic = probability a random positive ranks above a negative.
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos, neg = y == 1, y == 0
    npos, nneg = pos.sum(), neg.sum()
    if npos == 0 or nneg == 0:
        return 0.5
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def f1_score(y, yhat):
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0, prec, rec


if __name__ == "__main__":
    np.random.seed(0)
    X_raw, y = make_data()
    feats = churn_features(X_raw)                         # engineer time-series features

    n = len(y)
    split = int(0.7 * n)                                  # held-out customer split
    idx = np.random.RandomState(0).permutation(n)
    tr, te = idx[:split], idx[split:]
    Xtr, Xte = standardize(feats[tr], feats[te])
    ytr, yte = y[tr], y[te]

    model = LogisticRegression().fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    pred = model.predict(Xte)

    acc = float((pred == yte).mean())
    auc = auc_score(yte, proba)
    f1, prec, rec = f1_score(yte, pred)

    # Baselines: always predict the majority class; random ranker scores AUC 0.5.
    majority = int(round(ytr.mean()))
    base_acc = float((np.full_like(yte, majority) == yte).mean())
    base_f1, _, _ = f1_score(yte, np.full_like(yte, majority))

    print("Customer Churn Prediction System (logistic regression, from scratch)")
    print(f"  customers={n}  months={X_raw.shape[1]}  features={feats.shape[1]}  test={len(te)}")
    print(f"  churn rate (train): {ytr.mean():.2f}")
    print(f"  baseline (majority) accuracy: {base_acc:.3f}   F1: {base_f1:.3f}   AUC: 0.500")
    print(f"  churn model          accuracy: {acc:.3f}   F1: {f1:.3f}   AUC: {auc:.3f}")
    print(f"  precision: {prec:.3f}   recall: {rec:.3f}")
    print(f"  accuracy lift over baseline: {100 * (acc - base_acc):.1f} pts")
    print(f"  result: {'MODEL BEATS baseline' if acc > base_acc and auc > 0.5 else 'baseline wins'}")
    print("  highest-risk test customers (predicted churn prob, actual):")
    for j in np.argsort(-proba)[:5]:
        print(f"    p(churn)={proba[j]:.3f}   actual={'churn' if yte[j] else 'stay'}")
