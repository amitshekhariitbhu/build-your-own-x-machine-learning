import numpy as np


class CreditRiskModel:
    """Consumer credit-risk scorer: logistic regression trained from scratch
    with batch gradient descent. Outputs a probability of default and a
    hard good/bad decision, mirroring how a credit scorecard flags risk."""

    def __init__(self, lr=0.3, epochs=400, l2=1e-3):
        self.lr = lr            # gradient-descent step size
        self.epochs = epochs    # full passes over the data
        self.l2 = l2            # ridge penalty (keeps weights tame)

    @staticmethod
    def _sigmoid(z):
        # Numerically stable logistic function.
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                        np.exp(z) / (1.0 + np.exp(z)))

    def fit(self, X, y):
        # Standardize features so gradient descent conditions well.
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        Xs = (X - self.mean_) / self.std_
        n, d = Xs.shape

        self.w_ = np.zeros(d)
        self.b_ = 0.0
        for _ in range(self.epochs):
            p = self._sigmoid(Xs @ self.w_ + self.b_)
            err = p - y                       # dLoss/dz for log-loss
            gw = Xs.T @ err / n + self.l2 * self.w_
            gb = err.mean()
            self.w_ -= self.lr * gw
            self.b_ -= self.lr * gb
        return self

    def predict_proba(self, X):
        # Probability of default (the "bad" class = 1).
        Xs = (X - self.mean_) / self.std_
        return self._sigmoid(Xs @ self.w_ + self.b_)

    def predict(self, X, threshold=0.5):
        # 1 = predicted default / high risk, 0 = good standing.
        return (self.predict_proba(X) >= threshold).astype(int)


def roc_auc(scores, labels):
    """AUC via the Mann-Whitney statistic (rank of positives vs negatives)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def make_borrowers(n=3000, seed=0):
    """Synthetic loan applicants. Each has correlated financial features and a
    planted latent default risk: high debt-to-income, high credit utilization,
    many late payments and short history drive defaults; income and a long
    history protect. Default label is drawn from that true logit."""
    rng = np.random.RandomState(seed)

    income = rng.gamma(shape=4.0, scale=15.0, size=n)          # $k / year
    dti = np.clip(rng.normal(0.35, 0.15, n), 0.02, 1.2)        # debt-to-income
    util = np.clip(rng.beta(2.0, 3.0, n), 0.0, 1.0)            # credit utilization
    late = rng.poisson(1.2, n).astype(float)                  # late payments / yr
    history = np.clip(rng.normal(8.0, 4.0, n), 0.2, 30.0)      # years of history
    loan = rng.gamma(shape=3.0, scale=6.0, size=n)             # $k requested

    # True latent log-odds of default -> plant a recoverable signal.
    z = (-3.4
         + 4.0 * dti
         + 3.2 * util
         + 0.55 * late
         - 0.045 * income
         - 0.16 * history
         + 0.06 * loan)
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(size=n) < p).astype(int)

    X = np.column_stack([income, dti, util, late, history, loan])
    return X, y


if __name__ == "__main__":
    np.random.seed(0)
    X, y = make_borrowers()

    # Held-out split.
    n_test = len(X) // 3
    Xte, yte = X[:n_test], y[:n_test]
    Xtr, ytr = X[n_test:], y[n_test:]

    model = CreditRiskModel().fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    pred = model.predict(Xte)

    # Metrics on the held-out set.
    acc = (pred == yte).mean()
    tp = int(((pred == 1) & (yte == 1)).sum())
    fp = int(((pred == 1) & (yte == 0)).sum())
    fn = int(((pred == 0) & (yte == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = roc_auc(proba, yte)

    # Baselines: majority-class accuracy and a random (0.5) AUC ranker.
    default_rate = yte.mean()
    majority_acc = max(default_rate, 1.0 - default_rate)

    print("Held-out applicants : {}   defaults: {} ({:.1%})".format(
        len(yte), int(yte.sum()), default_rate))
    print("-" * 58)
    print("Accuracy            : {:.3f}   (majority baseline {:.3f})".format(
        acc, majority_acc))
    print("ROC AUC             : {:.3f}   (random baseline 0.500)".format(auc))
    print("Precision (default) : {:.3f}".format(precision))
    print("Recall    (default) : {:.3f}".format(recall))
    print("F1        (default) : {:.3f}".format(f1))
    print("-" * 58)
    ok = acc > majority_acc + 0.02 and auc > 0.70
    print("RESULT: credit-risk model beats baselines" if ok else "RESULT: FAILED")
