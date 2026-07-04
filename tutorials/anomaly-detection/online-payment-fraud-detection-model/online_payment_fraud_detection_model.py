import numpy as np


class LogisticFraudModel:
    """Supervised online-payment fraud detector: L2-regularized logistic
    regression trained from scratch with full-batch gradient descent. Class
    weighting counters the heavy legit/fraud imbalance so the rare positives
    still drive the gradient."""

    def __init__(self, lr=0.5, epochs=400, l2=1e-3, pos_weight=None):
        self.lr = lr                # learning rate
        self.epochs = epochs        # gradient-descent steps
        self.l2 = l2                # weight decay
        self.pos_weight = pos_weight  # up-weight fraud rows; None -> auto

    @staticmethod
    def _sigmoid(z):
        # Numerically stable logistic function.
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                        np.exp(z) / (1.0 + np.exp(z)))

    def fit(self, X, y):
        # Standardize features (stats stored for inference).
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-9
        Xs = (X - self.mean_) / self.std_
        n, d = Xs.shape

        # Per-class weights: balance total influence of fraud vs legit.
        pw = self.pos_weight
        if pw is None:
            pw = (y == 0).sum() / max((y == 1).sum(), 1)
        w = np.where(y == 1, pw, 1.0)

        self.coef_ = np.zeros(d)
        self.bias_ = 0.0
        for _ in range(self.epochs):
            p = self._sigmoid(Xs @ self.coef_ + self.bias_)
            g = w * (p - y)                      # weighted gradient of log-loss
            grad_w = Xs.T @ g / n + self.l2 * self.coef_
            grad_b = g.sum() / n
            self.coef_ -= self.lr * grad_w
            self.bias_ -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        Xs = (X - self.mean_) / self.std_
        return self._sigmoid(Xs @ self.coef_ + self.bias_)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def roc_auc(scores, labels):
    """AUC via the Mann-Whitney U statistic (rank of positives vs negatives)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def make_payments(n=12000, fraud_rate=0.02, seed=0):
    """Synthetic online-payment log (PaySim-style). Each row: transaction type
    (one-hot), amount, and origin balances before/after. Planted fraud signal:
    fraud only strikes cash-out / transfer, drains the account (new balance ~0,
    amount ~= old balance), and leaves a tell-tale balance-bookkeeping error."""
    rng = np.random.RandomState(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    # ---- Legit transactions -------------------------------------------------
    # Types: 0=PAYMENT 1=CASH_OUT 2=TRANSFER 3=DEBIT (fraud lives in 1 and 2).
    ltype = rng.choice(4, size=n_legit, p=[0.45, 0.30, 0.15, 0.10])
    lold = rng.gamma(2.0, 8000.0, size=n_legit)          # balance before
    lamt = np.minimum(lold * rng.uniform(0.0, 0.6, n_legit),
                      rng.gamma(2.0, 4000.0, size=n_legit))
    lnew = np.maximum(lold - lamt, 0.0)                  # balances reconcile

    # ---- Fraud transactions -------------------------------------------------
    ftype = rng.choice([1, 2], size=n_fraud)             # cash-out / transfer
    fold = rng.gamma(2.0, 9000.0, size=n_fraud)
    famt = fold * rng.uniform(0.9, 1.0, n_fraud)          # drain most of it
    fnew = np.zeros(n_fraud)                              # account emptied
    # Fraud bookkeeping doesn't add up cleanly (mule/mirror accounts).
    fnew += rng.uniform(0.0, 200.0, n_fraud) * (rng.rand(n_fraud) < 0.3)

    ttype = np.r_[ltype, ftype]
    old = np.r_[lold, fold]
    amt = np.r_[lamt, famt]
    new = np.r_[lnew, fnew]
    y = np.r_[np.zeros(n_legit), np.ones(n_fraud)].astype(int)

    # Feature matrix: one-hot type (4) + amount + old + new + balance-error.
    onehot = np.eye(4)[ttype]
    err = old - amt - new                                # 0 for clean legit rows
    X = np.column_stack([onehot, amt, old, new, err])

    idx = rng.permutation(n)
    return X[idx], y[idx]


if __name__ == "__main__":
    np.random.seed(0)
    X, y = make_payments()

    # Held-out split (stratification happens naturally via the shuffle).
    n_test = len(X) // 3
    Xte, yte, Xtr, ytr = X[:n_test], y[:n_test], X[n_test:], y[n_test:]

    model = LogisticFraudModel().fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    pred = model.predict(Xte)

    tp = int(((pred == 1) & (yte == 1)).sum())
    fp = int(((pred == 1) & (yte == 0)).sum())
    fn = int(((pred == 0) & (yte == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = float((pred == yte).mean())
    auc = roc_auc(proba, yte)

    # Baselines: majority classifier (predict "legit" for everyone) and a
    # random ranker (AUC 0.5). Majority looks accurate but catches zero fraud.
    prevalence = yte.mean()
    majority_acc = 1.0 - prevalence

    print("Held-out payments: {}  fraud: {} ({:.1%})".format(
        len(yte), int(yte.sum()), prevalence))
    print("-" * 58)
    print("Accuracy        : {:.3f}   (majority baseline {:.3f})".format(
        acc, majority_acc))
    print("ROC-AUC         : {:.3f}   (random baseline 0.500)".format(auc))
    print("-" * 58)
    print("Flagged as fraud: {}   (TP={}, FP={})".format(tp + fp, tp, fp))
    print("Precision       : {:.3f}   (random baseline {:.3f})".format(
        precision, prevalence))
    print("Recall          : {:.3f}".format(recall))
    print("F1              : {:.3f}   (majority baseline 0.000)".format(f1))
    print("-" * 58)
    ok = auc > 0.95 and f1 > 0.8 and recall > 0.8
    print("RESULT: model beats baselines" if ok else "RESULT: FAILED")
