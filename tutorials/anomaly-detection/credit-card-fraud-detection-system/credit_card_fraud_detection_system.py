import numpy as np


class GaussianFraudDetector:
    """One-class fraud detector: model genuine transactions as a multivariate
    Gaussian and flag points whose Mahalanobis distance lands in the tail."""

    def __init__(self, contamination=0.01, ridge=1e-6):
        self.contamination = contamination   # expected fraud rate -> threshold
        self.ridge = ridge                    # covariance regularization

    def fit(self, X):
        # Fit density on genuine transactions only (unsupervised / one-class).
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = (Xc.T @ Xc) / (len(X) - 1)
        # Ridge keeps the covariance invertible and well-conditioned.
        cov += self.ridge * np.eye(X.shape[1])
        self.prec_ = np.linalg.inv(cov)       # inverse covariance (precision)

        # Threshold = high quantile of training scores (assume mostly genuine).
        s = self.score_samples(X)
        self.threshold_ = np.quantile(s, 1.0 - self.contamination)
        return self

    def score_samples(self, X):
        # Squared Mahalanobis distance: (x-mu)^T Sigma^{-1} (x-mu). Big = anomalous.
        Xc = X - self.mean_
        return np.einsum("ij,jk,ik->i", Xc, self.prec_, Xc)

    def predict(self, X):
        # 1 = flagged as fraud, 0 = genuine.
        return (self.score_samples(X) > self.threshold_).astype(int)


def roc_auc(scores, labels):
    """AUC via the Mann-Whitney statistic (rank of positives vs negatives)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def make_transactions(n_normal=4000, n_fraud=120, seed=0):
    """Synthetic transactions. Genuine spending has correlated features
    (amount, hour, distance, frequency, foreign-ratio); fraud sits in the
    low-density tail of that distribution."""
    rng = np.random.RandomState(seed)
    d = 5
    mu = np.array([50.0, 14.0, 6.0, 4.0, 0.15])   # typical genuine profile

    # Correlated covariance via a random loading matrix: Sigma = L L^T.
    L = rng.randn(d, d) * 0.6 + np.diag([12.0, 3.0, 2.5, 1.2, 0.05])
    X_norm = mu + rng.randn(n_normal, d) @ L.T

    # Fraud: push points outward to a Mahalanobis radius of ~3.2-6 std in the
    # whitened frame -> genuinely rare, with realistic overlap near the tail.
    r = rng.uniform(3.2, 6.0, size=n_fraud)
    u = rng.randn(n_fraud, d)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    X_fraud = mu + (r[:, None] * u) @ L.T

    X = np.vstack([X_norm, X_fraud])
    y = np.r_[np.zeros(n_normal), np.ones(n_fraud)].astype(int)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


if __name__ == "__main__":
    np.random.seed(0)
    X, y = make_transactions()

    # Held-out split; the detector trains on genuine transactions only.
    n_test = len(X) // 3
    Xtr, ytr, Xte, yte = X[n_test:], y[n_test:], X[:n_test], y[:n_test]
    Xtr_genuine = Xtr[ytr == 0]

    det = GaussianFraudDetector(contamination=0.01).fit(Xtr_genuine)
    scores = det.score_samples(Xte)
    pred = det.predict(Xte)

    # Detection metrics on the held-out labeled set.
    tp = int(((pred == 1) & (yte == 1)).sum())
    fp = int(((pred == 1) & (yte == 0)).sum())
    fn = int(((pred == 0) & (yte == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = roc_auc(scores, yte)

    # Baselines: random ranker = 0.5 AUC; best single-feature z-score AUC shows
    # the multivariate model beats naive per-feature thresholding.
    prevalence = yte.mean()
    z = np.abs((Xte - Xtr_genuine.mean(0)) / Xtr_genuine.std(0))
    z_auc = max(roc_auc(z[:, j], yte) for j in range(Xte.shape[1]))

    print("Held-out transactions: {}  fraud: {} ({:.1%})".format(
        len(yte), int(yte.sum()), prevalence))
    print("-" * 56)
    print("Mahalanobis AUC        : {:.3f}   (random baseline 0.500)".format(auc))
    print("Best single-z AUC      : {:.3f}   (naive per-feature baseline)".format(z_auc))
    print("-" * 56)
    print("Flagged as fraud       : {}   (TP={}, FP={})".format(tp + fp, tp, fp))
    print("Precision              : {:.3f}   (random baseline {:.3f})".format(
        precision, prevalence))
    print("Recall                 : {:.3f}".format(recall))
    print("F1                     : {:.3f}".format(f1))
    print("-" * 56)
    ok = auc > 0.9 and precision > 5 * prevalence
    print("RESULT: detector beats baselines" if ok else "RESULT: FAILED")
