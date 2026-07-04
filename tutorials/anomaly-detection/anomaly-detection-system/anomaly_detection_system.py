import numpy as np

# Anomaly Detection System (from scratch)
# ---------------------------------------
# Model the NORMAL population as a single multivariate Gaussian and score every
# point by its squared Mahalanobis distance to that fitted distribution:
#       D2(x) = (x - mu)^T * Sigma^-1 * (x - mu)
# Unlike plain Euclidean distance, Mahalanobis stretches space by the inverse
# covariance, so it accounts for feature scales AND correlations -- a point can
# sit near the mean on every axis yet still be an outlier by breaking the joint
# correlation structure. For clean normal data D2 follows a chi-square(d) law,
# so a distance far in its tail marks an anomaly. We fit mu/Sigma UNSUPERVISED
# on training data (no labels), pick a threshold from an assumed contamination
# rate, then evaluate detection quality on a held-out labeled set.


class MahalanobisAnomalyDetector:
    """Fit a Gaussian to the data; flag points with large Mahalanobis distance."""

    def __init__(self, contamination=0.05):
        self.contamination = contamination      # assumed fraction of outliers

    def fit(self, X):
        # Estimate the normal population's center and covariance.
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = (Xc.T @ Xc) / (len(X) - 1)
        # Ridge on the diagonal keeps Sigma invertible / well-conditioned.
        self.inv_cov_ = np.linalg.inv(cov + 1e-6 * np.eye(X.shape[1]))
        # Threshold = upper-contamination quantile of the training scores.
        s = self.score_samples(X)
        self.threshold_ = np.quantile(s, 1.0 - self.contamination)
        return self

    def score_samples(self, X):
        # Squared Mahalanobis distance for each row (vectorized quadratic form).
        Xc = X - self.mean_
        return np.einsum("ij,jk,ik->i", Xc, self.inv_cov_, Xc)

    def predict(self, X):
        # 1 = anomaly when the distance exceeds the learned threshold, else 0.
        return (self.score_samples(X) > self.threshold_).astype(int)


# ------------------------------ metric helpers ------------------------------
def roc_auc(labels, scores):
    """Rank-based ROC AUC (Mann-Whitney): P(score[anomaly] > score[normal])."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def prf(labels, preds):
    """Precision, recall, F1 for the positive (anomaly) class."""
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


if __name__ == "__main__":
    np.random.seed(0)
    d = 6

    # --- normal population: correlated multivariate Gaussian (planted Sigma) ---
    A = np.random.randn(d, d)
    cov = A @ A.T + d * np.eye(d)          # random positive-definite covariance
    mu = np.zeros(d)
    n_normal, n_anom = 900, 90             # ~9% contamination
    X_norm = np.random.multivariate_normal(mu, cov, size=n_normal)

    # --- inject anomalies along the LEAST-variance direction of the normal ---
    # cloud: they look plausible per-axis but violate the joint correlation, so
    # only Mahalanobis (not raw Euclidean distance) reliably separates them.
    evals, evecs = np.linalg.eigh(cov)
    thin_dir = evecs[:, 0]                 # eigenvector of smallest variance
    offset = 4.2 * np.sqrt(evals[0]) * thin_dir
    signs = np.random.choice([-1.0, 1.0], size=(n_anom, 1))
    X_anom = mu + signs * offset + 0.6 * np.random.multivariate_normal(mu, cov, size=n_anom)

    # --- train/test split: detector is FIT on normal-only training data ---
    n_tr = 600
    X_train = X_norm[:n_tr]                              # unlabeled, normal only
    X_test = np.vstack([X_norm[n_tr:], X_anom])
    y_test = np.r_[np.zeros(n_normal - n_tr), np.ones(n_anom)].astype(int)

    det = MahalanobisAnomalyDetector(contamination=0.05).fit(X_train)
    scores = det.score_samples(X_test)
    preds = det.predict(X_test)

    prec, rec, f1 = prf(y_test, preds)
    auc = roc_auc(y_test, scores)

    # --- baselines: random ranking (AUC 0.5) and majority "never anomaly" ---
    prevalence = y_test.mean()
    maj_prec, maj_rec, maj_f1 = prf(y_test, np.zeros_like(y_test))

    print("Anomaly Detection via from-scratch Mahalanobis distance")
    print("-" * 62)
    print("features=%d  train(normal-only)=%d  test=%d  anomalies=%d (%.1f%%)"
          % (d, n_tr, len(y_test), int(y_test.sum()), 100 * prevalence))
    print("learned D2 threshold=%.2f   chi-square(d) mean=%d" % (det.threshold_, d))
    print("-" * 62)
    print("Mahalanobis   AUC=%.3f  Precision=%.3f  Recall=%.3f  F1=%.3f"
          % (auc, prec, rec, f1))
    print("Random rank   AUC=0.500")
    print("Majority(0)   Precision=0.000  Recall=0.000  F1=%.3f" % maj_f1)
    print("-" * 62)
    print("AUC lift over random: 0.500 -> %.3f" % auc)
    print("F1  lift over majority: %.3f -> %.3f" % (maj_f1, f1))
    ok = auc > 0.85 and f1 > 0.5 and f1 > maj_f1
    print("RESULT:", "PASS - clearly beats baselines" if ok else "FAIL")
