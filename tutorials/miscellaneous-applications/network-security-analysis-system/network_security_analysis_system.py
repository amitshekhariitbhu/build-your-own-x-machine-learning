import numpy as np

# Build a Network Security Analysis System from scratch.
# A sensor logs one feature vector per network CONNECTION: how long it lasted,
# how many bytes flowed each way, packet rate, how many connections hit the same
# host, how many distinct ports were touched, failed logins, and the SYN-without-
# ACK ratio. The vast majority of traffic is BENIGN and clusters tightly; a few
# connections are ATTACKS whose signatures push some features far off the normal
# manifold (DoS floods, port scans, data exfiltration, brute-force logins).
# The detector is trained ONLY on benign traffic (semi-supervised anomaly
# detection) -- it never sees a labelled attack. It fits a multivariate Gaussian
# to the benign cloud and scores each new connection by its MAHALANOBIS distance
# (negative log-likelihood) to that cloud; the threshold is the 99th percentile
# of benign validation scores. On a held-out mix it must beat a random detector
# (AUC 0.5) and a majority "predict benign" baseline on precision/recall/F1.

FEATURES = ["duration", "src_bytes", "dst_bytes", "pkt_rate",
            "same_host_conns", "distinct_ports", "failed_logins", "syn_ratio"]


def synth_traffic(n_benign, n_attack, seed):
    # Benign traffic: a tight Gaussian cloud in log/standardized feature space.
    rng = np.random.RandomState(seed)
    mu = np.array([2.0, 7.0, 8.5, 3.0, 2.0, 1.5, 0.2, 0.05])   # typical benign profile
    sd = np.array([0.6, 0.8, 0.9, 0.7, 0.6, 0.5, 0.3, 0.04])
    benign = mu + rng.randn(n_benign, len(mu)) * sd
    Xb = benign
    yb = np.zeros(n_benign, dtype=int)

    # Four attack families, each shifting a DIFFERENT subset of features hard.
    kinds = rng.randint(0, 4, size=n_attack)
    Xa = mu + rng.randn(n_attack, len(mu)) * sd                 # start near benign
    dos = kinds == 0                                            # flood: fast, many conns
    Xa[dos, 3] += rng.uniform(3.0, 5.0, dos.sum())             # pkt_rate spike
    Xa[dos, 4] += rng.uniform(3.0, 5.0, dos.sum())             # same_host_conns spike
    Xa[dos, 7] += rng.uniform(0.4, 0.8, dos.sum())             # syn_ratio spike
    Xa[dos, 0] -= rng.uniform(2.0, 3.0, dos.sum())             # short duration
    scan = kinds == 1                                          # port scan: many ports
    Xa[scan, 5] += rng.uniform(3.0, 5.0, scan.sum())          # distinct_ports spike
    Xa[scan, 1] -= rng.uniform(3.0, 5.0, scan.sum())          # tiny src_bytes
    exfil = kinds == 2                                         # exfiltration: huge upload
    Xa[exfil, 1] += rng.uniform(3.0, 5.0, exfil.sum())        # src_bytes spike
    Xa[exfil, 0] += rng.uniform(2.0, 3.5, exfil.sum())        # long duration
    brute = kinds == 3                                        # brute force logins
    Xa[brute, 6] += rng.uniform(3.0, 5.0, brute.sum())       # failed_logins spike
    ya = np.ones(n_attack, dtype=int)

    X = np.vstack([Xb, Xa])
    y = np.concatenate([yb, ya])
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


class GaussianAnomalyDetector:
    """Semi-supervised network intrusion detector.

    Fits a multivariate Gaussian N(mu, Sigma) to BENIGN traffic only, then scores
    each connection by its squared Mahalanobis distance -- the tail of the benign
    likelihood. Larger distance = more anomalous = more likely an intrusion.
    """

    def __init__(self, quantile=0.99):
        self.quantile = quantile      # benign-score percentile used as the threshold

    def fit(self, X_benign):
        X = np.asarray(X_benign, float)
        self.mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6          # ridge for a stable inverse
        self.prec = np.linalg.inv(cov)              # precision matrix Sigma^-1
        return self

    def score(self, X):
        # Squared Mahalanobis distance: (x-mu)^T Sigma^-1 (x-mu), vectorized.
        d = np.asarray(X, float) - self.mu
        return np.einsum("ij,jk,ik->i", d, self.prec, d)

    def calibrate(self, X_benign_val):
        # Threshold = high percentile of benign scores -> ~1% benign false alarms.
        self.threshold = np.quantile(self.score(X_benign_val), self.quantile)
        return self

    def predict(self, X):
        return (self.score(X) > self.threshold).astype(int)


def auc(scores, y):
    # Rank-based AUC = P(score(attack) > score(benign)) via the Mann-Whitney U.
    order = np.argsort(np.argsort(scores))
    pos = y == 1
    u = order[pos].sum() - pos.sum() * (pos.sum() - 1) / 2.0
    return u / (pos.sum() * (~pos).sum())


if __name__ == "__main__":
    np.random.seed(0)

    # Train on benign-only; hold out a fresh mixed set (benign + attacks) to test.
    X_tr, y_tr = synth_traffic(1200, 0, seed=0)               # benign-only training
    X_val, y_val = synth_traffic(400, 0, seed=1)              # benign-only calibration
    X_te, y_te = synth_traffic(600, 90, seed=2)               # held-out mix (~13% attacks)

    det = GaussianAnomalyDetector(quantile=0.99).fit(X_tr).calibrate(X_val)
    scores = det.score(X_te)
    pred = det.predict(X_te)

    tp = int(((pred == 1) & (y_te == 1)).sum())
    fp = int(((pred == 1) & (y_te == 0)).sum())
    fn = int(((pred == 0) & (y_te == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (pred == y_te).mean()

    base_rate = y_te.mean()                                   # attack prevalence
    majority_acc = 1 - base_rate                              # "always benign" accuracy
    roc = auc(scores, y_te)

    print("Connections  train(benign):%d  test:%d  attacks in test:%d (%.1f%%)"
          % (len(y_tr), len(y_te), int(y_te.sum()), 100 * base_rate))
    print("Features: " + ", ".join(FEATURES))
    print("-" * 66)
    print("Detector AUC              : %.3f   (random = 0.500)" % roc)
    print("Detector precision        : %.3f" % prec)
    print("Detector recall           : %.3f" % rec)
    print("Detector F1               : %.3f   (random-guess F1 approx = %.3f)"
          % (f1, base_rate))
    print("Detector accuracy         : %.3f" % acc)
    print("Majority-baseline accuracy: %.3f   (always predict 'benign')" % majority_acc)
    print("-" * 66)
    beats = (roc > 0.5 and f1 > base_rate and rec > 0.5 and acc >= majority_acc)
    print("Security analyzer beats random & majority baselines:", beats)
