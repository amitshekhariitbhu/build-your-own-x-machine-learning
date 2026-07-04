import numpy as np


class EarthquakePredictor:
    """Logistic-regression seismic hazard classifier from scratch.

    Given precursor features measured over a monitoring window, predicts whether
    a MAJOR earthquake (magnitude >= 6) follows (1) or not (0). Trained by
    full-batch gradient descent on standardized features with an L2 penalty;
    outputs a hazard probability via the sigmoid, thresholded at 0.5.
    """

    def __init__(self, lr=0.2, n_iter=1200, l2=1e-3):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2  # ridge penalty to keep weights sane
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sigma = None

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    def _standardize(self, X, fit=False):
        if fit:
            self.mu = X.mean(axis=0)
            self.sigma = X.std(axis=0) + 1e-8
        return (X - self.mu) / self.sigma

    def fit(self, X, y):
        X = self._standardize(np.asarray(X, dtype=float), fit=True)
        y = np.asarray(y, dtype=float).reshape(-1)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iter):
            p = self._sigmoid(X @ self.w + self.b)  # predicted hazard probs
            err = p - y
            grad_w = X.T @ err / n + self.l2 * self.w  # cross-entropy + ridge grad
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        X = self._standardize(np.asarray(X, dtype=float))
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


def make_seismic_data(n=800, seed=0):
    """Synthetic seismic monitoring windows with a planted hazard signal.

    Feature order:
      foreshock_rate  - small tremors counted in the window (rises before a quake)
      b_value         - Gutenberg-Richter slope (DROPS before a major quake)
      log_energy      - cumulative log seismic energy released
      strain_rate     - crustal strain accumulation (higher = more stress)
      radon_anomaly   - geochemical soil-gas precursor spike
      depth_km        - focal depth (shallow events are more hazardous)
    """
    rng = np.random.RandomState(seed)
    n_hi = n // 2                     # windows preceding a major quake
    n_lo = n - n_hi                   # quiet windows
    #                foreshock  b_value  log_energy  strain  radon  depth
    hi_mean = np.array([22.0,     0.80,     12.8,     6.5,    5.0,   18.0])
    lo_mean = np.array([12.0,     1.02,     11.4,     4.0,    2.6,   30.0])
    std = np.array([9.0, 0.16, 1.4, 2.6, 2.4, 12.0])  # per-feature spread (overlap)
    hi = rng.normal(hi_mean, std, size=(n_hi, 6))
    lo = rng.normal(lo_mean, std, size=(n_lo, 6))
    X = np.vstack([hi, lo])
    y = np.concatenate([np.ones(n_hi), np.zeros(n_lo)])
    X[:, 1] = np.clip(X[:, 1], 0.3, 1.6)   # b-value physically bounded
    X[:, 0] = np.clip(X[:, 0], 0, None)    # counts are non-negative
    perm = rng.permutation(n)
    return X[perm], y[perm].astype(int)


def metrics(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return acc, prec, rec, f1


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_seismic_data(n=800, seed=0)

    # held-out split: 70% train / 30% test
    n_train = int(0.7 * len(y))
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]

    clf = EarthquakePredictor(lr=0.2, n_iter=1200, l2=1e-3).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc, prec, rec, f1 = metrics(yte, pred)

    # majority-class baseline (always predict the more frequent training label)
    majority = int(round(ytr.mean()))
    base_pred = np.full_like(yte, majority)
    base_acc, _, _, base_f1 = metrics(yte, base_pred)

    print("=== Earthquake Prediction System (from scratch) ===")
    print(f"test windows          : {len(yte)}")
    print(f"major-quake windows   : {int(yte.sum())} / {len(yte)}")
    print(f"baseline (majority)   : acc={base_acc:.3f}  f1={base_f1:.3f}")
    print(f"logistic hazard model : acc={acc:.3f}  f1={f1:.3f}")
    print(f"  precision={prec:.3f}  recall={rec:.3f}")
    print(f"improvement over base : +{(acc - base_acc) * 100:.1f} acc points")
    assert acc > base_acc + 0.15, "model should clearly beat the majority baseline"
    print("PASS: earthquake predictor beats majority baseline")
