import numpy as np

# Wearable stress detection: acute stress triggers the sympathetic nervous
# system, which shows up in physiology -- heart rate climbs, heart-rate
# variability collapses, the skin sweats more (rising electrodermal activity
# with frequent micro-peaks), breathing speeds up, and finger temperature dips.
# We never touch a real sensor: we SIMULATE the per-window raw signals a wrist
# device records, extract the features a real detector uses, and learn
# CALM vs STRESSED from scratch with logistic regression.

FS = 4          # samples per second
WIN = 60        # window length in seconds -> T = FS*WIN samples


def make_window(rng, stressed, T=FS * WIN):
    """Synthetic multi-sensor window. Structure is PLANTED by the stress flag.

    Returns raw (hr, eda, resp, temp) signals for one ~60s window.
    STRESSED: faster jittery heart, sweaty rising skin, rapid breathing, cool skin.
    """
    t = np.arange(T) / FS
    # Heart rate (bpm): higher mean + LARGER beat-to-beat jitter (low HRV) when stressed.
    hr = (86.0 if stressed else 68.0) + rng.normal(0, 5.0 if stressed else 2.0, T)
    # Electrodermal activity (uS): rising tonic drift + frequent phasic sweat peaks.
    eda = (6.0 if stressed else 3.0) + (0.9 if stressed else 0.15) * t / t[-1]
    eda = eda + rng.normal(0, 0.05, T)
    n_scr = rng.randint(8, 16) if stressed else rng.randint(0, 4)   # sweat bursts
    for _ in range(n_scr):
        c = rng.randint(0, T)
        eda += (0.6 * np.exp(-((np.arange(T) - c) ** 2) / (2 * (FS * 1.5) ** 2)))
    # Respiration (breaths/min drives a sine): faster + more irregular when stressed.
    br = (22.0 if stressed else 14.0) + rng.normal(0, 2.0 if stressed else 0.5)
    resp = np.sin(2 * np.pi * (br / 60.0) * t) + rng.normal(0, 0.1, T)
    # Finger temperature (C): peripheral vasoconstriction cools the skin under stress.
    temp = (32.5 if stressed else 34.0) + rng.normal(0, 0.2, T)
    return hr, eda, resp, temp


def count_peaks(x, prominence):
    """Count local maxima that rise at least `prominence` above their neighbors."""
    peaks = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & \
            (x[1:-1] - np.minimum(x[:-2], x[2:]) > prominence)
    return int(peaks.sum())


def extract_features(hr, eda, resp, temp):
    """Turn raw sensor window into the 7 physiological stress features."""
    rmssd = np.sqrt(np.mean(np.diff(hr) ** 2))            # HRV: high jitter = low HRV
    eda_slope = np.polyfit(np.arange(len(eda)), eda, 1)[0]
    resp_rate = count_peaks(resp, 0.3) / (WIN / 60.0)     # breaths per minute
    return np.array([
        hr.mean(),                    # mean heart rate (up under stress)
        rmssd,                        # beat-to-beat variability (up = low HRV = stress)
        eda.mean(),                   # mean skin conductance (up under stress)
        count_peaks(eda, 0.15),       # phasic sweat bursts (SCR count)
        eda_slope * 1e3,              # EDA rising trend
        resp_rate,                    # breathing rate (up under stress)
        temp.mean(),                  # skin temperature (down under stress)
    ])


def make_dataset(n=1000, seed=0):
    # Balanced synthetic wearable log: one row per 60s window, label 1 = stressed.
    rng = np.random.RandomState(seed)
    X, y = [], []
    for _ in range(n):
        stressed = rng.rand() < 0.5
        X.append(extract_features(*make_window(rng, stressed)))
        y.append(int(stressed))
    return np.array(X), np.array(y, dtype=float)


class LogisticRegression:
    """L2-regularized logistic regression via full-batch gradient descent.

    p(stress | x) = sigmoid(x . w + b). The gradient of mean binary
    cross-entropy w.r.t. the logit is simply (p - y).
    """

    def __init__(self, lr=0.3, epochs=500, l2=1e-4):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def _proba(self, X):
        z = X @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.epochs):
            err = self._proba(X) - y                      # dBCE/dlogit
            self.w -= self.lr * (X.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        return self._proba(X)

    def predict(self, X):
        return (self._proba(X) >= 0.5).astype(int)


def prf(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_dataset(n=1000, seed=0)

    # Held-out split: learn on 70%, judge on the unseen 30%.
    idx = np.random.permutation(len(X))
    split = int(0.7 * len(X))
    tr, te = idx[:split], idx[split:]

    # Standardize with TRAIN statistics so gradient descent is well-conditioned.
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    ytr, yte = y[tr], y[te]

    model = LogisticRegression().fit(Xtr, ytr)
    pred = model.predict(Xte)

    acc = np.mean(pred == yte)
    prec, rec, f1 = prf(yte, pred)
    base_acc = max(yte.mean(), 1 - yte.mean())            # majority-class guess

    names = ["mean_HR", "HRV_rmssd", "mean_EDA", "SCR_peaks",
             "EDA_slope", "resp_rate", "skin_temp"]
    print("Windows: %d   Train: %d   Test: %d   (%ds @ %dHz each)"
          % (len(X), len(tr), len(te), WIN, FS))
    print("Stress prevalence: %.3f" % y.mean())
    print("-" * 62)
    print("Stress detector    acc: %.4f  P: %.4f  R: %.4f  F1: %.4f"
          % (acc, prec, rec, f1))
    print("Majority baseline  acc: %.4f  (always predict most common class)"
          % base_acc)
    print("-" * 62)
    order = np.argsort(np.abs(model.w))[::-1]
    print("Learned feature weights (|w| ranked):")
    for j in order:
        print("   %-11s % .3f" % (names[j], model.w[j]))
    print("-" * 62)
    print("Detector beats baseline: %s" % (acc > base_acc and f1 > 0.6))
