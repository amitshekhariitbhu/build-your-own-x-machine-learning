import numpy as np

# Eye-Aspect-Ratio (EAR) is the classic drowsiness cue: it is high when the
# eyes are open and collapses toward zero when they close. We never touch a
# camera here -- instead we SIMULATE the per-frame EAR signal for short video
# windows, extract the features a real system uses (PERCLOS, blink count,
# longest closure, ...), and learn ALERT vs DROWSY from scratch.

CLOSED = 0.20          # EAR below this = eye considered closed (blink/microsleep)


def make_ear_window(rng, drowsy, T=60):
    """Synthetic EAR time series for one ~2s window (T frames @ 30fps).

    ALERT : eyes wide (high baseline), rare and SHORT blinks (1-3 frames).
    DROWSY: droopy eyes (lower baseline), frequent LONG closures / microsleeps.
    """
    base = 0.26 if drowsy else 0.31                       # eyes-open baseline
    ear = base + rng.normal(0, 0.015, T)                  # steady-state jitter
    if drowsy:
        n_events, dur_lo, dur_hi = rng.randint(3, 6), 4, 12   # long microsleeps
    else:
        n_events, dur_lo, dur_hi = rng.randint(0, 3), 1, 3    # quick blinks
    for _ in range(n_events):
        start = rng.randint(0, T)
        end = min(start + rng.randint(dur_lo, dur_hi + 1), T)
        ear[start:end] = 0.10 + rng.normal(0, 0.01, end - start)   # closure dip
    return np.clip(ear, 0.02, 0.40)


def longest_run(mask):
    """Length of the longest run of True in a boolean array (loop, from scratch)."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def extract_features(ear):
    """Turn a raw EAR window into the 5 drowsiness features used downstream."""
    closed = ear < CLOSED
    edges = np.diff(closed.astype(int))
    n_blinks = int((edges == 1).sum() + (1 if closed[0] else 0))   # falling edges
    return np.array([
        ear.mean(),                 # mean openness (drops when drowsy)
        ear.std(),                  # variability
        closed.mean(),              # PERCLOS: fraction of time eyes are closed
        n_blinks,                   # number of closure events
        longest_run(closed),        # longest single closure (microsleep length)
    ])


def make_dataset(n=1200, seed=0):
    # Balanced synthetic driver log: each row is one window's feature vector,
    # label 1 = drowsy. Structure is PLANTED via the EAR generator above.
    rng = np.random.RandomState(seed)
    X, y = [], []
    for _ in range(n):
        drowsy = rng.rand() < 0.5
        X.append(extract_features(make_ear_window(rng, drowsy)))
        y.append(int(drowsy))
    return np.array(X), np.array(y, dtype=float)


class LogisticRegression:
    """L2-regularized logistic regression via full-batch gradient descent.

    p(drowsy | x) = sigmoid(x . w + b). The gradient of the mean binary
    cross-entropy w.r.t. the logit is just (p - y), keeping the update compact.
    """

    def __init__(self, lr=0.3, epochs=400, l2=1e-4):
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

    X, y = make_dataset(n=1200, seed=0)

    # Held-out split: learn thresholds on 70%, judge on the unseen 30%.
    idx = np.random.permutation(len(X))
    split = int(0.7 * len(X))
    tr, te = idx[:split], idx[split:]

    # Standardize features with TRAIN statistics so gradient descent is stable.
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    ytr, yte = y[tr], y[te]

    model = LogisticRegression().fit(Xtr, ytr)
    pred = model.predict(Xte)

    acc = np.mean(pred == yte)
    prec, rec, f1 = prf(yte, pred)
    base_acc = max(yte.mean(), 1 - yte.mean())            # majority-class guess

    names = ["mean_EAR", "std_EAR", "PERCLOS", "blinks", "max_closure"]
    print("Windows: %d   Train: %d   Test: %d   Frames/window: 60"
          % (len(X), len(tr), len(te)))
    print("Drowsy prevalence: %.3f" % y.mean())
    print("-" * 60)
    print("Drowsiness detector   acc: %.4f  P: %.4f  R: %.4f  F1: %.4f"
          % (acc, prec, rec, f1))
    print("Majority baseline     acc: %.4f  (always predict most common class)"
          % base_acc)
    print("-" * 60)
    order = np.argsort(np.abs(model.w))[::-1]
    print("Learned feature weights (|w| ranked):")
    for j in order:
        print("   %-12s % .3f" % (names[j], model.w[j]))
    print("-" * 60)
    print("Detector beats baseline: %s" % (acc > base_acc and f1 > 0.5))
