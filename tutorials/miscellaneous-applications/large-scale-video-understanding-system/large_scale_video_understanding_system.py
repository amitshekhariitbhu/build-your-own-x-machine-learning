import numpy as np

# Large-Scale Video Understanding System (from scratch).
# A video is a sequence of per-frame feature vectors. Recognising the ACTION
# needs both APPEARANCE (what is in the scene) and MOTION (how it moves over
# time). We build a temporal feature extractor + a softmax classifier and show
# that modelling motion is what makes video understanding work: several action
# classes share the same appearance and differ ONLY in their temporal dynamics.


# --- Action templates: (appearance prototype id, motion kind, motion dir id) -
# Classes 0/1/2 share appearance A, classes 3/4 share appearance B -> they are
# separable ONLY through temporal motion, not from any single frame.
ACTIONS = [
    ("A", "still",  0),   # 0: scene A, no motion
    ("A", "drift+", 0),   # 1: scene A drifting forward
    ("A", "drift-", 0),   # 2: scene A drifting backward
    ("B", "drift+", 1),   # 3: scene B drifting forward
    ("B", "osc",    1),   # 4: scene B oscillating (net drift ~0, high energy)
    ("C", "drift+", 2),   # 5: scene C drifting forward
]
LABELS = ["A-still", "A-fwd", "A-back", "B-fwd", "B-osc", "C-fwd"]


def make_videos(n=720, T=16, D=8, seed=0):
    """Synthesise clips: (N,T,D) frame-feature tensors with planted action
    structure. Each clip = appearance base + a class-specific temporal path."""
    rng = np.random.RandomState(seed)
    appear = {"A": rng.randn(D), "B": rng.randn(D), "C": rng.randn(D)}
    dirs = rng.randn(3, D)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)     # unit motion axes
    tau = (np.arange(T) - (T - 1) / 2.0) / T                # centred time, [-.5,.5]

    X = np.zeros((n, T, D))
    y = np.zeros(n, dtype=int)
    for i in range(n):
        c = rng.randint(len(ACTIONS))
        appn, kind, dj = ACTIONS[c]
        base = appear[appn] + 0.30 * rng.randn(D)           # per-clip appearance jitter
        m = dirs[dj]
        if kind == "still":
            path = np.zeros((T, 1))
        elif kind == "drift+":
            path = (2.2 * tau)[:, None]
        elif kind == "drift-":
            path = (-2.2 * tau)[:, None]
        else:  # oscillation: strong per-frame motion but ~zero net drift
            path = (1.6 * np.sin(2 * np.pi * tau))[:, None]
        X[i] = base[None, :] + path * m[None, :] + 0.20 * rng.randn(T, D)
        y[i] = c
    return X, y


def extract_features(X):
    """Turn each raw clip into an appearance+motion descriptor.
      appearance : mean-pooled frame features (what the scene looks like)
      net-motion : mean signed frame-to-frame delta (direction of drift)
      dynamics   : mean |frame delta| (how much the scene moves at all)"""
    appearance = X.mean(axis=1)
    diffs = X[:, 1:, :] - X[:, :-1, :]
    net_motion = diffs.mean(axis=1)
    dynamics = np.abs(diffs).mean(axis=1)
    return appearance, np.concatenate([appearance, net_motion, dynamics], axis=1)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxClassifier:
    """Multiclass logistic regression trained by full-batch gradient descent."""

    def __init__(self, lr=0.5, epochs=400, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        n, d = X.shape
        self.classes = np.unique(y)
        k = len(self.classes)
        Y = np.eye(k)[np.searchsorted(self.classes, y)]     # one-hot targets
        self.W = np.zeros((d, k))
        self.b = np.zeros(k)
        for _ in range(self.epochs):
            probs = _softmax(X @ self.W + self.b)
            g = (probs - Y) / n                             # dL/dlogits
            self.W -= self.lr * (X.T @ g + self.l2 * self.W)
            self.b -= self.lr * g.sum(axis=0)
        return self

    def predict(self, X):
        return self.classes[np.argmax(X @ self.W + self.b, axis=1)]


def standardize(train, *others):
    mu, sd = train.mean(0), train.std(0) + 1e-8
    return [(a - mu) / sd for a in (train,) + others]


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_videos(n=720, T=16, D=8, seed=0)
    app, feats = extract_features(X)                        # appearance-only + full

    idx = np.random.permutation(len(y))
    split = int(0.7 * len(y))
    tr, te = idx[:split], idx[split:]

    # Full system: appearance + motion + dynamics.
    Ftr, Fte = standardize(feats[tr], feats[te])
    full = SoftmaxClassifier().fit(Ftr, y[tr])
    full_pred = full.predict(Fte)
    full_acc = np.mean(full_pred == y[te])

    # Ablation baseline: appearance ONLY (no temporal understanding).
    Atr, Ate = standardize(app[tr], app[te])
    appo = SoftmaxClassifier().fit(Atr, y[tr])
    appo_acc = np.mean(appo.predict(Ate) == y[te])

    # Trivial baseline: always predict the majority training class.
    majority = np.bincount(y[tr]).argmax()
    base_acc = np.mean(y[te] == majority)

    print("Videos: %d   Frames/clip: %d   Feat-dim/frame: %d   Actions: %d"
          % (len(y), X.shape[1], X.shape[2], len(ACTIONS)))
    print("Train: %d   Test: %d" % (len(tr), len(te)))
    print("-" * 62)
    print("Majority-class baseline  accuracy: %.4f" % base_acc)
    print("Appearance-only model    accuracy: %.4f" % appo_acc)
    print("Full video understanding accuracy: %.4f" % full_acc)
    print("-" * 62)
    print("Per-action recall (full system):")
    for c in range(len(ACTIONS)):
        mask = y[te] == c
        rec = np.mean(full_pred[mask] == c) if mask.any() else 0.0
        print("  %-8s recall: %.3f" % (LABELS[c], rec))
    print("-" * 62)
    print("Motion modelling beats appearance-only: %s (%.4f > %.4f)"
          % (full_acc > appo_acc + 0.05, full_acc, appo_acc))
    print("Full system beats majority baseline:    %s (%.4f > %.4f)"
          % (full_acc > base_acc, full_acc, base_acc))
