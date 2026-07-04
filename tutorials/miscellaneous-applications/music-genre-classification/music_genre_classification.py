import numpy as np


class MLPGenreClassifier:
    """One-hidden-layer neural net for music-genre classification, from scratch.

    Standardizes audio features, then trains a ReLU hidden layer + softmax
    output by full-batch gradient descent on the cross-entropy loss.
    Pure numpy -- forward pass, backprop, and weight updates are all manual.
    """

    def __init__(self, hidden=24, lr=0.15, n_iter=700, l2=1e-4, seed=0):
        self.hidden = hidden
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2  # ridge penalty on weights
        self.seed = seed
        self.W1 = self.b1 = self.W2 = self.b2 = None
        self.mu = self.sigma = None
        self.n_classes = None

    def _standardize(self, X, fit=False):
        if fit:
            self.mu = X.mean(axis=0)
            self.sigma = X.std(axis=0) + 1e-8
        return (X - self.mu) / self.sigma

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)  # stabilize
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        X = self._standardize(np.asarray(X, dtype=float), fit=True)
        y = np.asarray(y, dtype=int).reshape(-1)
        n, d = X.shape
        self.n_classes = int(y.max()) + 1
        Y = np.eye(self.n_classes)[y]  # one-hot targets
        # He-style init for the ReLU layer, small init for the output layer
        self.W1 = rng.randn(d, self.hidden) * np.sqrt(2.0 / d)
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.randn(self.hidden, self.n_classes) * 0.01
        self.b2 = np.zeros(self.n_classes)
        for _ in range(self.n_iter):
            # forward
            Z1 = X @ self.W1 + self.b1
            H = np.maximum(0, Z1)            # ReLU activation
            P = self._softmax(H @ self.W2 + self.b2)
            # backprop (cross-entropy + softmax gradient collapses to P - Y)
            dZ2 = (P - Y) / n
            gW2 = H.T @ dZ2 + self.l2 * self.W2
            gb2 = dZ2.sum(axis=0)
            dH = dZ2 @ self.W2.T
            dZ1 = dH * (Z1 > 0)             # ReLU derivative
            gW1 = X.T @ dZ1 + self.l2 * self.W1
            gb1 = dZ1.sum(axis=0)
            # gradient-descent step
            self.W2 -= self.lr * gW2
            self.b2 -= self.lr * gb2
            self.W1 -= self.lr * gW1
            self.b1 -= self.lr * gb1
        return self

    def predict_proba(self, X):
        X = self._standardize(np.asarray(X, dtype=float))
        H = np.maximum(0, X @ self.W1 + self.b1)
        return self._softmax(H @ self.W2 + self.b2)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


GENRES = ["classical", "jazz", "rock", "hiphop", "electronic"]
# Audio-style feature order (as if extracted from a signal):
FEATURES = ["tempo_bpm", "spectral_centroid", "zero_crossing_rate",
            "rms_energy", "spectral_rolloff", "mfcc1", "mfcc2", "harmonic_ratio"]


def make_genre_data(n_per=180, seed=0):
    """Synthetic per-genre audio features with planted, overlapping structure.

    Each genre has a characteristic timbre/rhythm profile -- e.g. classical is
    slow, dark and highly harmonic; electronic is fast, bright and percussive.
    Gaussian spread around those centers makes the classes overlap so the task
    is non-trivial but recoverable.
    """
    rng = np.random.RandomState(seed)
    # per-genre feature means over the 8 features above
    means = {
        "classical":  [ 75, 1400, 0.04, 0.18, 2500,  -120,  40, 0.92],
        "jazz":       [110, 2100, 0.09, 0.30, 3800,   -80,  25, 0.70],
        "rock":       [130, 3000, 0.13, 0.55, 5200,   -40,  10, 0.45],
        "hiphop":     [ 95, 2600, 0.11, 0.62, 4600,   -55,  18, 0.40],
        "electronic": [128, 3400, 0.15, 0.60, 6000,   -30,   5, 0.35],
    }
    # per-feature standard deviations (shared across genres) -> class overlap
    stds = np.array([12, 450, 0.03, 0.10, 900, 22, 12, 0.10])
    X, y = [], []
    for c, g in enumerate(GENRES):
        X.append(rng.normal(means[g], stds, size=(n_per, len(FEATURES))))
        y.append(np.full(n_per, c))
    X = np.vstack(X)
    y = np.concatenate(y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm].astype(int)


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true, y_pred, k):
    f1s = []
    for c in range(k):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_genre_data(n_per=180, seed=0)

    # held-out split: 70% train / 30% test
    n_train = int(0.7 * len(y))
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]

    clf = MLPGenreClassifier(hidden=24, lr=0.15, n_iter=700).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy(yte, pred)
    f1 = macro_f1(yte, pred, clf.n_classes)

    # majority-class baseline + random-guess level
    majority = int(np.bincount(ytr).argmax())
    base_acc = accuracy(yte, np.full_like(yte, majority))
    rand_acc = 1.0 / len(GENRES)

    print("=== Music Genre Classification (1-hidden-layer NN, from scratch) ===")
    print(f"genres                : {', '.join(GENRES)}")
    print(f"features              : {', '.join(FEATURES)}")
    print(f"train / test samples  : {len(ytr)} / {len(yte)}")
    print(f"random-guess accuracy : {rand_acc:.3f}")
    print(f"majority baseline acc : {base_acc:.3f}")
    print(f"neural-net accuracy   : {acc:.3f}")
    print(f"neural-net macro-F1   : {f1:.3f}")
    print(f"improvement over base : +{(acc - base_acc) * 100:.1f} acc points")
    assert acc > base_acc + 0.30, "model should clearly beat the majority baseline"
    print("PASS: genre classifier clearly beats majority + random baselines")
