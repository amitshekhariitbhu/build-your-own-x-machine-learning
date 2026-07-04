import numpy as np

# 5x7 dot-matrix templates for a set of handwritten characters ("1" = ink).
CHAR_TEMPLATES = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
}
CHARS = list(CHAR_TEMPLATES)                                     # class index -> letter


def _shift(img, dy, dx):
    """Translate an image by (dy, dx), padding vacated pixels with background."""
    out = np.zeros_like(img)
    ys, xs = slice(max(0, dy), 7 + min(0, dy)), slice(max(0, dx), 5 + min(0, dx))
    yt, xt = slice(max(0, -dy), 7 - max(0, dy)), slice(max(0, -dx), 5 - max(0, dx))
    out[ys, xs] = img[yt, xt]
    return out


def make_char_images(n_per_class=200, noise=0.18, flip_prob=0.03):
    """Synthesize noisy 7x5 grayscale handwriting with planted per-letter structure."""
    templates = {c: np.array([[int(v) for v in row] for row in rows], float)
                 for c, rows in CHAR_TEMPLATES.items()}
    X, y = [], []
    for k, c in enumerate(CHARS):
        base = templates[c]
        for _ in range(n_per_class):
            img = _shift(base, np.random.randint(-1, 2), np.random.randint(-1, 2))
            img = img + noise * np.random.randn(*img.shape)      # additive gray noise
            flip = np.random.rand(*img.shape) < flip_prob        # salt-and-pepper flips
            img[flip] = 1.0 - img[flip]
            img *= np.random.uniform(0.8, 1.2)                   # global intensity jitter
            X.append(img.ravel())
            y.append(k)
    return np.array(X), np.array(y)


class MLPClassifier:
    """One-hidden-layer neural net (ReLU + softmax) trained by manual backprop."""

    def __init__(self, n_in, n_hidden, n_out, lr=0.2, epochs=700, l2=1e-4, seed=1):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(n_in, n_hidden) * np.sqrt(2.0 / n_in)     # He init
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.randn(n_hidden, n_out) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(n_out)
        self.n_out, self.lr, self.epochs, self.l2 = n_out, lr, epochs, l2

    @staticmethod
    def _softmax(Z):
        E = np.exp(Z - Z.max(axis=1, keepdims=True))             # stabilize exp
        return E / E.sum(axis=1, keepdims=True)

    def _forward(self, X):
        A1 = np.maximum(0, X @ self.W1 + self.b1)                # ReLU hidden activation
        P = self._softmax(A1 @ self.W2 + self.b2)
        return A1, P

    def fit(self, X, y):
        n = len(y)
        Y = np.eye(self.n_out)[y]                                # one-hot targets
        for _ in range(self.epochs):
            A1, P = self._forward(X)
            # Output layer gradients from cross-entropy (P - Y).
            dZ2 = (P - Y) / n
            dW2 = A1.T @ dZ2 + self.l2 * self.W2
            db2 = dZ2.sum(axis=0)
            # Backprop through ReLU into the hidden layer.
            dA1 = (dZ2 @ self.W2.T) * (A1 > 0)
            dW1 = X.T @ dA1 + self.l2 * self.W1
            db1 = dA1.sum(axis=0)
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
        return self

    def predict(self, X):
        return np.argmax(self._forward(X)[1], axis=1)


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_char_images()

    # Shuffle then hold out 30% for testing.
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    cut = int(0.7 * len(y))
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

    clf = MLPClassifier(n_in=35, n_hidden=96, n_out=len(CHARS)).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = (pred == yte).mean()

    # Majority-class baseline: always predict the most frequent training label.
    majority = np.bincount(ytr).argmax()
    base_acc = (yte == majority).mean()

    print("Characters:               ", "".join(CHARS))
    print("Test samples:             ", len(yte))
    print("Majority baseline acc:     {:.3f}".format(base_acc))
    print("MLP classifier acc:        {:.3f}".format(acc))
    print("Improvement over baseline: {:.3f}".format(acc - base_acc))
    print("Sample truth -> pred:     ",
          " ".join("{}->{}".format(CHARS[t], CHARS[p]) for t, p in zip(yte[:8], pred[:8])))
    print("Beats baseline:", acc > base_acc + 0.3)
