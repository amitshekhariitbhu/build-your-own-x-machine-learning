import numpy as np

# 5x7 dot-matrix templates for digits 0-9 ("1" = ink, "0" = background).
DIGIT_TEMPLATES = {
    0: ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    1: ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    2: ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    3: ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    4: ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    5: ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    6: ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    7: ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    8: ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    9: ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}


def make_digit_images(n_per_class=120, noise=0.35, flip_prob=0.06):
    """Synthesize noisy 7x5 grayscale digit images with planted class structure."""
    templates = {d: np.array([[int(c) for c in row] for row in rows], float)
                 for d, rows in DIGIT_TEMPLATES.items()}
    X, y = [], []
    for d, base in templates.items():
        for _ in range(n_per_class):
            img = base.copy()
            img += noise * np.random.randn(*img.shape)          # additive gray noise
            flip = np.random.rand(*img.shape) < flip_prob        # salt-and-pepper flips
            img[flip] = 1.0 - base[flip]
            img *= np.random.uniform(0.8, 1.2)                   # global intensity jitter
            X.append(img.ravel())
            y.append(d)
    return np.array(X), np.array(y)


class SoftmaxClassifier:
    """Multiclass logistic regression trained by full-batch gradient descent."""

    def __init__(self, n_classes, lr=0.5, epochs=300, l2=1e-3):
        self.n_classes = n_classes
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)                     # stabilize exp
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        n, d = X.shape
        self.W = np.zeros((d, self.n_classes))
        self.b = np.zeros(self.n_classes)
        Y = np.eye(self.n_classes)[y]                            # one-hot targets

        for _ in range(self.epochs):
            P = self._softmax(X @ self.W + self.b)
            # Cross-entropy gradients (P - Y), plus L2 shrinkage on weights.
            dW = X.T @ (P - Y) / n + self.l2 * self.W
            db = (P - Y).mean(axis=0)
            self.W -= self.lr * dW
            self.b -= self.lr * db
        return self

    def predict(self, X):
        return np.argmax(X @ self.W + self.b, axis=1)


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_digit_images()

    # Shuffle then hold out 30% for testing.
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    cut = int(0.7 * len(y))
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

    clf = SoftmaxClassifier(n_classes=10).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = (pred == yte).mean()

    # Majority-class baseline: always predict the most frequent training label.
    majority = np.bincount(ytr).argmax()
    base_acc = (yte == majority).mean()

    print("Test samples:            ", len(yte))
    print("Majority baseline acc:    {:.3f}".format(base_acc))
    print("Softmax classifier acc:   {:.3f}".format(acc))
    print("Improvement over baseline: {:.3f}".format(acc - base_acc))
    print("Beats baseline:", acc > base_acc + 0.3)
