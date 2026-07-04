import numpy as np

# Image classification model built from scratch: a fixed bank of hand-written
# convolution filters extracts edge/blob features, then a softmax classifier
# (multiclass logistic regression, trained by gradient descent) labels images.
# Four planted classes in 12x12 grayscale images:
#   0 = horizontal bars, 1 = vertical bars, 2 = diagonal stripes, 3 = center blob.

H = W = 12


def make_image_dataset(n_per_class=150, noise=0.25):
    """Synthesize grayscale images whose SHAPE/TEXTURE encodes the class."""
    yy, xx = np.mgrid[0:H, 0:W]
    X, y = [], []
    for _ in range(n_per_class):
        # 0: horizontal bars -- bright on alternating rows.
        hbar = ((yy // 2) % 2).astype(float)
        # 1: vertical bars -- bright on alternating columns.
        vbar = ((xx // 2) % 2).astype(float)
        # 2: diagonal stripes -- bright along shifting diagonals.
        diag = (((xx + yy) // 2) % 2).astype(float)
        # 3: center blob -- bright disk in the middle.
        blob = (((xx - W / 2) ** 2 + (yy - H / 2) ** 2) < 12).astype(float)
        for cls, base in enumerate((hbar, vbar, diag, blob)):
            img = base + noise * np.random.randn(H, W)       # additive gray noise
            img *= np.random.uniform(0.8, 1.2)               # intensity jitter
            X.append(img)
            y.append(cls)
    return np.array(X), np.array(y)


# --- Convolutional feature bank (fixed, hand-written kernels) ---------------
FILTERS = np.array([
    [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],      # horizontal edge (Sobel-like)
    [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],      # vertical edge
    [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],      # diagonal edge
    [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], # Laplacian blob/center detector
], float)


def conv2d(X, k):
    """Valid, stride-1 convolution of a batch (N,H,W) with a KxK kernel."""
    kh, kw = k.shape
    oh, ow = H - kh + 1, W - kw + 1
    out = np.zeros((X.shape[0], oh, ow))
    for i in range(kh):                                      # accumulate over
        for j in range(kw):                                 # kernel offsets
            out += X[:, i:i + oh, j:j + ow] * k[i, j]
    return out


def pool2x2(M):
    """Mean-pool feature maps by 2x2 blocks to shrink and add invariance."""
    n, h, w = M.shape
    return M[:, :h // 2 * 2, :w // 2 * 2].reshape(n, h // 2, 2, w // 2, 2).mean((2, 4))


def extract_features(X):
    """Apply each filter -> ReLU -> pool -> flatten into one feature vector."""
    feats = []
    for k in FILTERS:
        fmap = np.maximum(conv2d(X, k), 0.0)                # ReLU keeps edges
        feats.append(pool2x2(fmap).reshape(X.shape[0], -1))
    return np.concatenate(feats, axis=1)


class SoftmaxClassifier:
    """Multiclass logistic regression trained by full-batch gradient descent."""

    def __init__(self, n_classes, lr=0.3, epochs=400, l2=1e-3):
        self.n_classes, self.lr, self.epochs, self.l2 = n_classes, lr, epochs, l2

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)                # stabilize exp
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        n, d = X.shape
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8       # standardize features
        Xs = (X - self.mu) / self.sd
        self.W = np.zeros((d, self.n_classes))
        self.b = np.zeros(self.n_classes)
        Y = np.eye(self.n_classes)[y]                       # one-hot targets
        for _ in range(self.epochs):
            P = self._softmax(Xs @ self.W + self.b)
            dW = Xs.T @ (P - Y) / n + self.l2 * self.W      # cross-entropy grads
            db = (P - Y).mean(0)
            self.W -= self.lr * dW
            self.b -= self.lr * db
        return self

    def predict(self, X):
        Xs = (X - self.mu) / self.sd
        return np.argmax(Xs @ self.W + self.b, axis=1)


if __name__ == "__main__":
    np.random.seed(0)

    X_img, y = make_image_dataset()
    X = extract_features(X_img)                             # conv-bank features

    # Shuffle then hold out 30% for testing.
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    cut = int(0.7 * len(y))
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

    clf = SoftmaxClassifier(n_classes=4).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = (pred == yte).mean()

    # Majority-class baseline: always predict the most frequent training label.
    majority = np.bincount(ytr).argmax()
    base_acc = (yte == majority).mean()

    print("Classes:                   4 (h-bars, v-bars, diagonal, blob)")
    print("Feature dim per image:    ", X.shape[1])
    print("Test samples:             ", len(yte))
    print("Majority baseline acc:     {:.3f}".format(base_acc))
    print("Conv+softmax model acc:    {:.3f}".format(acc))
    print("Improvement over baseline: {:.3f}".format(acc - base_acc))
    print("Beats baseline:", acc > base_acc + 0.3)
