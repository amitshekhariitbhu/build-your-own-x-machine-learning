import numpy as np

# Image recognition system built from scratch: recognize which digit (0-9) is
# drawn in a small noisy grayscale image. We render each digit from a 7x5 dot
# font onto a larger canvas at a RANDOM shift (so raw pixels don't line up),
# extract translation-tolerant HOG-like gradient-orientation features by hand,
# and recognize with a from-scratch k-nearest-neighbours vote. Ten planted
# classes -> a real 10-way recognition task, scored on a held-out split.

# 7-rows x 5-cols dot-matrix glyphs for the ten digits (the LATENT structure).
FONT = {
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
GH, GW = 7, 5          # glyph size
H, W = 9, 7            # canvas size (room to shift the glyph around)


def glyph(d):
    return np.array([[int(c) for c in row] for row in FONT[d]], float)


def make_dataset(n_per_class=100, noise=0.15):
    """Render each digit at a random shift + intensity jitter + gray noise."""
    X, y = [], []
    for d in range(10):
        g = glyph(d)
        for _ in range(n_per_class):
            canvas = np.zeros((H, W))
            dy = np.random.randint(0, H - GH + 1)          # random vertical shift
            dx = np.random.randint(0, W - GW + 1)          # random horizontal shift
            canvas[dy:dy + GH, dx:dx + GW] = g
            canvas *= np.random.uniform(0.8, 1.2)          # brightness jitter
            canvas += noise * np.random.randn(H, W)        # additive gray noise
            X.append(canvas)
            y.append(d)
    return np.array(X), np.array(y)


def hog_features(X, nbins=8, cells=3):
    """Hand-written HOG: per-cell histograms of gradient ORIENTATION, magnitude
    weighted. Pooling over cells makes it tolerant to the random glyph shift."""
    N = X.shape[0]
    gx = np.zeros_like(X)
    gy = np.zeros_like(X)
    gx[:, :, 1:-1] = X[:, :, 2:] - X[:, :, :-2]            # central differences
    gy[:, 1:-1, :] = X[:, 2:, :] - X[:, :-2, :]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0           # unsigned orientation
    b = np.minimum((ang / (180.0 / nbins)).astype(int), nbins - 1)
    rs = np.linspace(0, H, cells + 1).astype(int)          # cell row edges
    cs = np.linspace(0, W, cells + 1).astype(int)          # cell col edges
    feats = []
    for i in range(cells):
        for j in range(cells):
            m = mag[:, rs[i]:rs[i + 1], cs[j]:cs[j + 1]].reshape(N, -1)
            bb = b[:, rs[i]:rs[i + 1], cs[j]:cs[j + 1]].reshape(N, -1)
            hist = np.stack([(m * (bb == k)).sum(1) for k in range(nbins)], axis=1)
            feats.append(hist)
    F = np.concatenate(feats, axis=1)                      # (N, cells*cells*nbins)
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)   # L2 normalize


class KNNRecognizer:
    """From-scratch k-nearest-neighbours classifier (Euclidean, majority vote)."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = X, y
        return self

    def predict(self, X):
        # Pairwise distances via (a-b)^2 = a^2 - 2ab + b^2, then vote on k nearest.
        d2 = (X ** 2).sum(1)[:, None] - 2 * X @ self.X.T + (self.X ** 2).sum(1)[None, :]
        nn = np.argsort(d2, axis=1)[:, :self.k]
        return np.array([np.bincount(self.y[row]).argmax() for row in nn])


if __name__ == "__main__":
    np.random.seed(0)

    X_img, y = make_dataset()
    X = hog_features(X_img)                                # hand-built features

    idx = np.random.permutation(len(y))                   # shuffle then split
    X, y = X[idx], y[idx]
    cut = int(0.7 * len(y))
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

    model = KNNRecognizer(k=5).fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = (pred == yte).mean()

    majority = np.bincount(ytr).argmax()                  # most frequent digit
    base_acc = (yte == majority).mean()

    print("Classes:                   10 digits (0-9)")
    print("Image size:                {}x{} grayscale".format(H, W))
    print("Feature dim per image:    ", X.shape[1], "(HOG-like)")
    print("Train / test samples:     ", len(ytr), "/", len(yte))
    print("Majority baseline acc:     {:.3f}".format(base_acc))
    print("HOG + kNN recognizer acc:  {:.3f}".format(acc))
    print("Improvement over baseline: {:.3f}".format(acc - base_acc))
    print("Beats baseline:", acc > base_acc + 0.4)
