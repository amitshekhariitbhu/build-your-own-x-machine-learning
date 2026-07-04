import numpy as np

# Bird species identification from scratch: HOG-like edge features + softmax.
# Each bird is a tiny grayscale silhouette whose SHAPE encodes the species:
#   0 = duck   (wide horizontal body  -> strong horizontal edges)
#   1 = heron  (tall vertical body    -> strong vertical edges)
#   2 = owl    (round compact body    -> edges in all orientations)
#   3 = swallow(crossed diagonal wings -> strong diagonal edges)
# We hand-build gradient-orientation histograms (HOG) then train a
# multinomial logistic (softmax) classifier with plain gradient descent.

def make_birds(n, H=16):
    # Draw 4 bird silhouettes on HxH tiles with jitter, size wobble + noise.
    X = np.zeros((n, H, H))
    y = np.arange(n) % 4
    c = (H - 1) / 2.0
    for i in range(n):
        oy, ox = np.random.randint(-1, 2, 2)          # position jitter
        s = 1.0 + 0.12 * np.random.randn()            # size wobble
        yy, xx = np.mgrid[0:H, 0:H]
        dy, dx = (yy - c - oy) / s, (xx - c - ox) / s
        if y[i] == 0:                                  # duck: wide ellipse
            m = (dx / 6.0) ** 2 + (dy / 2.3) ** 2 <= 1.0
        elif y[i] == 1:                                # heron: tall ellipse
            m = (dx / 2.3) ** 2 + (dy / 6.0) ** 2 <= 1.0
        elif y[i] == 2:                                # owl: round body
            m = dx ** 2 + dy ** 2 <= 20.0
        else:                                          # swallow: diagonal X
            r = (dx ** 2 + dy ** 2) <= 36.0
            m = r & ((np.abs(dx - dy) <= 1.3) | (np.abs(dx + dy) <= 1.3))
        X[i][m] = 1.0
    X += 0.12 * np.random.randn(n, H, H)              # sensor noise
    return X, y

def hog_features(X, nbins=9, cell=4):
    # Manual HOG: unsigned gradient-orientation histograms over cell grid.
    N, H, W = X.shape
    gx = np.zeros_like(X); gy = np.zeros_like(X)
    gx[:, :, 1:-1] = X[:, :, 2:] - X[:, :, :-2]       # horizontal gradient
    gy[:, 1:-1, :] = X[:, 2:, :] - X[:, :-2, :]       # vertical gradient
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ori = np.arctan2(gy, gx) % np.pi                  # unsigned angle [0, pi)
    b = np.minimum((ori / (np.pi / nbins)).astype(int), nbins - 1)
    onehot = (b[..., None] == np.arange(nbins)) * mag[..., None]  # (N,H,W,nb)
    ncy, ncx = H // cell, W // cell
    onehot = onehot[:, :ncy * cell, :ncx * cell]
    onehot = onehot.reshape(N, ncy, cell, ncx, cell, nbins)
    feat = onehot.sum(axis=(2, 4)).reshape(N, ncy * ncx * nbins)  # per-cell hist
    feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
    return feat

class SoftmaxClassifier:
    # Multinomial logistic regression trained by full-batch gradient descent.
    def __init__(self, n_classes, lr=0.5, epochs=300, l2=1e-3):
        self.K, self.lr, self.epochs, self.l2 = n_classes, lr, epochs, l2

    def fit(self, X, y):
        N, D = X.shape
        self.W = np.zeros((D, self.K)); self.b = np.zeros(self.K)
        Y = np.eye(self.K)[y]                          # one-hot targets
        for _ in range(self.epochs):
            P = self._softmax(X @ self.W + self.b)
            G = (P - Y) / N
            self.W -= self.lr * (X.T @ G + self.l2 * self.W)
            self.b -= self.lr * G.sum(axis=0)
        return self

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return (X @ self.W + self.b).argmax(axis=1)

if __name__ == "__main__":
    np.random.seed(0)
    names = ["duck", "heron", "owl", "swallow"]

    Xtr_img, ytr = make_birds(320)
    Xte_img, yte = make_birds(160)

    # Standardize HOG features using train statistics only.
    Ftr, Fte = hog_features(Xtr_img), hog_features(Xte_img)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd

    clf = SoftmaxClassifier(n_classes=4).fit(Ftr, ytr)
    pred = clf.predict(Fte)
    acc = np.mean(pred == yte)

    # Baseline: always guess the most common training species.
    baseline = np.bincount(ytr, minlength=4).max() / len(ytr)

    # Per-species recall so we can see every bird is recognized.
    print("Species            :", ", ".join(names))
    for k in range(4):
        rec = np.mean(pred[yte == k] == k)
        print(f"  recall {names[k]:8s}:", round(float(rec), 3))
    print("Feature dim        :", Ftr.shape[1], "(HOG cells x orientations)")
    print("Majority baseline  :", round(float(baseline), 4))
    print("Softmax accuracy   :", round(float(acc), 4))
    print("Beats baseline     :", bool(acc > baseline + 0.3))
