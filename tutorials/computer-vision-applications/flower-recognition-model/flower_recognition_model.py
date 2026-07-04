import numpy as np

# Flower recognition from scratch: radial Fourier descriptors + softmax.
# Each flower is a tiny grayscale silhouette whose PETAL COUNT encodes the class:
#   0 = tulip     (3 petals)
#   1 = lily      (5 petals)
#   2 = daisy     (8 thin petals)
#   3 = sunflower (round disk, ~0 petals)
# We trace the silhouette boundary radius r(theta) around the centroid, then take
# the magnitudes of its angular harmonics (a hand-built DFT). Petal count p shows
# up as a peak at harmonic p, and |DFT| is ROTATION invariant. A multinomial
# logistic (softmax) classifier trained by gradient descent reads those features.

def make_flowers(n, H=28):
    # Draw 4 flower silhouettes as polar shapes r_bound = R0*(1 + amp*cos(p*theta)).
    X = np.zeros((n, H, H)); y = np.arange(n) % 4
    petals = [3, 5, 8, 0]; R0s = [8.0, 8.0, 8.0, 9.5]; amps = [0.55, 0.55, 0.55, 0.05]
    c = (H - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:H]
    for i in range(n):
        k = y[i]; p = petals[k]
        R0 = R0s[k] * (1.0 + 0.08 * np.random.randn())   # size wobble
        oy, ox = np.random.randint(-1, 2, 2)             # position jitter
        phase = np.random.uniform(0, 2 * np.pi)          # random rotation
        dy, dx = yy - c - oy, xx - c - ox
        ang = np.arctan2(dy, dx); rad = np.sqrt(dy ** 2 + dx ** 2)
        r_bound = R0 * (1.0 + amps[k] * np.cos(p * ang + phase))
        X[i][rad <= r_bound] = 1.0                        # petals share R0: only
    X += 0.12 * np.random.randn(n, H, H)                  # shape (not size) differs
    return X, y

def flower_features(X, nbins=48, nharm=10):
    # Radial Fourier descriptor: max boundary radius per angle bin -> harmonic mags.
    N, H, W = X.shape
    M = X > 0.5
    yy, xx = np.mgrid[0:H, 0:W]
    thetas = 2 * np.pi * np.arange(nbins) / nbins
    harm = np.arange(1, nharm + 1)
    C = np.cos(np.outer(harm, thetas))                    # (nharm, nbins) DFT basis
    S = np.sin(np.outer(harm, thetas))
    F = np.zeros((N, nharm + 1))
    for i in range(N):
        m = M[i]
        if m.sum() < 3:
            continue
        ys, xs = yy[m].astype(float), xx[m].astype(float)
        dy, dx = ys - ys.mean(), xs - xs.mean()           # center on centroid
        r = np.sqrt(dy ** 2 + dx ** 2)
        ang = np.arctan2(dy, dx) % (2 * np.pi)
        b = np.minimum((ang / (2 * np.pi / nbins)).astype(int), nbins - 1)
        rb = np.zeros(nbins)
        np.maximum.at(rb, b, r)                            # boundary radius per bin
        rb[rb == 0] = rb[rb > 0].mean() if np.any(rb > 0) else 0.0
        mean_r = rb.mean()
        mag = np.sqrt((C @ (rb - mean_r)) ** 2 + (S @ (rb - mean_r)) ** 2)
        F[i, 0] = mean_r                                   # overall size
        F[i, 1:] = mag / (mean_r + 1e-8)                  # scale-normed petal signal
    return F

class SoftmaxClassifier:
    # Multinomial logistic regression trained by full-batch gradient descent.
    def __init__(self, n_classes, lr=0.5, epochs=400, l2=1e-3):
        self.K, self.lr, self.epochs, self.l2 = n_classes, lr, epochs, l2

    def fit(self, X, y):
        N, D = X.shape
        self.W = np.zeros((D, self.K)); self.b = np.zeros(self.K)
        Y = np.eye(self.K)[y]
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
    names = ["tulip", "lily", "daisy", "sunflower"]; petals = [3, 5, 8, 0]

    Xtr_img, ytr = make_flowers(360)
    Xte_img, yte = make_flowers(160)

    Ftr, Fte = flower_features(Xtr_img), flower_features(Xte_img)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8                # standardize on train
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd

    clf = SoftmaxClassifier(n_classes=4).fit(Ftr, ytr)
    pred = clf.predict(Fte)
    acc = np.mean(pred == yte)
    baseline = np.bincount(ytr, minlength=4).max() / len(ytr)  # majority guess

    # Interpretability: the dominant recovered harmonic should match petal count.
    raw = flower_features(Xte_img)
    print("Flower classes     :", ", ".join(names))
    for k in range(4):
        rec = np.mean(pred[yte == k] == k)
        dom = int(np.round(np.mean((raw[yte == k, 1:].argmax(1) + 1))))
        print(f"  {names[k]:9s} recall {rec:.3f} | recovered petals~{dom} (true {petals[k]})")
    print("Feature dim        :", Ftr.shape[1], "(mean radius + 10 harmonic mags)")
    print("Majority baseline  :", round(float(baseline), 4))
    print("Softmax accuracy   :", round(float(acc), 4))
    print("Beats baseline     :", bool(acc > baseline + 0.3))
