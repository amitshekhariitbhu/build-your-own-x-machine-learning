import numpy as np

# Brain Tumor Detection from scratch.
#
# We synthesize tiny grayscale "MRI slices": a soft round brain region plus
# scanner texture and bright speckle artifacts. A random half of the slices
# also carry a planted tumor -- a smooth bright Gaussian blob somewhere inside
# the brain. The single brightest pixel is NOT a giveaway (speckles are just as
# bright), so a detector must integrate over a blob-shaped neighborhood.
#
# Pipeline, all hand-rolled:
#   1) manual 2D convolution as a matched filter (Gaussian blob detector)
#   2) a few interpretable intensity/shape features per slice
#   3) logistic regression trained with batch gradient descent
# Reported on a held-out split: accuracy / precision / recall / F1 vs majority.

IMG = 28          # slice is IMG x IMG pixels
R = 11.0          # brain disk radius


def gaussian_kernel(size, sigma):
    # Normalized 2D Gaussian used both to plant tumors and as the matched filter.
    ax = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def make_slice(has_tumor, rng):
    # Brain disk: smooth intensity falloff from center, plus tissue texture.
    ax = np.arange(IMG) - (IMG - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    dist = np.sqrt(xx ** 2 + yy ** 2)
    brain = np.clip(1.0 - (dist / R) ** 2, 0, 1) * 0.55
    brain += rng.normal(0, 0.05, brain.shape) * (brain > 0)      # tissue texture
    img = np.clip(brain, 0, 1)

    # Bright speckle artifacts on every slice: isolated hot pixels as bright as
    # a tumor, so peak intensity alone cannot separate the classes.
    for _ in range(rng.randint(4, 9)):
        yx = rng.randint(0, IMG, 2)
        img[yx[0], yx[1]] = min(1.0, img[yx[0], yx[1]] + rng.uniform(0.4, 0.7))

    if has_tumor:
        # Plant a compact bright blob at a random spot well inside the brain.
        while True:
            cy, cx = rng.uniform(-R * 0.6, R * 0.6, 2) + (IMG - 1) / 2.0
            sig = rng.uniform(1.6, 2.4)
            amp = rng.uniform(0.45, 0.7)
            blob = amp * np.exp(-((xx + (IMG - 1) / 2.0 - cx) ** 2 +
                                  (yy + (IMG - 1) / 2.0 - cy) ** 2) / (2 * sig ** 2))
            img = np.clip(img + blob, 0, 1)
            break
    return img


def make_dataset(n, rng):
    X = np.zeros((n, IMG, IMG))
    y = (rng.rand(n) < 0.5).astype(int)          # ~balanced labels
    for i in range(n):
        X[i] = make_slice(y[i], rng)
    return X, y


def conv2d_valid(img, k):
    # Manual valid 2D convolution: accumulate kernel-tap-shifted image slices.
    kh, kw = k.shape
    oh, ow = img.shape[0] - kh + 1, img.shape[1] - kw + 1
    out = np.zeros((oh, ow))
    for i in range(kh):
        for j in range(kw):
            out += k[i, j] * img[i:i + oh, j:j + ow]
    return out


def features(X, matched):
    # Hand-crafted per-slice features. The matched filter is the key one: it
    # smooths away single-pixel speckles but lights up on true blobs.
    feats = []
    for img in X:
        resp = conv2d_valid(img, matched)
        flat = np.sort(resp.ravel())
        feats.append([
            flat[-1],                       # peak matched-filter response
            flat[-8:].mean(),               # mean of strongest blob responses
            img.std(),                      # global contrast
            np.percentile(img, 98),         # raw hot-pixel level (shared by speckle)
            (resp > 0.5 * flat[-1]).mean(), # fraction of strong contiguous response
        ])
    return np.array(feats)


def sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


class LogisticRegression:
    """Binary logistic regression via batch gradient descent (from scratch)."""

    def __init__(self, lr=0.5, epochs=800, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.mu) / self.sd
        n, d = Xs.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.epochs):
            p = sigmoid(Xs @ self.w + self.b)
            g = p - y
            self.w -= self.lr * (Xs.T @ g / n + self.l2 * self.w)
            self.b -= self.lr * g.mean()
        return self

    def predict_proba(self, X):
        return sigmoid((X - self.mu) / self.sd @ self.w + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    X, y = make_dataset(500, rng)
    matched = gaussian_kernel(7, 2.0)                 # blob detector kernel
    F = features(X, matched)

    # Held-out split.
    idx = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]

    clf = LogisticRegression().fit(F[tr], y[tr])
    pred = clf.predict(F[te])
    yte = y[te]

    acc = (pred == yte).mean()
    tp = np.sum((pred == 1) & (yte == 1))
    fp = np.sum((pred == 1) & (yte == 0))
    fn = np.sum((pred == 0) & (yte == 1))
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    majority = max(yte.mean(), 1 - yte.mean())        # predict-the-common-class baseline

    print("Synthetic MRI slices: {} train / {} test  ({}x{} px)".format(
        len(tr), len(te), IMG, IMG))
    print("Tumor prevalence (test): {:.2f}".format(yte.mean()))
    print("-" * 44)
    print("Majority baseline accuracy : {:.3f}".format(majority))
    print("Detector accuracy          : {:.3f}".format(acc))
    print("Precision {:.3f}  Recall {:.3f}  F1 {:.3f}".format(prec, rec, f1))
    print("-" * 44)
    print("PASS" if acc > majority + 0.15 and f1 > 0.8 else "FAIL")
