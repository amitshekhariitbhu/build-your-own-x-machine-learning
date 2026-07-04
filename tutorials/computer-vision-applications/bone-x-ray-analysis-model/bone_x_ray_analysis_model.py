import numpy as np

# Bone X-Ray Analysis Model from scratch (fracture detection).
#
# We synthesize tiny grayscale "X-rays": a dark soft-tissue background with a
# bright vertical bone shaft running down the middle, plus scanner texture and
# scattered dark specks (tissue/overlap artifacts). A random half of the images
# carry a planted FRACTURE -- a thin dark band that cuts fully across the shaft,
# creating a gap. Crucially the specks are as dark as fracture pixels, so the
# single darkest pixel is NOT a giveaway; a detector must find a dark dip that
# spans the WHOLE width of the bone, not one isolated pixel.
#
# Pipeline, all hand-rolled:
#   1) locate the bright bone columns, reduce to a per-row brightness profile
#   2) a 1D "notch" matched filter on the profile finds a full-width dark dip
#   3) a manual 2D Sobel picks up the horizontal fracture edges inside the bone
#   4) logistic regression trained with batch gradient descent
# Reported on a held-out split: accuracy / precision / recall / F1 vs majority.

IMG = 28              # image is IMG x IMG pixels
CX = IMG // 2         # bone shaft center column
HALF = 4              # bone shaft half-width


def make_image(has_fracture, rng):
    ax = np.arange(IMG)
    # Soft-tissue background with mild texture.
    img = 0.15 + rng.normal(0, 0.03, (IMG, IMG))

    # Bright bone shaft: brighter at the core, fading to the edges, with a
    # gentle vertical density gradient and per-image brightness variation.
    col = np.clip(1.0 - ((ax - CX) / (HALF + 1.0)) ** 2, 0, 1)
    shaft = np.outer(np.ones(IMG), col) * rng.uniform(0.6, 0.8)
    shaft += np.outer(np.linspace(-0.05, 0.05, IMG), np.ones(IMG)) * col
    img = np.clip(img + shaft, 0, 1)

    # Scattered dark specks: as dark as a fracture but isolated single pixels,
    # so peak darkness alone cannot separate the two classes.
    for _ in range(rng.randint(6, 13)):
        r, c = rng.randint(0, IMG), rng.randint(CX - HALF, CX + HALF + 1)
        img[r, c] = max(0.0, img[r, c] - rng.uniform(0.4, 0.6))

    if has_fracture:
        # Dark band crossing the full shaft width at a random row (1-2 px thick).
        r0 = rng.randint(6, IMG - 6)
        thick = rng.randint(1, 3)
        drop = rng.uniform(0.4, 0.6)
        img[r0:r0 + thick, CX - HALF:CX + HALF + 1] -= drop
        img = np.clip(img, 0, 1)
    return img


def make_dataset(n, rng):
    X = np.zeros((n, IMG, IMG))
    y = (rng.rand(n) < 0.5).astype(int)          # ~balanced labels
    for i in range(n):
        X[i] = make_image(y[i], rng)
    return X, y


def conv1d_valid(v, k):
    # Manual valid 1D convolution over the row-brightness profile.
    n = len(v) - len(k) + 1
    out = np.zeros(n)
    for j in range(len(k)):
        out += k[j] * v[j:j + n]
    return out


def conv2d_valid(img, k):
    # Manual valid 2D convolution: sum kernel-tap-shifted image slices.
    kh, kw = k.shape
    oh, ow = img.shape[0] - kh + 1, img.shape[1] - kw + 1
    out = np.zeros((oh, ow))
    for i in range(kh):
        for j in range(kw):
            out += k[i, j] * img[i:i + oh, j:j + ow]
    return out


NOTCH = np.array([1., 1., -4., 1., 1.])                 # dip detector on profile
SOBEL_Y = np.array([[-1., -2., -1.],                    # horizontal-edge detector
                    [0., 0., 0.],
                    [1., 2., 1.]])


def features(X):
    # Hand-crafted per-image features built from the two convolutions.
    feats = []
    for img in X:
        col_mean = img.mean(0)
        bone = col_mean > 0.5 * col_mean.max()          # bright bone columns
        p = img[:, bone].mean(1)                         # per-row brightness

        dip = conv1d_valid(p, NOTCH).max()               # strongest full-width dip
        depth = np.median(p) - p.min()                   # how dark the darkest row
        edges = np.abs(conv2d_valid(img, SOBEL_Y))
        feats.append([
            dip,                                         # matched-filter dip response
            depth,                                       # relative darkest-row depth
            edges[:, bone[1:-1]].sum(1).max(),           # peak horizontal-edge row
            p.std(),                                     # profile roughness
            img[:, bone].min(),                          # darkest pixel (shared w/ speck)
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
    F = features(X)

    # Held-out split.
    idx = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]

    clf = LogisticRegression().fit(F[tr], y[tr])
    pred = clf.predict(F[te])
    yte = y[te]

    acc = (pred == yte).mean()
    tp = int(((pred == 1) & (yte == 1)).sum())
    fp = int(((pred == 1) & (yte == 0)).sum())
    fn = int(((pred == 0) & (yte == 1)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    majority = max(yte.mean(), 1 - yte.mean())           # predict-the-common-class
    print("Bone X-Ray Fracture Detection (held-out test set)")
    print(f"  test images        : {len(yte)}  (fracture rate {yte.mean():.2f})")
    print(f"  majority baseline  : {majority:.3f} accuracy")
    print(f"  model accuracy     : {acc:.3f}")
    print(f"  precision          : {prec:.3f}")
    print(f"  recall             : {rec:.3f}")
    print(f"  F1 score           : {f1:.3f}")
    print(f"  --> beats baseline : {acc > majority}  (+{acc - majority:.3f})")
