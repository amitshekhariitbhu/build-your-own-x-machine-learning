import numpy as np

# Face Mask Detection from scratch.
#
# We synthesize tiny grayscale "face" images: a skin-toned oval with two dark
# eyes and a nose. HALF the faces are unmasked -- they show a dark mouth over
# textured skin in the lower face. The other half wear a mask -- the lower face
# (nose/mouth/chin) is covered by a brighter, smoother patch with a sharp top
# edge where the mask meets the cheeks.
#
# Absolute brightness is NOT a giveaway: skin tone and mask shade both vary and
# overlap, and the face position jitters. A detector must combine several cues:
# the lower-vs-upper brightness ratio, the horizontal edge left by the mask's
# top border, and whether a dark mouth is still visible.
#
# Pipeline, all hand-rolled:
#   1) manual 2D convolution with a Sobel kernel (horizontal-edge detector)
#   2) interpretable region features per face
#   3) logistic regression trained with batch gradient descent
# Reported on a held-out split: accuracy / precision / recall / F1 vs majority.

IMG = 24                      # face image is IMG x IMG pixels
CY, CX = 10, 11               # nominal face center (row, col)


def blob(xx, yy, cy, cx, r):
    # Soft circular blob in [0,1], used for eyes and mouth.
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * r ** 2))


def make_face(masked, rng):
    ax = np.arange(IMG)
    xx, yy = np.meshgrid(ax, ax)
    jy, jx = rng.randint(-2, 3), rng.randint(-2, 3)   # small position jitter
    cy, cx = CY + jy, CX + jx

    img = np.full((IMG, IMG), 0.15) + rng.normal(0, 0.03, (IMG, IMG))  # background

    # Skin oval: varying tone so absolute brightness cannot separate classes.
    skin = rng.uniform(0.45, 0.62)
    oval = (((xx - cx) / 9.0) ** 2 + ((yy - cy) / 10.0) ** 2) <= 1.0
    img[oval] = skin + rng.normal(0, 0.04, img.shape)[oval]            # skin texture

    # Two dark eyes and a faint nose in the upper face.
    img -= 0.32 * blob(xx, yy, cy - 4, cx - 4, 1.4)
    img -= 0.32 * blob(xx, yy, cy - 4, cx + 4, 1.4)
    img -= 0.10 * blob(xx, yy, cy, cx, 1.2)

    if masked:
        # Mask covers the lower face: brighter, smoother, sharp top boundary.
        # Shade overlaps skin tone, so brightness alone is ambiguous.
        shade = rng.uniform(0.60, 0.90)
        region = oval & (yy >= cy)
        img[region] = shade + rng.normal(0, 0.04, img.shape)[region]   # some texture
        img -= 0.10 * blob(xx, yy, cy + 6, cx, 5.0) * region           # fold shadow
    else:
        # Unmasked: a dark mouth sits on textured skin in the lower face,
        # sometimes with a lighter chin highlight to muddy the brightness cue.
        img -= 0.28 * blob(xx, yy, cy + 6, cx, 2.2)
        img += rng.uniform(0, 0.12) * blob(xx, yy, cy + 8, cx, 3.0)

    return np.clip(img, 0, 1)


def make_dataset(n, rng):
    X = np.zeros((n, IMG, IMG))
    y = (rng.rand(n) < 0.5).astype(int)               # ~balanced labels
    for i in range(n):
        X[i] = make_face(y[i], rng)
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


SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float)  # horizontal edges


def features(X):
    # Hand-crafted region features. Bands are relative to the nominal center;
    # jitter blurs them, so the classifier must weigh several noisy cues.
    upper = slice(CY - 6, CY - 1)     # eyes / forehead rows
    lower = slice(CY + 3, CY + 11)    # mouth / chin rows
    cols = slice(CX - 7, CX + 8)
    feats = []
    for img in X:
        up = img[upper, cols]
        lo = img[lower, cols]
        edge = np.abs(conv2d_valid(img, SOBEL_Y))     # horizontal-edge magnitude
        band = edge[CY - 2:CY + 2, CX - 6:CX + 6]     # near the mask top border
        feats.append([
            lo.mean(),                    # lower-face brightness (mask brighter)
            lo.mean() / (up.mean() + 1e-6),   # lower/upper ratio (tone-invariant)
            lo.std(),                     # lower-face texture (mask smoother)
            band.mean(),                  # edge energy at the mask border
            (lo < 0.35).mean(),           # dark-mouth pixel fraction (unmasked)
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
    tp = np.sum((pred == 1) & (yte == 1))
    fp = np.sum((pred == 1) & (yte == 0))
    fn = np.sum((pred == 0) & (yte == 1))
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    majority = max(yte.mean(), 1 - yte.mean())        # predict-the-common-class baseline

    print("Synthetic faces: {} train / {} test  ({}x{} px)".format(
        len(tr), len(te), IMG, IMG))
    print("Mask prevalence (test): {:.2f}".format(yte.mean()))
    print("-" * 46)
    print("Majority baseline accuracy : {:.3f}".format(majority))
    print("Mask detector accuracy     : {:.3f}".format(acc))
    print("Precision {:.3f}  Recall {:.3f}  F1 {:.3f}".format(prec, rec, f1))
    print("-" * 46)
    print("PASS" if acc > majority + 0.15 and f1 > 0.8 else "FAIL")
