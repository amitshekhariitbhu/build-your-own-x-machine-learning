import numpy as np

# Fake Currency Detection from scratch.
#
# We synthesize tiny grayscale "banknote" images. A GENUINE note carries three
# security features that are expensive to reproduce:
#   * fine microprint texture -- a crisp high-frequency print pattern,
#   * a bright vertical security thread at a fixed column,
#   * a soft watermark -- a smooth bright blob in a corner region.
# A FAKE note independently DEGRADES each of these (smudged print, missing or
# faint thread, washed-out watermark), with per-note strength drawn from ranges
# that OVERLAP the genuine ones. So no single cue separates the classes -- a
# detector must fuse several noisy signals.
#
# Paper tone varies for both classes, so absolute brightness is NOT a giveaway;
# every feature below is built to be tone-invariant (differences / high-pass).
#
# Pipeline, all hand-rolled:
#   1) manual 2D convolution with a Laplacian (high-pass) + Sobel-x kernel
#   2) interpretable, tone-invariant security features per note
#   3) logistic regression trained with batch gradient descent
# Positive class = FAKE. Reported on a held-out split: accuracy / precision /
# recall / F1 vs a majority baseline.

H, W = 24, 48                 # banknote image is H x W pixels
TX = 14                       # nominal security-thread column
WM = (np.s_[4:13], np.s_[33:45])  # watermark region (rows, cols)


def make_note(fake, rng):
    ax_x, ax_y = np.arange(W), np.arange(H)
    xx, yy = np.meshgrid(ax_x, ax_y)

    tone = rng.uniform(0.45, 0.72)                      # paper tone varies a lot
    img = np.full((H, W), tone) + rng.normal(0, 0.02, (H, W))

    # Per-note security-feature strengths. Genuine ranges sit ABOVE fake ranges
    # but overlap at the edges, so any lone feature stays ambiguous.
    if fake:
        mp = rng.uniform(0.02, 0.16)   # microprint amplitude (smudged)
        th = rng.uniform(0.00, 0.18)   # thread brightness (faint/missing)
        wm = rng.uniform(0.00, 0.13)   # watermark strength (washed out)
    else:
        mp = rng.uniform(0.20, 0.30)
        th = rng.uniform(0.26, 0.42)
        wm = rng.uniform(0.18, 0.30)

    # Microprint: crisp high-frequency guilloche pattern (sharp print => strong
    # Laplacian response). Fakes have low amplitude, so the pattern is smeared.
    img += mp * np.sin(1.9 * xx) * np.sin(1.9 * yy)

    # Security thread: bright vertical band with tiny horizontal jitter.
    tcol = TX + rng.randint(-1, 2)
    img[:, tcol:tcol + 2] += th

    # Watermark: smooth bright blob centered in the watermark region.
    cy, cx = 8, 39
    img += wm * np.exp(-((xx - cx) ** 2 / 40.0 + (yy - cy) ** 2 / 20.0))

    return np.clip(img, 0, 1)


def make_dataset(n, rng):
    X = np.zeros((n, H, W))
    y = (rng.rand(n) < 0.5).astype(int)                 # 1 = fake, ~balanced
    for i in range(n):
        X[i] = make_note(y[i], rng)
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


LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)   # high-pass
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)   # vertical edges


def features(X):
    # Hand-crafted, tone-invariant security features.
    feats = []
    for img in X:
        hp = conv2d_valid(img, LAPLACIAN)               # microprint sharpness
        vedge = np.abs(conv2d_valid(img, SOBEL_X))      # vertical-edge magnitude

        col = img.mean(0)                               # column brightness profile
        thread = col[TX - 2:TX + 3].max() - np.median(col)   # thread vs paper
        vband = vedge[:, TX - 3:TX + 3].mean()          # edge energy at thread borders

        wm_region = img[WM]
        watermark = wm_region.mean() - img.mean()       # watermark vs paper (tone-free)

        feats.append([
            np.abs(hp).mean(),     # microprint high-frequency energy
            hp.std(),              # print texture contrast
            thread,                # security-thread brightness contrast
            vband,                 # security-thread edge energy
            watermark,             # watermark region excess brightness
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

    majority = max(yte.mean(), 1 - yte.mean())          # predict-common-class baseline

    print("Synthetic banknotes: {} train / {} test  ({}x{} px)".format(
        len(tr), len(te), H, W))
    print("Fake prevalence (test): {:.2f}".format(yte.mean()))
    print("-" * 46)
    print("Majority baseline accuracy : {:.3f}".format(majority))
    print("Fake detector accuracy     : {:.3f}".format(acc))
    print("Precision {:.3f}  Recall {:.3f}  F1 {:.3f}".format(prec, rec, f1))
    print("-" * 46)
    print("PASS" if acc > majority + 0.15 and f1 > 0.8 else "FAIL")
