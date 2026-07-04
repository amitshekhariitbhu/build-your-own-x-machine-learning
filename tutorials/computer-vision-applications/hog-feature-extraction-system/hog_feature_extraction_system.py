import numpy as np

# HOG (Histogram of Oriented Gradients) Feature Extraction System from scratch.
#
# The full Dalal-Triggs pipeline, hand-rolled with numpy:
#   1) image gradients (central differences) -> magnitude + UNSIGNED orientation
#   2) per-cell orientation histograms with SOFT (linear) bin voting
#   3) overlapping 2x2-cell BLOCKS, L2-Hys normalization (clip 0.2, renormalize)
# To prove the descriptor is meaningful we plant a latent factor HOG is built to
# recover -- edge/stripe ORIENTATION. Synthetic 24x24 gratings come in 4 classes
# (stripes at 0/45/90/135 deg). Phase, frequency, contrast and polarity are all
# randomized so the RAW PIXELS of one class look wildly different image-to-image;
# only the oriented-gradient structure is stable. A from-scratch softmax trained
# on HOG features must clear the majority baseline AND beat the same classifier
# trained on raw pixels -- showing HOG captures orientation the pixels hide.

SZ = 24                       # image side (grayscale)
CELL, BINS = 6, 9             # HOG cell size and unsigned-orientation bins
BLK = 2                       # block = BLK x BLK cells (overlapping, stride 1)
ANGLES = [0, 45, 90, 135]     # the 4 planted stripe orientations (degrees)


def make_grating(theta_deg, rng):
    # A sinusoidal stripe pattern whose EDGES run at angle theta_deg. Phase,
    # frequency, contrast and additive noise are random so within-class pixel
    # appearance varies a lot; the gradient orientation stays ~theta_deg.
    yy, xx = np.mgrid[0:SZ, 0:SZ].astype(float)
    th = np.deg2rad(theta_deg)
    proj = xx * np.cos(th) + yy * np.sin(th)      # coordinate across the stripes
    cycles = rng.uniform(2.0, 4.0)                 # spatial frequency
    phase = rng.uniform(0, 2 * np.pi)              # random phase (+ polarity)
    amp = rng.uniform(0.3, 0.5)                     # random contrast
    img = 0.5 + amp * np.sin(2 * np.pi * cycles * proj / SZ + phase)
    img += rng.normal(0, 0.05, img.shape)          # sensor noise
    return np.clip(img, 0, 1)


def hog(img):
    # Hand-rolled Histogram of Oriented Gradients descriptor.
    gx = np.zeros_like(img); gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx) % np.pi               # unsigned orientation 0..pi
    binw = np.pi / BINS

    # Soft (linear) voting: each pixel splits its magnitude across the two
    # nearest orientation bins by distance to their centers.
    c = ang / binw - 0.5
    lo = np.floor(c).astype(int)
    frac = c - lo
    lo_idx = lo % BINS
    hi_idx = (lo + 1) % BINS

    ny, nx = SZ // CELL, SZ // CELL
    cells = np.zeros((ny, nx, BINS))
    for i in range(ny):
        for j in range(nx):
            s = (slice(i * CELL, (i + 1) * CELL), slice(j * CELL, (j + 1) * CELL))
            li, hi = lo_idx[s].ravel(), hi_idx[s].ravel()
            fr, m = frac[s].ravel(), mag[s].ravel()
            cells[i, j] = (np.bincount(li, weights=(1 - fr) * m, minlength=BINS) +
                           np.bincount(hi, weights=fr * m, minlength=BINS))

    # Overlapping block normalization (L2-Hys) over BLK x BLK cell windows.
    feat = []
    for i in range(ny - BLK + 1):
        for j in range(nx - BLK + 1):
            v = cells[i:i + BLK, j:j + BLK].ravel()
            v = v / np.sqrt((v * v).sum() + 1e-6)   # L2
            v = np.minimum(v, 0.2)                    # clip
            v = v / np.sqrt((v * v).sum() + 1e-6)   # renormalize -> L2-Hys
            feat.append(v)
    return np.concatenate(feat)


class Softmax:
    """Multinomial logistic regression trained by full-batch gradient descent."""

    def __init__(self, lr=0.5, epochs=400, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y, K):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xs = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]   # + bias column
        Y = np.eye(K)[y]
        n, d = Xs.shape
        self.W = np.zeros((d, K))
        for _ in range(self.epochs):
            z = Xs @ self.W
            z -= z.max(1, keepdims=True)
            p = np.exp(z); p /= p.sum(1, keepdims=True)
            grad = Xs.T @ (p - Y) / n + self.l2 * self.W
            self.W -= self.lr * grad
        return self

    def predict(self, X):
        Xs = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        return (Xs @ self.W).argmax(1)


def build_dataset(per_class, rng):
    imgs, y = [], []
    for k, a in enumerate(ANGLES):
        for _ in range(per_class):
            imgs.append(make_grating(a, rng))
            y.append(k)
    return np.array(imgs), np.array(y)


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    imgs, y = build_dataset(per_class=150, rng=rng)
    Xhog = np.array([hog(im) for im in imgs])          # HOG descriptors
    Xpix = imgs.reshape(len(imgs), -1)                  # raw-pixel baseline
    K = len(ANGLES)

    idx = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]
    yte = y[te]

    hog_acc = (Softmax().fit(Xhog[tr], y[tr], K).predict(Xhog[te]) == yte).mean()
    pix_acc = (Softmax().fit(Xpix[tr], y[tr], K).predict(Xpix[te]) == yte).mean()
    majority = np.bincount(yte).max() / len(yte)

    # Per-class recall (macro) for the HOG model, to show it works on every angle.
    pred = Softmax().fit(Xhog[tr], y[tr], K).predict(Xhog[te])
    recalls = [(pred[yte == k] == k).mean() for k in range(K)]

    print("Descriptor length / image : {}  ({}x{} px, {}x{} cells, {} bins, "
          "{}x{} blocks)".format(Xhog.shape[1], SZ, SZ, SZ // CELL, SZ // CELL,
                                 BINS, BLK, BLK))
    print("Train / test images       : {} / {}   classes: {} deg".format(
        len(tr), len(te), ANGLES))
    print("-" * 60)
    print("Task: recover stripe ORIENTATION (random phase/freq/contrast)")
    print("  Majority-class baseline accuracy : {:.3f}".format(majority))
    print("  Raw-pixel softmax accuracy       : {:.3f}".format(pix_acc))
    print("  HOG + softmax accuracy           : {:.3f}".format(hog_acc))
    print("  HOG per-orientation recall       : " +
          "  ".join("{}d:{:.2f}".format(a, r) for a, r in zip(ANGLES, recalls)))
    print("-" * 60)
    ok = hog_acc > majority + 0.4 and hog_acc > pix_acc + 0.05 and min(recalls) > 0.6
    print("PASS" if ok else "FAIL")
