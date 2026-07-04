import numpy as np

# Face Landmark Detection Model from scratch (keypoint regression MLP).
#
# Synthetic 32x32 grayscale "face" images: a bright skin OVAL on a dark
# textured background, with the human layout planted inside -- two dark eyes,
# a nose blob and a dark mouth line. The face CENTER, its width/height, the
# eye separation, the eye row and the mouth width/row are all randomized, so
# the K=5 landmarks we must recover (left/right eye, nose, left/right mouth
# corner) slide and stretch across the frame -- a model has to READ the face
# geometry, not memorize a fixed layout. Everything is hand-rolled:
#   1) a rasterizer that paints an ellipse + disks + a segment and records the
#      exact ground-truth landmark coordinates
#   2) a one-hidden-layer MLP (ReLU + sigmoid) trained by full-batch back-prop
#      that regresses all landmark (x, y) coordinates straight from the pixels
# Reported on a held-out split: mean per-landmark pixel error and PCK
# (percentage of correct keypoints within a tolerance) vs a mean-face baseline.

S = 32                      # image side (pixels)
K = 5                       # landmarks: L-eye, R-eye, nose, L-mouth, R-mouth
YY, XX = np.mgrid[0:S, 0:S]


def disk(img, cx, cy, r, val):
    # Paint a filled disk of radius r.
    img[(XX - cx) ** 2 + (YY - cy) ** 2 <= r * r] = val


def segment(img, p0, p1, val, thick):
    # Paint a thick line segment via point-to-segment distance.
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy + 1e-9
    t = np.clip(((XX - x0) * dx + (YY - y0) * dy) / L2, 0, 1)
    px, py = x0 + t * dx, y0 + t * dy
    img[np.sqrt((XX - px) ** 2 + (YY - py) ** 2) <= thick] = val


def make_sample(rng):
    # Build one face image and its 5 ground-truth landmark coordinates.
    img = np.full((S, S), 0.15) + rng.normal(0, 0.03, (S, S))    # dark textured bg
    cx = rng.uniform(9, 23)                                      # face center x (roams)
    cy = rng.uniform(10, 22)                                     # face center y (roams)
    rw = rng.uniform(5.5, 7.0)                                   # face half-width
    rh = rng.uniform(7.0, 8.5)                                   # face half-height

    skin = rng.uniform(0.60, 0.80)                               # bright skin oval
    img[((XX - cx) / rw) ** 2 + ((YY - cy) / rh) ** 2 <= 1] = skin

    ew = rng.uniform(0.42, 0.55) * rw                            # eye half-separation
    ey = cy - rng.uniform(0.22, 0.36) * rh                       # eye row
    le, re = (cx - ew, ey), (cx + ew, ey)                        # eye centers
    ny = cy + rng.uniform(0.04, 0.18) * rh                       # nose row
    nose = (cx + rng.uniform(-0.6, 0.6), ny)                     # nose tip
    my = cy + rng.uniform(0.42, 0.60) * rh                       # mouth row
    mw = rng.uniform(0.34, 0.50) * rw                            # mouth half-width
    lm, rm = (cx - mw, my), (cx + mw, my)                        # mouth corners

    dark = rng.uniform(0.05, 0.20)                               # dark features
    disk(img, le[0], le[1], 1.5, dark)                          # eyes
    disk(img, re[0], re[1], 1.5, dark)
    disk(img, nose[0], nose[1], 1.2, skin - 0.25)              # shaded nose
    segment(img, lm, rm, dark, 1.1)                            # mouth
    img = np.clip(img, 0, 1)

    marks = np.array([le, re, nose, lm, rm], float)             # (K, 2) = (x, y)
    return img.ravel(), marks.ravel()


class LandmarkNet:
    """One-hidden-layer MLP regressing normalized landmark coords from pixels."""

    def __init__(self, n_in, n_hidden=96, n_out=2 * K, lr=0.35, epochs=1200, l2=1e-4, seed=0):
        r = np.random.RandomState(seed)
        self.W1 = r.randn(n_in, n_hidden) * np.sqrt(2.0 / n_in)  # He init (ReLU)
        self.b1 = np.zeros(n_hidden)
        self.W2 = r.randn(n_hidden, n_out) * np.sqrt(1.0 / n_hidden)
        self.b2 = np.zeros(n_out)
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)                                   # ReLU
        a2 = 1.0 / (1.0 + np.exp(-(a1 @ self.W2 + self.b2)))     # sigmoid -> [0,1]
        return z1, a1, a2

    def fit(self, X, Y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6           # standardize pixels
        Xs, N = (X - self.mu) / self.sd, len(X)
        for _ in range(self.epochs):
            z1, a1, a2 = self._forward(Xs)
            d2 = (a2 - Y) * a2 * (1 - a2) * (2.0 / N)           # MSE * sigmoid'
            dW2 = a1.T @ d2 + self.l2 * self.W2
            d1 = (d2 @ self.W2.T) * (z1 > 0)                    # ReLU'
            dW1 = Xs.T @ d1 + self.l2 * self.W1
            self.W2 -= self.lr * dW2; self.b2 -= self.lr * d2.sum(0)
            self.W1 -= self.lr * dW1; self.b1 -= self.lr * d1.sum(0)
        return self

    def predict(self, X):
        return self._forward((X - self.mu) / self.sd)[2]        # normalized coords


def mark_errors(pred, true):
    # pred/true: (N, 2K) normalized -> per-landmark Euclidean pixel distances.
    p = pred.reshape(-1, K, 2) * S
    t = true.reshape(-1, K, 2) * S
    return np.sqrt(((p - t) ** 2).sum(-1))                      # (N, K)


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    n = 1000
    X = np.zeros((n, S * S)); Y = np.zeros((n, 2 * K))
    for i in range(n):
        X[i], Y[i] = make_sample(rng)
    Y /= S                                                       # normalize coords

    idx = rng.permutation(n); cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]

    net = LandmarkNet(S * S).fit(X[tr], Y[tr])
    pred = net.predict(X[te])

    err = mark_errors(pred, Y[te])                              # (Nte, K) pixels
    base_face = Y[tr].mean(0)                                   # mean training face
    base_err = mark_errors(np.tile(base_face, (len(te), 1)), Y[te])

    tol = 2.5                                                   # PCK tolerance (px)
    pck, base_pck = (err < tol).mean(), (base_err < tol).mean()
    names = ["L-eye", "R-eye", "nose", "L-mouth", "R-mouth"]

    print("Faces: {}  ({}x{} px)   Landmarks: {}   Train/Test: {}/{}".format(
        n, S, S, K, len(tr), len(te)))
    print("-" * 58)
    print("Per-landmark mean pixel error (held-out):")
    for k in range(K):
        print("  {:8s}  Net {:5.2f} px   baseline {:5.2f} px".format(
            names[k], err[:, k].mean(), base_err[:, k].mean()))
    print("-" * 58)
    print("  Mean-face baseline error : {:5.2f} px".format(base_err.mean()))
    print("  LandmarkNet mean error   : {:5.2f} px".format(err.mean()))
    print("  Baseline PCK@{:.1f}px       : {:.3f}".format(tol, base_pck))
    print("  LandmarkNet PCK@{:.1f}px    : {:.3f}".format(tol, pck))
    print("-" * 58)
    ok = err.mean() < 0.6 * base_err.mean() and pck > base_pck + 0.30
    print("PASS" if ok else "FAIL")
