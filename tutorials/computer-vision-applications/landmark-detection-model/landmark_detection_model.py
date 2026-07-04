import numpy as np

# Landmark Detection Model from scratch (facial-landmark coordinate regression).
#
# Synthetic 24x24 grayscale "faces": a bright oval head on a dark background,
# with two eyes, a nose and a mouth painted on. The head CENTER, SIZE, the
# eye SPACING and the mouth WIDTH/HEIGHT are all randomized, so the L=5
# landmarks we must recover (left-eye, right-eye, nose, left/right mouth
# corner) shift and stretch across the frame -- a model must READ the face
# geometry, not memorize a fixed layout. Everything is hand-rolled:
#   1) a rasterizer that draws the face and plants exact landmark coords
#   2) a one-hidden-layer MLP (ReLU + sigmoid) trained by full-batch back-prop
#      that regresses every landmark (x, y) directly from the pixels
# Reported on a held-out split: mean per-landmark pixel error and PCK
# (percentage of correct keypoints within a tolerance) vs a mean-shape baseline.

S = 24                      # image side (pixels)
L = 5                       # landmarks: L-eye, R-eye, nose, L-mouth, R-mouth
YY, XX = np.mgrid[0:S, 0:S]


def stamp(img, cx, cy, r, val):
    # Paint a filled disc of radius r centered at (cx, cy).
    img[(XX - cx) ** 2 + (YY - cy) ** 2 <= r * r] = val


def make_sample(rng):
    # Build one face image and its 5 ground-truth landmark coordinates.
    img = np.full((S, S), 0.10) + rng.normal(0, 0.03, (S, S))    # dark textured bg
    cx = rng.uniform(9, 15)                                       # head center x
    cy = rng.uniform(9, 15)                                       # head center y
    rx = rng.uniform(6.0, 7.5)                                    # head half-width
    ry = rng.uniform(7.0, 8.5)                                    # head half-height
    skin = rng.uniform(0.55, 0.72)
    img[((XX - cx) / rx) ** 2 + ((YY - cy) / ry) ** 2 <= 1] = skin   # oval head

    eye_dx = rng.uniform(2.6, 3.6)                                # eye spacing
    eye_y = cy - rng.uniform(1.8, 2.6)                            # eye row
    nose_y = cy + rng.uniform(0.4, 1.2)                           # nose row
    mouth_dx = rng.uniform(1.8, 3.0)                              # mouth half-width
    mouth_y = cy + rng.uniform(3.0, 4.0)                          # mouth row

    le = (cx - eye_dx, eye_y)                                     # left eye
    re = (cx + eye_dx, eye_y)                                     # right eye
    no = (cx, nose_y)                                             # nose tip
    ml = (cx - mouth_dx, mouth_y)                                 # left mouth corner
    mr = (cx + mouth_dx, mouth_y)                                 # right mouth corner

    stamp(img, le[0], le[1], 1.0, 0.02)                          # dark eyes
    stamp(img, re[0], re[1], 1.0, 0.02)
    stamp(img, no[0], no[1], 0.9, 0.30)                          # shaded nose
    for t in np.linspace(0, 1, 9):                               # mouth line
        stamp(img, ml[0] + t * (mr[0] - ml[0]), mouth_y, 0.7, 0.02)
    img = np.clip(img, 0, 1)

    marks = np.array([le, re, no, ml, mr], float)               # (L, 2) = (x, y)
    return img.ravel(), marks.ravel()


class LandmarkNet:
    """One-hidden-layer MLP regressing normalized landmark coords from pixels."""

    def __init__(self, n_in, n_hidden=96, n_out=2 * L, lr=0.35, epochs=1400, l2=1e-4, seed=0):
        r = np.random.RandomState(seed)
        self.W1 = r.randn(n_in, n_hidden) * np.sqrt(2.0 / n_in)   # He init (ReLU)
        self.b1 = np.zeros(n_hidden)
        self.W2 = r.randn(n_hidden, n_out) * np.sqrt(1.0 / n_hidden)
        self.b2 = np.zeros(n_out)
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def _forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)                                    # ReLU
        a2 = 1.0 / (1.0 + np.exp(-(a1 @ self.W2 + self.b2)))      # sigmoid -> [0,1]
        return z1, a1, a2

    def fit(self, X, Y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6            # standardize pixels
        Xs, N = (X - self.mu) / self.sd, len(X)
        for _ in range(self.epochs):
            z1, a1, a2 = self._forward(Xs)
            d2 = (a2 - Y) * a2 * (1 - a2) * (2.0 / N)            # MSE * sigmoid'
            dW2 = a1.T @ d2 + self.l2 * self.W2
            d1 = (d2 @ self.W2.T) * (z1 > 0)                     # ReLU'
            dW1 = Xs.T @ d1 + self.l2 * self.W1
            self.W2 -= self.lr * dW2; self.b2 -= self.lr * d2.sum(0)
            self.W1 -= self.lr * dW1; self.b1 -= self.lr * d1.sum(0)
        return self

    def predict(self, X):
        return self._forward((X - self.mu) / self.sd)[2]        # normalized coords


def mark_errors(pred, true):
    # pred/true: (N, 2L) normalized -> per-landmark Euclidean pixel distances.
    p = pred.reshape(-1, L, 2) * S
    t = true.reshape(-1, L, 2) * S
    return np.sqrt(((p - t) ** 2).sum(-1))                       # (N, L)


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    n = 900
    X = np.zeros((n, S * S)); Y = np.zeros((n, 2 * L))
    for i in range(n):
        X[i], Y[i] = make_sample(rng)
    Y /= S                                                        # normalize coords

    idx = rng.permutation(n); cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]

    net = LandmarkNet(S * S).fit(X[tr], Y[tr])
    pred = net.predict(X[te])

    err = mark_errors(pred, Y[te])                               # (Nte, L) pixels
    base_shape = Y[tr].mean(0)                                   # mean training face
    base_err = mark_errors(np.tile(base_shape, (len(te), 1)), Y[te])

    tol = 2.0                                                    # PCK tolerance (px)
    pck, base_pck = (err < tol).mean(), (base_err < tol).mean()
    names = ["L-eye", "R-eye", "nose", "L-mouth", "R-mouth"]

    print("Faces: {}  ({}x{} px)   Landmarks: {}   Train/Test: {}/{}".format(
        n, S, S, L, len(tr), len(te)))
    print("-" * 58)
    print("Per-landmark mean pixel error (held-out):")
    for k in range(L):
        print("  {:8s} LandmarkNet {:5.2f} px   baseline {:5.2f} px".format(
            names[k], err[:, k].mean(), base_err[:, k].mean()))
    print("-" * 58)
    print("  Mean-shape baseline error : {:5.2f} px".format(base_err.mean()))
    print("  LandmarkNet mean error    : {:5.2f} px".format(err.mean()))
    print("  Baseline PCK@{:.0f}px         : {:.3f}".format(tol, base_pck))
    print("  LandmarkNet PCK@{:.0f}px      : {:.3f}".format(tol, pck))
    print("-" * 58)
    ok = err.mean() < 0.7 * base_err.mean() and pck > base_pck + 0.25
    print("PASS" if ok else "FAIL")
