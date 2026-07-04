import numpy as np

# Human Face Detection System from scratch (Viola-Jones style).
#
# We synthesize tiny 24x24 grayscale patches. FACE patches are a skin oval with
# the human layout: a bright forehead, two dark eyes with eyebrows, a bright
# vertical nose bridge between them, bright cheeks, and a dark mouth below.
# NON-FACE patches are structured clutter -- random bars, blobs and gradients
# with matched brightness -- so the detector cannot cheat on average intensity
# and must recognise the actual arrangement of light/dark regions.
#
# Everything is hand-rolled:
#   1) integral image  -> O(1) rectangle sums
#   2) a small bank of Haar-like features (the eyes/nose/mouth contrasts a face
#      Viola-Jones cascade keys on), variance-normalised per patch for lighting
#      invariance
#   3) logistic regression trained by batch gradient descent
#
# We report (a) held-out face-vs-nonface accuracy / precision / recall / F1 vs a
# majority baseline, and (b) a sliding-window LOCALISER's mean IoU vs a random
# window -- proving the system both classifies and locates faces.

P = 24                         # patch side (px)
CY, CX = 11, 11                # nominal face center


def blob(xx, yy, cy, cx, r):
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * r ** 2))


def make_face(rng):
    ax = np.arange(P)
    xx, yy = np.meshgrid(ax, ax)
    jy, jx = rng.randint(-2, 3), rng.randint(-2, 3)   # position jitter
    cy, cx = CY + jy, CX + jx
    img = 0.20 + rng.normal(0, 0.03, (P, P))          # dark background
    skin = rng.uniform(0.55, 0.78)                    # varying skin tone
    oval = (((xx - cx) / 9.0) ** 2 + ((yy - cy) / 10.5) ** 2) <= 1.0
    img[oval] = skin + rng.normal(0, 0.04, (P, P))[oval]
    img -= 0.35 * blob(xx, yy, cy - 3, cx - 4, 1.7)   # left eye
    img -= 0.35 * blob(xx, yy, cy - 3, cx + 4, 1.7)   # right eye
    img -= 0.14 * blob(xx, yy, cy - 5, cx - 4, 2.0)   # left brow
    img -= 0.14 * blob(xx, yy, cy - 5, cx + 4, 2.0)   # right brow
    img += 0.10 * blob(xx, yy, cy, cx, 1.5)           # bright nose bridge
    img -= 0.27 * blob(xx, yy, cy + 6, cx, 2.3)       # mouth
    return np.clip(img, 0, 1)


def make_clutter(rng):
    ax = np.arange(P)
    xx, yy = np.meshgrid(ax, ax)
    img = rng.uniform(0.25, 0.65) + rng.normal(0, 0.05, (P, P))
    for _ in range(rng.randint(2, 6)):                # planted structure, not pure noise
        kind = rng.randint(3)
        if kind == 0:                                 # bar / edge
            r, w = rng.randint(0, P), rng.randint(1, 5)
            img[r:r + w, :] += rng.uniform(-0.4, 0.4)
        elif kind == 1:                               # blob
            img += rng.uniform(-0.4, 0.4) * blob(
                xx, yy, rng.randint(0, P), rng.randint(0, P), rng.uniform(1.5, 4))
        else:                                         # gradient
            img += rng.uniform(-0.3, 0.3) * ((xx if rng.rand() < 0.5 else yy) / P)
    return np.clip(img, 0, 1)


def integral(img):
    return np.pad(img, ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def rect_sum(ii, r0, c0, r1, c1):
    return ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0]


# Haar-like bank: each feature is a list of (r0,c0,r1,c1,weight) rectangles,
# each contributing its MEAN (sum/area) so features are scale-consistent.
BANK = [
    [(2, 3, 6, 21, +1), (6, 3, 11, 21, -1)],                 # forehead > eye band
    [(6, 3, 11, 21, +1), (12, 3, 17, 21, -1)],               # eye band < cheeks
    [(5, 2, 12, 9, -1), (5, 9, 12, 15, +2), (5, 15, 12, 22, -1)],  # nose bridge > eyes
    [(13, 5, 16, 19, +1), (16, 5, 20, 19, -1)],              # above-mouth > mouth
    [(6, 2, 11, 8, -1), (6, 9, 11, 15, +2), (6, 15, 11, 21, -1)],  # two eyes vs bridge
    [(2, 3, 22, 21, +1)],                                    # overall region energy
]


def haar_features(patch):
    m, s = patch.mean(), patch.std() + 1e-6           # variance normalisation
    ii = integral((patch - m) / s)
    out = np.empty(len(BANK))
    for k, rects in enumerate(BANK):
        out[k] = sum(w * rect_sum(ii, r0, c0, r1, c1) / ((r1 - r0) * (c1 - c0))
                     for (r0, c0, r1, c1, w) in rects)
    return out


def featurize(patches):
    return np.array([haar_features(p) for p in patches])


def sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


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
            g = sigmoid(Xs @ self.w + self.b) - y
            self.w -= self.lr * (Xs.T @ g / n + self.l2 * self.w)
            self.b -= self.lr * g.mean()
        return self

    def score(self, X):
        return sigmoid((X - self.mu) / self.sd @ self.w + self.b)

    def predict(self, X):
        return (self.score(X) >= 0.5).astype(int)


def iou(b1, b2):
    r0, c0 = max(b1[0], b2[0]), max(b1[1], b2[1])
    r1, c1 = min(b1[0] + P, b2[0] + P), min(b1[1] + P, b2[1] + P)
    inter = max(0, r1 - r0) * max(0, c1 - c0)
    return inter / (2 * P * P - inter)


def make_scene(rng, S=44):
    scene = make_clutter(rng)                         # start from clutter, tile to S
    scene = np.pad(scene, ((0, S - P), (0, S - P)), mode="reflect")
    scene += rng.normal(0, 0.04, scene.shape)
    r, c = rng.randint(0, S - P + 1), rng.randint(0, S - P + 1)
    scene[r:r + P, c:c + P] = make_face(rng)          # plant one face
    return np.clip(scene, 0, 1), (r, c)


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    # ---- Face vs non-face classification on a held-out split ----
    n = 800
    y = (rng.rand(n) < 0.5).astype(int)
    patches = [make_face(rng) if yi else make_clutter(rng) for yi in y]
    F = featurize(patches)

    idx = rng.permutation(n)
    cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    clf = LogisticRegression().fit(F[tr], y[tr])
    pred, yte = clf.predict(F[te]), y[te]

    acc = (pred == yte).mean()
    tp = np.sum((pred == 1) & (yte == 1)); fp = np.sum((pred == 1) & (yte == 0))
    fn = np.sum((pred == 0) & (yte == 1))
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    majority = max(yte.mean(), 1 - yte.mean())

    # ---- Sliding-window localiser: mean IoU vs a random window ----
    S, stride, N = 44, 2, 40
    det_iou, rnd_iou = [], []
    for _ in range(N):
        scene, true = make_scene(rng)
        best, best_box = -1.0, (0, 0)
        for r in range(0, S - P + 1, stride):
            for c in range(0, S - P + 1, stride):
                sc = clf.score(haar_features(scene[r:r + P, c:c + P])[None])[0]
                if sc > best:
                    best, best_box = sc, (r, c)
        det_iou.append(iou(best_box, true))
        rnd = (rng.randint(0, S - P + 1), rng.randint(0, S - P + 1))
        rnd_iou.append(iou(rnd, true))
    det_iou, rnd_iou = np.mean(det_iou), np.mean(rnd_iou)

    print("Synthetic patches: {} train / {} test  ({}x{} px)".format(
        len(tr), len(te), P, P))
    print("Face prevalence (test): {:.2f}".format(yte.mean()))
    print("-" * 48)
    print("CLASSIFICATION (face vs non-face)")
    print("  Majority baseline accuracy : {:.3f}".format(majority))
    print("  Face detector accuracy     : {:.3f}".format(acc))
    print("  Precision {:.3f}  Recall {:.3f}  F1 {:.3f}".format(prec, rec, f1))
    print("-" * 48)
    print("LOCALISATION (sliding window over {} scenes)".format(N))
    print("  Random-window mean IoU     : {:.3f}".format(rnd_iou))
    print("  Detector mean IoU          : {:.3f}".format(det_iou))
    print("-" * 48)
    ok = acc > majority + 0.15 and f1 > 0.85 and det_iou > rnd_iou + 0.3
    print("PASS" if ok else "FAIL")
