import numpy as np

# Pedestrian Detection System from scratch (HOG features + linear SVM).
#
# Synthetic 16x32 grayscale patches. POSITIVES contain a standing pedestrian
# silhouette (round head, vertical torso, two legs); NEGATIVES contain
# background clutter (horizontal / diagonal bars, blobs). Figure polarity is
# randomized so MEAN BRIGHTNESS is not a giveaway -- a detector must read the
# ORIENTED-GRADIENT structure. Everything is hand-rolled:
#   1) image gradients + unsigned-orientation HOG (per-cell histograms, L2-norm)
#   2) a linear soft-margin SVM trained by sub-gradient descent
#   3) a sliding-window detector that localizes the pedestrian in a larger scene
# Reported on held-out data: classification accuracy / F1 vs a majority
# baseline, and sliding-window localization IoU vs a random-window baseline.

W, H = 16, 32          # detection window (a person is tall and narrow)
CELL, BINS = 8, 9      # HOG cell size and unsigned-orientation bins
SH, SW = 48, 40        # scene size for the sliding-window demo


def bg(shape, rng):
    # Textured mid-gray background.
    return np.clip(rng.uniform(0.35, 0.55) + rng.normal(0, 0.05, shape), 0, 1)


def draw_figure(img, cx, top, fg, rng):
    # Paint a pedestrian silhouette (head + torso + two legs) onto img in place.
    Hh, Ww = img.shape
    yy, xx = np.mgrid[0:Hh, 0:Ww]
    head = (xx - cx) ** 2 + (yy - (top + 5)) ** 2 <= 3.0 ** 2
    torso = (np.abs(xx - cx) <= 2.7) & (yy >= top + 8) & (yy <= top + 21)
    legL = (np.abs(xx - (cx - 1.6)) <= 1.0) & (yy >= top + 21) & (yy <= top + 30)
    legR = (np.abs(xx - (cx + 1.6)) <= 1.0) & (yy >= top + 21) & (yy <= top + 30)
    mask = head | torso | legL | legR
    img[mask] = np.clip(fg + rng.normal(0, 0.04, int(mask.sum())), 0, 1)


def add_clutter(img, rng):
    # Random non-pedestrian structure: horizontal bars, blobs, diagonal streaks.
    Hh, Ww = img.shape
    yy, xx = np.mgrid[0:Hh, 0:Ww]
    for _ in range(rng.randint(2, 5)):
        val = np.clip(img.mean() + rng.uniform(0.25, 0.4) *
                      (1 if rng.rand() < 0.5 else -1), 0.05, 0.95)
        t = rng.randint(3)
        if t == 0:                                   # horizontal bar
            y0, th = rng.randint(0, Hh - 3), rng.randint(1, 3)
            img[y0:y0 + th + 1, :] = val
        elif t == 1:                                 # round blob
            cy, cx, r = rng.randint(0, Hh), rng.randint(0, Ww), rng.uniform(2, 4)
            img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = val
        else:                                        # diagonal streak
            c, wdt = rng.uniform(-Ww, Ww), rng.uniform(1.5, 3)
            img[np.abs(yy - xx - c) <= wdt] = val


def make_positive(rng):
    img = bg((H, W), rng)
    contrast = rng.uniform(0.25, 0.4) * (1 if rng.rand() < 0.5 else -1)   # polarity
    fg = np.clip(img.mean() + contrast, 0.05, 0.95)
    draw_figure(img, W / 2 + rng.uniform(-2, 2), rng.randint(-1, 2), fg, rng)
    return img


def make_negative(rng):
    img = bg((H, W), rng)
    add_clutter(img, rng)
    return img


def hog(img):
    # Hand-rolled Histogram of Oriented Gradients.
    gx, gy = np.zeros_like(img), np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx) % np.pi                      # unsigned 0..pi
    b = np.minimum((ang / (np.pi / BINS)).astype(int), BINS - 1)
    ny, nx = img.shape[0] // CELL, img.shape[1] // CELL
    feat = []
    for cy in range(ny):
        for cx in range(nx):
            sl = (slice(cy * CELL, (cy + 1) * CELL), slice(cx * CELL, (cx + 1) * CELL))
            h = np.bincount(b[sl].ravel(), weights=mag[sl].ravel(), minlength=BINS)
            feat.append(h / np.sqrt((h * h).sum() + 1e-6))   # L2-normalize each cell
    return np.concatenate(feat)


class LinearSVM:
    """Soft-margin linear SVM trained by sub-gradient descent (from scratch)."""

    def __init__(self, lr=0.5, epochs=300, C=1.0):
        self.lr, self.epochs, self.C = lr, epochs, C

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.mu) / self.sd
        yy = np.where(y == 1, 1.0, -1.0)
        n, d = Xs.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.epochs):
            viol = yy * (Xs @ self.w + self.b) < 1                 # margin violators
            dw = self.w - self.C * (Xs[viol].T @ yy[viol]) / n     # hinge sub-gradient
            db = -self.C * yy[viol].sum() / n
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self

    def decision(self, X):
        return (X - self.mu) / self.sd @ self.w + self.b

    def predict(self, X):
        return (self.decision(X) >= 0).astype(int)


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def make_scene(rng):
    # Larger image with clutter and one pedestrian at a random location.
    scene = bg((SH, SW), rng)
    add_clutter(scene, rng)
    x0, y0 = rng.randint(0, SW - W + 1), rng.randint(0, SH - H + 1)
    base = scene[y0:y0 + H, x0:x0 + W].mean()
    fg = np.clip(base + rng.uniform(0.25, 0.4) * (1 if rng.rand() < 0.5 else -1), 0.05, 0.95)
    draw_figure(scene, x0 + W / 2, y0, fg, rng)
    return scene, (x0, y0, W, H)


def detect(scene, svm, stride=4):
    # Slide a person-sized window; return the highest-scoring box.
    best, box = -1e9, (0, 0, W, H)
    for y in range(0, SH - H + 1, stride):
        for x in range(0, SW - W + 1, stride):
            s = svm.decision(hog(scene[y:y + H, x:x + W])[None])[0]
            if s > best:
                best, box = s, (x, y, W, H)
    return box


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    # Build a balanced classification dataset of HOG features.
    n = 300
    X = np.array([hog(make_positive(rng)) for _ in range(n)] +
                 [hog(make_negative(rng)) for _ in range(n)])
    y = np.array([1] * n + [0] * n)

    idx = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]

    svm = LinearSVM().fit(X[tr], y[tr])
    pred, yte = svm.predict(X[te]), y[te]
    acc = (pred == yte).mean()
    tp = np.sum((pred == 1) & (yte == 1))
    fp = np.sum((pred == 1) & (yte == 0))
    fn = np.sum((pred == 0) & (yte == 1))
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    majority = max(yte.mean(), 1 - yte.mean())

    # Sliding-window localization on fresh scenes.
    scenes = 40
    det, rnd = [], []
    for _ in range(scenes):
        scene, gt = make_scene(rng)
        det.append(iou(detect(scene, svm), gt))
        rb = (rng.randint(0, SW - W + 1), rng.randint(0, SH - H + 1), W, H)
        rnd.append(iou(rb, gt))
    det, rnd = np.array(det), np.array(rnd)

    print("HOG features/window : {}  ({}x{} px, {}x{} cells, {} bins)".format(
        X.shape[1], H, W, H // CELL, W // CELL, BINS))
    print("Train / test patches: {} / {}".format(len(tr), len(te)))
    print("-" * 52)
    print("[Classification]  window is pedestrian vs background")
    print("  Majority baseline accuracy : {:.3f}".format(majority))
    print("  HOG+SVM accuracy           : {:.3f}".format(acc))
    print("  Precision {:.3f}  Recall {:.3f}  F1 {:.3f}".format(prec, rec, f1))
    print("-" * 52)
    print("[Detection]  sliding-window localization over {} scenes".format(scenes))
    print("  Random-window mean IoU     : {:.3f}".format(rnd.mean()))
    print("  Detector    mean IoU       : {:.3f}".format(det.mean()))
    print("  Detector hit-rate @IoU>0.5 : {:.3f}".format((det > 0.5).mean()))
    print("-" * 52)
    ok = acc > majority + 0.3 and f1 > 0.8 and det.mean() > rnd.mean() + 0.3
    print("PASS" if ok else "FAIL")
