import numpy as np
import time

OBJ = 6  # side length (px) of the square object we detect


def sigmoid(z):
    # Numerically-stable logistic squashing to (0, 1).
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


class PatchClassifier:
    """Logistic regression on flattened OBJxOBJ pixels, trained by full-batch
    gradient descent. This is the detector's per-window head: given a patch it
    returns P(object). A solid bright square scores high; background and bright
    line/edge clutter score low."""

    def __init__(self, lr=0.5, epochs=400, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        X = X.reshape(len(X), -1)
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.epochs):
            g = sigmoid(X @ self.w + self.b) - y            # dL/dlogit
            self.w -= self.lr * (X.T @ g / n + self.l2 * self.w)
            self.b -= self.lr * g.mean()
        return self

    def score(self, X):
        X = X.reshape(len(X), -1)
        return sigmoid(X @ self.w + self.b)


def make_patches(rng, n):
    # Balanced training patches: object squares vs background / line clutter.
    X, y = np.zeros((n, OBJ, OBJ)), np.zeros(n)
    for i in range(n):
        if i % 2 == 0:                                      # object
            X[i] = 1.0 + 0.08 * rng.randn(OBJ, OBJ)
            y[i] = 1
        else:                                               # negative
            X[i] = 0.08 * rng.randn(OBJ, OBJ)
            if rng.rand() < 0.5:                            # bright line clutter
                if rng.rand() < 0.5:
                    X[i, rng.randint(OBJ)] = 0.9
                else:
                    X[i, :, rng.randint(OBJ)] = 0.9
    return np.clip(X, 0, 1.5), y


def make_scene(rng, H=32, W=32):
    # A frame: 0-3 planted square objects + bright line clutter + noise.
    img = 0.08 * rng.randn(H, W)
    boxes = []
    for _ in range(rng.randint(1, 4)):
        for _try in range(10):
            r, c = rng.randint(0, H - OBJ + 1), rng.randint(0, W - OBJ + 1)
            if all(iou((r, c), b) == 0 for b in boxes):     # no overlap
                img[r:r + OBJ, c:c + OBJ] = 1.0 + 0.05 * rng.randn(OBJ, OBJ)
                boxes.append((r, c))
                break
    for _ in range(rng.randint(2, 5)):                      # distractor lines
        if rng.rand() < 0.5:
            r, c = rng.randint(0, H), rng.randint(0, W - OBJ)
            img[r, c:c + OBJ] = 0.9
        else:
            r, c = rng.randint(0, H - OBJ), rng.randint(0, W)
            img[r:r + OBJ, c] = 0.9
    return np.clip(img, 0, 1.5), boxes


def iou(a, b):
    # IoU of two OBJxOBJ squares given their (row, col) top-left corners.
    r1, c1 = max(a[0], b[0]), max(a[1], b[1])
    r2, c2 = min(a[0], b[0]) + OBJ, min(a[1], b[1]) + OBJ
    inter = max(0, r2 - r1) * max(0, c2 - c1)
    return inter / (2 * OBJ * OBJ - inter)


def nms(boxes, scores, iou_thr=0.1):
    # Greedy non-maximum suppression: keep top score, drop overlaps, repeat.
    order, kept, used = np.argsort(-scores), [], set()
    for i in order:
        if i in used:
            continue
        kept.append((tuple(boxes[i]), scores[i]))
        used.update(j for j in order
                    if j not in used and iou(boxes[i], boxes[j]) > iou_thr)
    return kept


def detect(clf, img, stride=2, thr=0.6):
    # Slide the window over the frame, score every position, threshold, NMS.
    H, W = img.shape
    pos = np.array([(r, c) for r in range(0, H - OBJ + 1, stride)
                    for c in range(0, W - OBJ + 1, stride)])
    patches = np.stack([img[r:r + OBJ, c:c + OBJ] for r, c in pos])
    probs = clf.score(patches)
    keep = probs >= thr
    if not keep.any():
        return []
    return nms(pos[keep], probs[keep])


def evaluate(preds, gts, iou_thr=0.5):
    # Greedy match predictions (score-sorted) to ground truth by IoU >= thr.
    used, tp, fp = [False] * len(gts), 0, 0
    for box, _ in sorted(preds, key=lambda x: -x[1]):
        best, bi = iou_thr, -1
        for gi, g in enumerate(gts):
            v = iou(box, g)
            if not used[gi] and v >= best:
                best, bi = v, gi
        if bi >= 0:
            tp += 1
            used[bi] = True
        else:
            fp += 1
    return tp, fp, len(gts) - sum(used)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    # Train the per-window object/background classifier.
    Xtr, ytr = make_patches(rng, 400)
    clf = PatchClassifier().fit(Xtr, ytr)

    # Stream held-out frames and detect objects in each (the "real-time" loop).
    frames = [make_scene(rng) for _ in range(60)]
    grid = [(r, c) for r in range(0, 32 - OBJ + 1, 2)
            for c in range(0, 32 - OBJ + 1, 2)]

    d_tp = d_fp = d_fn = b_tp = b_fp = b_fn = 0
    t0 = time.time()
    for img, gts in frames:
        preds = detect(clf, img)
        tp, fp, fn = evaluate(preds, gts)
        d_tp, d_fp, d_fn = d_tp + tp, d_fp + fp, d_fn + fn
        # Baseline: same number of boxes as the detector, placed at random.
        ridx = rng.randint(0, len(grid), len(preds))
        rand_preds = [(grid[i], 1.0) for i in ridx]
        tp, fp, fn = evaluate(rand_preds, gts)
        b_tp, b_fp, b_fn = b_tp + tp, b_fp + fp, b_fn + fn
    fps = len(frames) / (time.time() - t0)

    dp, dr, df = prf(d_tp, d_fp, d_fn)
    bp, br, bf = prf(b_tp, b_fp, b_fn)
    print("Frames processed     :", len(frames))
    print("Throughput (fps)     :", round(fps, 1), "(real-time)")
    print("Ground-truth objects :", d_tp + d_fn)
    print("Detector  P/R/F1     : {:.2f} / {:.2f} / {:.2f}".format(dp, dr, df))
    print("Random    P/R/F1     : {:.2f} / {:.2f} / {:.2f}".format(bp, br, bf))
    print("F1 lift over random  : {:.1f}x".format(df / bf if bf else float("inf")))
    print("PASS" if df > 0.8 and df > 3 * max(bf, 1e-6) else "FAIL")
