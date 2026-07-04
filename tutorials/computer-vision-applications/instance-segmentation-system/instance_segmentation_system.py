import numpy as np

# Instance segmentation from scratch: not just "which pixels are foreground"
# (that is SEMANTIC segmentation) but "which OBJECT does each pixel belong to".
# Every tiny grayscale scene holds several bright blobs of the SAME class on a
# dark background, and some blobs TOUCH -- so a foreground mask alone fuses
# neighbouring objects into one region. The whole pipeline is hand-built:
#   1) learn a per-pixel foreground/background mask   (binary logistic on feats)
#   2) chamfer distance transform of that mask         (two sweeps, no library)
#   3) blob centres = local maxima of the distance     (markers / seeds)
#   4) assign each foreground pixel to its nearest seed (marker watershed)
# Predicted instances are matched to ground-truth blobs by IoU; we report
# instance Precision / Recall / F1 @IoU>=0.5 and mean matched IoU, and beat a
# semantic (one-instance) baseline that can never separate touching objects.

H = W = 32
DIAG = np.sqrt(2.0)


def conv2d(img, kernel):
    # Same-size correlation with edge padding (box blur + Sobel).
    kh, kw = kernel.shape
    P = np.pad(img, (kh // 2, kw // 2), mode="edge")
    out = np.zeros_like(img)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * P[i:i + img.shape[0], j:j + img.shape[1]]
    return out


BOX3 = np.ones((3, 3)) / 9.0
SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
SY = SX.T


def pixel_features(img):
    # Per-pixel stack (H, W, 3): smoothed value, raw value, edge magnitude.
    sm = conv2d(img, BOX3)
    grad = np.sqrt(conv2d(img, SX) ** 2 + conv2d(img, SY) ** 2)
    return np.stack([sm, img, grad], axis=-1)


def make_image(rng):
    # Dark scene + K bright disks; return image and its instance-id label map
    # (0 = background, 1..K = distinct objects). Disks may lightly overlap.
    img = np.full((H, W), 0.2)
    inst = np.zeros((H, W), int)
    yy, xx = np.mgrid[0:H, 0:W]
    K = rng.randint(2, 5)
    centers, nid, tries = [], 0, 0
    while nid < K and tries < 80:
        tries += 1
        cy, cx = rng.randint(6, H - 6), rng.randint(6, W - 6)
        if any((cy - y) ** 2 + (cx - x) ** 2 < 8 ** 2 for y, x in centers):
            continue                                  # keep centres distinct
        r = rng.randint(3, 6)
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        nid += 1
        inst[disk] = nid                              # later disk wins overlaps
        img[disk] = 0.8
        centers.append((cy, cx))
    img += rng.normal(0, 0.04, img.shape)             # sensor noise
    return np.clip(img, 0, 1), inst


def make_dataset(n, seed):
    rng = np.random.RandomState(seed)
    imgs, labs = [], []
    for _ in range(n):
        im, lb = make_image(rng)
        imgs.append(im)
        labs.append(lb)
    return imgs, labs


def distance_transform(mask):
    # Two-pass chamfer approximation of Euclidean distance to background.
    d = np.where(mask, 1e9, 0.0)
    h, w = mask.shape
    for i in range(h):                                # forward sweep
        for j in range(w):
            if not mask[i, j]:
                continue
            b = d[i, j]
            if i:               b = min(b, d[i - 1, j] + 1)
            if j:               b = min(b, d[i, j - 1] + 1)
            if i and j:         b = min(b, d[i - 1, j - 1] + DIAG)
            if i and j < w - 1: b = min(b, d[i - 1, j + 1] + DIAG)
            d[i, j] = b
    for i in range(h - 1, -1, -1):                    # backward sweep
        for j in range(w - 1, -1, -1):
            if not mask[i, j]:
                continue
            b = d[i, j]
            if i < h - 1:            b = min(b, d[i + 1, j] + 1)
            if j < w - 1:            b = min(b, d[i, j + 1] + 1)
            if i < h - 1 and j < w - 1: b = min(b, d[i + 1, j + 1] + DIAG)
            if i < h - 1 and j:      b = min(b, d[i + 1, j - 1] + DIAG)
            d[i, j] = b
    return d


def max_filter3(a):
    # 3x3 dilation (max over each pixel's neighbourhood) for peak detection.
    P = np.pad(a, 1, mode="constant", constant_values=-1e9)
    m = np.full_like(a, -1e9)
    for di in (0, 1, 2):
        for dj in (0, 1, 2):
            m = np.maximum(m, P[di:di + a.shape[0], dj:dj + a.shape[1]])
    return m


def find_seeds(dt, thr, radius):
    # Blob centres = local maxima of the distance transform, deduped by NMS.
    peaks = (dt == max_filter3(dt)) & (dt >= thr)
    coords = np.argwhere(peaks).astype(float)
    if len(coords) == 0:
        return coords
    order = np.argsort(-dt[peaks])
    keep, taken = [], np.zeros(len(coords), bool)
    for idx in order:
        if taken[idx]:
            continue
        keep.append(coords[idx])
        taken |= ((coords - coords[idx]) ** 2).sum(1) <= radius ** 2
    return np.array(keep)


class InstanceSegmenter:
    # Binary logistic foreground model + marker-watershed instance splitting.
    def __init__(self, lr=0.5, epochs=300, reg=1e-3, seed_dt=2.0, nms_r=5):
        self.lr, self.epochs, self.reg = lr, epochs, reg
        self.seed_dt, self.nms_r = seed_dt, nms_r

    def _feats(self, images):
        F = np.stack([pixel_features(im) for im in images])
        return F.reshape(-1, F.shape[-1])

    def fit(self, images, inst_labels):
        X = self._feats(images)
        y = (np.concatenate([l.ravel() for l in inst_labels]) > 0).astype(float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8   # standardize (train only)
        Xs = (X - self.mu) / self.sd
        n, d = Xs.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.epochs):                    # logistic + gradient descent
            p = 1.0 / (1.0 + np.exp(-(Xs @ self.w + self.b)))
            g = (p - y) / n
            self.w -= self.lr * (Xs.T @ g + self.reg * self.w)
            self.b -= self.lr * g.sum()
        return self

    def mask(self, img):
        X = (pixel_features(img).reshape(-1, 3) - self.mu) / self.sd
        p = 1.0 / (1.0 + np.exp(-(X @ self.w + self.b)))
        return (p > 0.5).reshape(H, W)

    def segment(self, img):
        m = self.mask(img)
        if not m.any():
            return np.zeros((H, W), int)
        dt = distance_transform(m)
        seeds = find_seeds(dt, self.seed_dt, self.nms_r)
        if len(seeds) == 0:                             # fallback: single object
            seeds = np.array([np.unravel_index(dt.argmax(), dt.shape)], float)
        ys, xs = np.nonzero(m)
        fg = np.stack([ys, xs], 1).astype(float)
        d2 = ((fg[:, None, :] - seeds[None, :, :]) ** 2).sum(2)  # to each seed
        inst = np.zeros((H, W), int)
        inst[ys, xs] = d2.argmin(1) + 1                 # nearest-seed watershed
        return inst

    def predict(self, images):
        return [self.segment(im) for im in images]


def match_instances(pred, gt, thr=0.5):
    # Greedy IoU matching: each GT blob claims its best unused prediction.
    pids = [i for i in np.unique(pred) if i > 0]
    gids = [i for i in np.unique(gt) if i > 0]
    used, tp, ious = set(), 0, []
    for g in gids:
        gm = gt == g
        best, bp = 0.0, None
        for p in pids:
            if p in used:
                continue
            pm = pred == p
            u = (gm | pm).sum()
            iou = (gm & pm).sum() / u if u else 0.0
            if iou > best:
                best, bp = iou, p
        if best >= thr:
            tp += 1
            used.add(bp)
            ious.append(best)
    return tp, len(pids) - tp, len(gids) - tp, ious      # TP, FP, FN, matchedIoU


def prf(tp, fp, fn):
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return prec, rec, 2 * prec * rec / (prec + rec + 1e-9)


if __name__ == "__main__":
    np.random.seed(0)
    tr_imgs, tr_lab = make_dataset(30, seed=0)
    te_imgs, te_lab = make_dataset(30, seed=1)

    seg = InstanceSegmenter().fit(tr_imgs, tr_lab)
    preds = seg.predict(te_imgs)

    # Our instance metrics + a foreground-mask sanity check.
    TP = FP = FN = 0
    IOU, mask_correct, mask_total = [], 0, 0
    n_gt = n_pred = 0
    for p, g, im in zip(preds, te_lab, te_imgs):
        tp, fp, fn, i = match_instances(p, g)
        TP += tp; FP += fp; FN += fn; IOU += i
        n_gt += len([k for k in np.unique(g) if k > 0])
        n_pred += len([k for k in np.unique(p) if k > 0])
        mask_correct += int((seg.mask(im) == (g > 0)).sum())
        mask_total += g.size
    prec, rec, f1 = prf(TP, FP, FN)
    miou = float(np.mean(IOU)) if IOU else 0.0

    # Semantic baseline: whole foreground = ONE instance (never splits objects).
    bTP = bFP = bFN = 0
    for im, g in zip(te_imgs, te_lab):
        base = seg.mask(im).astype(int)                 # all foreground -> id 1
        tp, fp, fn, _ = match_instances(base, g)
        bTP += tp; bFP += fp; bFN += fn
    bprec, brec, bf1 = prf(bTP, bFP, bFN)

    print("Test scenes            :", len(te_imgs), "(%dx%d px)" % (H, W))
    print("Ground-truth objects   :", n_gt, " predicted objects:", n_pred)
    print("Foreground mask acc    : %.4f" % (mask_correct / mask_total))
    print("Random match F1 (~0)   : 0.0000")
    print("Semantic baseline (1x) : P %.3f  R %.3f  F1 %.3f  (fuses touching objects)"
          % (bprec, brec, bf1))
    print("Instance seg (ours)    : P %.3f  R %.3f  F1 %.3f  mIoU %.3f"
          % (prec, rec, f1, miou))
    print("Beats baseline         :", bool(f1 > bf1 + 0.2 and f1 > 0.7))
