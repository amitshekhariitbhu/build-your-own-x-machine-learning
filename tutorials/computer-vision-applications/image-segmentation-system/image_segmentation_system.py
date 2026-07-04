import numpy as np

# Image segmentation from scratch: label EVERY pixel with its region class using
# hand-built local features + a from-scratch softmax classifier trained per-pixel.
# Three planted regions live in each tiny grayscale scene:
#   0 = background    : dark,   smooth
#   1 = object blob   : bright, smooth
#   2 = textured patch: mid mean brightness but STRIPED (high local variance)
# The stripes make region 2 span the same raw-brightness range as 0 and 1, so a
# plain intensity threshold cannot separate it -- the texture (local std) feature
# is what pulls it apart. We report held-out pixel accuracy and mean IoU and beat
# a majority-class baseline (and a naive brightness-threshold segmenter).

H = W = 28


def conv2d(img, kernel):
    # Same-size correlation with edge padding (used for box blur + Sobel).
    kh, kw = kernel.shape
    P = np.pad(img, (kh // 2, kw // 2), mode="edge")
    out = np.zeros_like(img)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * P[i:i + img.shape[0], j:j + img.shape[1]]
    return out


BOX3 = np.ones((3, 3)) / 9.0
BOX5 = np.ones((5, 5)) / 25.0
SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
SY = SX.T


def pixel_features(img):
    # Per-pixel feature stack (H, W, 4): smoothed value, local mean, texture, edges.
    sm = conv2d(img, BOX3)                                   # denoised brightness
    mean5 = conv2d(img, BOX5)                                # broad local mean
    var5 = conv2d(img * img, BOX5) - mean5 ** 2
    std5 = np.sqrt(np.clip(var5, 0, None))                  # texture / roughness
    grad = np.sqrt(conv2d(img, SX) ** 2 + conv2d(img, SY) ** 2)  # edge strength
    return np.stack([sm, mean5, std5, grad], axis=-1)


def make_image(rng):
    # Clean scene + its integer label map (the latent structure to recover).
    labels = np.zeros((H, W), int)                          # 0 = background
    img = np.full((H, W), 0.25)
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = rng.randint(7, 21, 2)                          # bright smooth blob
    r = rng.randint(4, 7)
    blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    labels[blob] = 1
    img[blob] = 0.75
    ph, pw = rng.randint(7, 12), rng.randint(7, 12)        # textured striped patch
    py, px = rng.randint(0, H - ph), rng.randint(0, W - pw)
    patch = np.zeros((H, W), bool)
    patch[py:py + ph, px:px + pw] = True
    stripes = 0.5 + 0.25 * np.sign(np.sin(xx * 1.6))       # mean .5, spans .25-.75
    labels[patch] = 2
    img[patch] = stripes[patch]
    img += rng.normal(0, 0.05, img.shape)                  # sensor noise
    return np.clip(img, 0, 1), labels


def make_dataset(n, seed):
    rng = np.random.RandomState(seed)
    imgs, labs = [], []
    for _ in range(n):
        im, lb = make_image(rng)
        imgs.append(im)
        labs.append(lb)
    return imgs, labs


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class PixelSegmenter:
    # Per-pixel multinomial logistic regression (softmax) via gradient descent.
    def __init__(self, n_classes=3, lr=0.5, epochs=400, reg=1e-3):
        self.K, self.lr, self.epochs, self.reg = n_classes, lr, epochs, reg

    def _feats(self, images):
        F = np.stack([pixel_features(im) for im in images])
        return F.reshape(-1, F.shape[-1])                  # (N*H*W, 4)

    def fit(self, images, labels):
        X = self._feats(images)
        y = np.concatenate([l.ravel() for l in labels])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8      # standardize (train only)
        X = (X - self.mu) / self.sd
        n, d = X.shape
        Y = np.eye(self.K)[y]
        counts = np.bincount(y, minlength=self.K)
        cw = n / (self.K * counts)                         # inverse-frequency weights
        w = cw[y][:, None]                                 # (n,1), mean weight = 1
        self.W = np.zeros((d, self.K))
        self.b = np.zeros(self.K)
        for _ in range(self.epochs):
            P = softmax(X @ self.W + self.b)
            G = w * (P - Y) / n                            # weighted softmax + CE
            self.W -= self.lr * (X.T @ G + self.reg * self.W)
            self.b -= self.lr * G.sum(0)
        return self

    def predict(self, images):
        X = (self._feats(images) - self.mu) / self.sd
        lab = (X @ self.W + self.b).argmax(1)
        return lab.reshape(len(images), H, W)


def mean_iou(pred, true, K):
    # Mean intersection-over-union across the K region classes.
    ious = []
    for c in range(K):
        p, t = pred == c, true == c
        u = (p | t).sum()
        ious.append((p & t).sum() / u if u else 1.0)
    return float(np.mean(ious))


if __name__ == "__main__":
    np.random.seed(0)
    K = 3
    names = ["background", "object", "texture"]

    tr_imgs, tr_labs = make_dataset(24, seed=0)
    te_imgs, te_labs = make_dataset(24, seed=1)

    seg = PixelSegmenter(n_classes=K).fit(tr_imgs, tr_labs)
    pred = seg.predict(te_imgs)
    true = np.stack(te_labs)

    acc = float(np.mean(pred == true))
    miou = mean_iou(pred, true, K)

    # Baseline 1: predict the majority (background) class for every pixel.
    maj = np.bincount(true.ravel(), minlength=K).argmax()
    base_acc = float(np.mean(true == maj))
    base_pred = np.full_like(true, maj)
    base_miou = mean_iou(base_pred, true, K)
    rand_acc = 1.0 / K

    # Baseline 2: naive brightness threshold (dark->bg, bright->object) -- no texture.
    thr_pred = np.stack([(im > 0.62).astype(int) for im in te_imgs])
    thr_acc = float(np.mean(thr_pred == true))
    thr_miou = mean_iou(thr_pred, true, K)                 # texture IoU = 0 (never guessed)

    # mIoU is the honest metric: background is ~81% of pixels, so plain accuracy is
    # inflated for baselines that ignore the minority object/texture regions.
    print("Regions            :", ", ".join(names))
    print("Train / test images:", len(tr_imgs), "/", len(te_imgs),
          "(%dx%d px each)" % (H, W))
    for c in range(K):
        rec = float(np.mean(pred[true == c] == c))
        print("  recall %-10s:" % names[c], round(rec, 3))
    print("Feature dim        : 4 (value, mean, texture, edge)")
    print("Random baseline    : acc %.4f" % rand_acc)
    print("Majority baseline  : acc %.4f  mIoU %.4f" % (base_acc, base_miou))
    print("Threshold baseline : acc %.4f  mIoU %.4f  (no texture feature)"
          % (thr_acc, thr_miou))
    print("Segmenter (ours)   : acc %.4f  mIoU %.4f" % (acc, miou))
    best_base_miou = max(base_miou, thr_miou)
    print("Beats baselines    :",
          bool(acc > base_acc and acc > thr_acc and miou > best_base_miou + 0.15))
