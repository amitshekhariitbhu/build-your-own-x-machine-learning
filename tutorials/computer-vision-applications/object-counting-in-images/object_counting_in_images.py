import numpy as np

# Object Counting in Images (from scratch).
#
# Each synthetic image is a single grayscale scene: a smoothly shaded, noisy
# background with a random number of blob-like OBJECTS (filled disks of varying
# radius and brightness) scattered in it, kept apart by a small gap so distinct
# objects stay distinct. Objects are always brighter than the local background,
# but their sizes vary a lot -- so the total bright AREA does not tell you HOW
# MANY there are; the count must come from spatial STRUCTURE. Unlike a video
# counter, there is no background stream to subtract; everything is inferred
# from the one image. The pipeline is hand-rolled:
#   1) OTSU thresholding -- pick the global threshold that maximizes between-
#      class variance of the pixel histogram (foreground vs background),
#   2) morphological OPENING (erode then dilate) to erase speckle noise,
#   3) connected-component labeling (flood fill) + area filter -> one blob per
#      object, so the object count is the number of surviving components.
# Held-out signal: count MAE / exact-image accuracy / total-count error, each
# printed next to a best-constant baseline it must clearly beat.

H, W = 48, 48                 # image size
GAP = 3                       # min empty pixels enforced between objects
NOISE = 0.05                  # background sensor noise std


def make_background(rng):
    # Smoothly shaded backdrop: a random low-frequency tilt + offset.
    gy = np.linspace(-1, 1, H)[:, None] * rng.uniform(-0.1, 0.1)
    gx = np.linspace(-1, 1, W)[None, :] * rng.uniform(-0.1, 0.1)
    return np.clip(0.35 + gy + gx, 0, 1)


def make_image(n, rng):
    # Drop up to n non-overlapping bright disks on a noisy background.
    img = np.clip(make_background(rng) + rng.normal(0, NOISE, (H, W)), 0, 1)
    placed = []                                            # (cy, cx, r)
    for _ in range(300):
        if len(placed) >= n:
            break
        r = rng.randint(2, 6)
        cy, cx = rng.randint(r + 1, H - r - 1), rng.randint(r + 1, W - r - 1)
        if all((cy - py) ** 2 + (cx - px) ** 2 > (r + pr + GAP) ** 2
               for (py, px, pr) in placed):
            placed.append((cy, cx, r))
    yy, xx = np.mgrid[0:H, 0:W]
    for (cy, cx, r) in placed:
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        bright = img[disk] + rng.uniform(0.35, 0.55)       # varied brightness
        img[disk] = np.clip(bright + rng.normal(0, 0.03, bright.shape), 0, 1)
    return img, len(placed)


def otsu(img):
    # Global threshold maximizing between-class variance of the 32-bin histogram.
    hist, edges = np.histogram(img, bins=32, range=(0, 1))
    p = hist / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)                                       # background weight
    w1 = 1 - w0                                             # foreground weight
    m0 = np.cumsum(p * centers) / np.clip(w0, 1e-9, None)
    total = (p * centers).sum()
    m1 = (total - np.cumsum(p * centers)) / np.clip(w1, 1e-9, None)
    between = w0 * w1 * (m0 - m1) ** 2                      # variance per split
    return centers[np.argmax(between)]


def erode(m):
    # 4-neighborhood binary erosion (survive only if all neighbors set).
    out = m.copy()
    out[1:, :] &= m[:-1, :]; out[:-1, :] &= m[1:, :]
    out[:, 1:] &= m[:, :-1]; out[:, :-1] &= m[:, 1:]
    return out


def dilate(m):
    # 4-neighborhood binary dilation (set if any neighbor is set).
    out = m.copy()
    out[1:, :] |= m[:-1, :]; out[:-1, :] |= m[1:, :]
    out[:, 1:] |= m[:, :-1]; out[:, :-1] |= m[:, 1:]
    return out


def label(mask):
    # Connected-component labeling via iterative 8-connected flood fill.
    lab = np.zeros(mask.shape, int)
    cur = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and lab[i, j] == 0:
                cur += 1
                stack = [(i, j)]; lab[i, j] = cur
                while stack:
                    y, x = stack.pop()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < H and 0 <= nx < W and
                                    mask[ny, nx] and lab[ny, nx] == 0):
                                lab[ny, nx] = cur; stack.append((ny, nx))
    return lab


class ObjectCounter:
    """Counts objects via Otsu threshold + morphological opening + components."""

    def __init__(self, min_area=6):
        self.min_area = min_area

    def mask(self, img):
        # Otsu alone collapses when there are no bright objects (it just splits
        # the background noise). Floor it at a contrast margin above the dominant
        # (background) level so empty scenes yield an empty mask.
        thr = max(otsu(img), np.median(img) + 0.15)        # per-image threshold
        return dilate(erode(img > thr))                    # opening kills speckle

    def predict(self, img):
        counts = np.bincount(label(self.mask(img)).ravel())
        return int(np.sum(counts[1:] >= self.min_area))    # drop bg + tiny blobs


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    images, truth = [], []
    for _ in range(160):
        img, c = make_image(rng.randint(0, 8), rng)        # 0..7 objects/image
        images.append(img); truth.append(c)
    truth = np.array(truth)

    cut = 110                                              # train (baseline) / test
    counter = ObjectCounter()
    pred = np.array([counter.predict(im) for im in images[cut:]])
    true = truth[cut:]

    mae = np.mean(np.abs(pred - true))
    exact = np.mean(pred == true)
    const = int(round(truth[:cut].mean()))                 # best constant guess
    base_mae = np.mean(np.abs(const - true))
    base_exact = np.mean(const == true)
    tot_err = abs(pred.sum() - true.sum())                 # total-count error

    print("Images: {} train (baseline) / {} test".format(cut, len(true)))
    print("Objects per image: 0..7    test total (ground truth): {}".format(true.sum()))
    print("-" * 58)
    print("[Per-image count]                        MAE    exact-acc")
    print("  Baseline (always predict {})             {:.3f}    {:.3f}".format(
        const, base_mae, base_exact))
    print("  Otsu + opening + connected components    {:.3f}    {:.3f}".format(
        mae, exact))
    print("-" * 58)
    print("[Total count]  true={}  predicted={}  error={}".format(
        true.sum(), pred.sum(), tot_err))
    print("-" * 58)
    ok = mae < base_mae - 1.0 and exact > 0.75 and tot_err <= 6
    print("PASS" if ok else "FAIL")
