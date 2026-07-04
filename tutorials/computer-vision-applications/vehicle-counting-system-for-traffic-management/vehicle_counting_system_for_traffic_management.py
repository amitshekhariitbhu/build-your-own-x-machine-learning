import numpy as np

# Vehicle Counting System for Traffic Management (from scratch).
#
# Each synthetic frame is a top-down grayscale patch of a multi-lane road: a
# smoothly shaded asphalt background + sensor noise, with a random number of
# vehicles (small rectangles) dropped into the lanes. Vehicle POLARITY is
# randomized -- some are brighter than the road, some darker -- so a plain
# brightness sum cannot report how many there are; the count has to come from
# spatial STRUCTURE. The whole pipeline is hand-rolled:
#   1) background subtraction -- estimate the empty road as the per-pixel MEDIAN
#      over many frames (vehicles move, so each pixel is road most of the time),
#   2) threshold |frame - background| into a foreground mask,
#   3) morphological OPENING (erode then dilate) to kill speckle noise,
#   4) connected-component labeling (flood fill) + area filter -> one blob per
#      vehicle, so the vehicle count is the number of surviving components.
# Held-out signal: count MAE / exact-frame accuracy / total-throughput error,
# each printed next to a best-constant baseline it must clearly beat.

H, W = 36, 72                 # frame size (a stretch of road)
LANES = (7, 17, 27)           # vehicle row centers (three lanes)
GAP = 3                       # min empty pixels enforced between vehicles
NOISE = 0.04                  # asphalt sensor noise std


def make_road(rng):
    # Empty road: mild top-to-bottom brightness gradient (lighting).
    grad = np.linspace(-0.08, 0.08, H)[:, None] * np.ones((1, W))
    return np.clip(0.55 + grad, 0, 1)


def _too_close(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    return not (ax >= bx + bw + GAP or bx >= ax + aw + GAP or
                ay >= by + bh + GAP or by >= ay + ah + GAP)


def make_frame(road, n, rng):
    # Drop up to n non-touching vehicles onto the road; return frame + true count.
    img = np.clip(road + rng.normal(0, NOISE, road.shape), 0, 1)
    boxes = []
    for _ in range(200):
        if len(boxes) >= n:
            break
        vh, vw = rng.randint(4, 6), rng.randint(6, 9)
        cy = LANES[rng.randint(len(LANES))] + rng.randint(-1, 2)
        y, x = cy - vh // 2, rng.randint(2, W - vw - 2)
        box = (x, y, vw, vh)
        if 0 <= y and y + vh < H and all(not _too_close(box, b) for b in boxes):
            boxes.append(box)
    for (x, y, vw, vh) in boxes:
        sign = 1 if rng.rand() < 0.5 else -1                 # randomized polarity
        val = road[y:y + vh, x:x + vw] + sign * rng.uniform(0.25, 0.40)
        img[y:y + vh, x:x + vw] = np.clip(val + rng.normal(0, 0.03, (vh, vw)), 0, 1)
    return img, len(boxes)


def erode(m):
    # 4-neighborhood binary erosion (a pixel survives only if all neighbors set).
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
    # Connected-component labeling by iterative 8-connected flood fill.
    lab = np.zeros(mask.shape, int)
    cur = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
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


class VehicleCounter:
    """Counts vehicles via median background subtraction + connected components."""

    def __init__(self, thresh=0.15, min_area=8):
        self.thresh, self.min_area = thresh, min_area

    def fit(self, frames):
        # Learn the empty road as the per-pixel median across many frames.
        self.bg = np.median(np.stack(frames), axis=0)
        return self

    def mask(self, frame):
        m = np.abs(frame - self.bg) > self.thresh     # foreground vs road
        return dilate(erode(m))                        # opening removes speckle

    def predict(self, frame):
        counts = np.bincount(label(self.mask(frame)).ravel())
        return int(np.sum(counts[1:] >= self.min_area))   # drop bg + tiny blobs


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    road = make_road(rng)
    frames, truth = [], []
    for _ in range(180):
        img, c = make_frame(road, rng.randint(0, 7), rng)   # 0..6 vehicles/frame
        frames.append(img); truth.append(c)
    truth = np.array(truth)

    cut = 120
    counter = VehicleCounter().fit(frames[:cut])            # bg from train frames
    pred = np.array([counter.predict(f) for f in frames[cut:]])
    true = truth[cut:]

    mae = np.mean(np.abs(pred - true))
    exact = np.mean(pred == true)
    const = int(round(truth[:cut].mean()))                  # best constant guess
    base_mae = np.mean(np.abs(const - true))
    base_exact = np.mean(const == true)
    tot_err = abs(pred.sum() - true.sum())                  # throughput over stream

    print("Frames: {} train (background model) / {} test".format(cut, len(true)))
    print("Vehicles per frame: 0..6   test total (ground truth): {}".format(true.sum()))
    print("-" * 56)
    print("[Per-frame count]                      MAE    exact-acc")
    print("  Baseline (always predict {})           {:.3f}    {:.3f}".format(
        const, base_mae, base_exact))
    print("  Median-bg + connected components      {:.3f}    {:.3f}".format(
        mae, exact))
    print("-" * 56)
    print("[Total throughput]  true={}  predicted={}  error={}".format(
        true.sum(), pred.sum(), tot_err))
    print("-" * 56)
    ok = mae < base_mae - 1.0 and exact > 0.8 and tot_err <= 3
    print("PASS" if ok else "FAIL")
