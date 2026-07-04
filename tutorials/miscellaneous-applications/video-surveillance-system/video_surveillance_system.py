import numpy as np

# Develop a Video Surveillance System from scratch.
# A fixed camera watches a static scene under gradual lighting drift + sensor
# noise. The system must (1) flag every frame in which a moving INTRUDER appears
# and (2) locate it. We plant a bright blob that enters the scene during known
# intervals and moves across it. Detection uses adaptive background subtraction:
# a running background model updated FAST on background pixels (to track lighting
# drift) but only SLOWLY on foreground pixels (rate beta << alpha) so a moving
# intruder is never absorbed, yet its lingering "ghost" still heals once it
# leaves. We threshold the difference and apply a 3x3 speckle filter. Ground-
# truth motion labels + centroids let us score it against a majority-class
# baseline (detection) and a center-guess baseline (localization); a naive STATIC
# background is shown to be far worse.

H, W, T = 48, 64, 130          # frame height, width, number of frames
WARMUP = 15                    # first frames (intruder-free) that seed the model


def make_video(seed=0):
    # Build a synthetic surveillance clip with planted intruder events.
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    # Static scene: smooth room structure + fixed fine-grain texture.
    base = 120 + 30 * np.sin(2 * np.pi * xx / W) + 20 * np.cos(2 * np.pi * yy / H)
    base = base + rng.randn(H, W) * 4.0
    noise_sigma = 6.0

    # Intruder present during two intervals; label = 1 while present.
    present = np.zeros(T, dtype=int)
    events = [(20, 40), (78, 100)]
    for a, b in events:
        present[a:b] = 1
    true_cxy = np.full((T, 2), np.nan)          # (col, row) of intruder center

    frames = np.empty((T, H, W))
    for t in range(T):
        drift = 22.0 * np.sin(2 * np.pi * t / 40.0)      # global lighting change
        frame = base + drift + rng.randn(H, W) * noise_sigma
        if present[t]:
            a, b = next((a, b) for a, b in events if a <= t < b)
            p = (t - a) / (b - a - 1)                     # 0->1 progress in event
            cx, cy = 6 + p * (W - 12), 12 + p * (H - 24)  # left->right, top->down
            true_cxy[t] = (cx, cy)
            frame = frame + 65.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 4.0 ** 2))
        frames[t] = frame
    return frames, present, true_cxy, noise_sigma


def denoise(mask):
    # Speckle filter: keep a foreground pixel only if >=4 of its 8 neighbours are
    # also foreground. Removes isolated noise pixels, keeps the solid blob.
    p = np.pad(mask.astype(np.int16), 1)
    nb = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
          p[1:-1, :-2] + p[1:-1, 2:] +
          p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])
    return mask & (nb >= 4)


class VideoSurveillance:
    """Adaptive background-subtraction motion detector + intruder localizer."""

    def __init__(self, thresh=24.0, min_area=25, alpha=0.3, beta=0.2, adaptive=True):
        self.thresh, self.min_area = thresh, min_area
        self.alpha, self.beta, self.adaptive = alpha, beta, adaptive

    def fit(self, warmup_frames):
        # Background model = mean of intruder-free warm-up frames.
        self.B0 = warmup_frames.mean(axis=0)
        return self

    def process(self, frames):
        # Returns per-frame motion flag and estimated (col, row) intruder center.
        motion = np.zeros(len(frames), dtype=int)
        cxy = np.full((len(frames), 2), np.nan)
        B = self.B0.copy()
        for t, f in enumerate(frames):
            mask = denoise(np.abs(f - B) > self.thresh)      # foreground pixels
            if mask.sum() >= self.min_area:
                motion[t] = 1
                ys, xs = np.nonzero(mask)
                cxy[t] = (xs.mean(), ys.mean())              # blob centroid
            if self.adaptive:
                # Background pixels adapt FAST (alpha) to follow lighting drift;
                # foreground pixels adapt SLOWLY (beta) so a moving intruder is
                # not absorbed, yet its ghost heals shortly after it leaves.
                upd = (1 - self.alpha) * B + self.alpha * f
                B = np.where(mask, (1 - self.beta) * B + self.beta * f, upd)
        return motion, cxy


def prf1(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


if __name__ == "__main__":
    np.random.seed(0)

    frames, present, true_cxy, noise_sigma = make_video(seed=0)

    # Seed the model on the intruder-free warm-up, then scan the whole clip.
    guard = VideoSurveillance().fit(frames[:WARMUP])
    motion, cxy = guard.process(frames)

    acc = np.mean(motion == present)
    prec, rec, f1 = prf1(present, motion)

    # Baseline 1: always predict the majority class (no motion information used).
    majority = int(present.mean() >= 0.5)
    base_pred = np.full_like(present, majority)
    base_acc = np.mean(base_pred == present)
    _, _, base_f1 = prf1(present, base_pred)

    # Baseline 2: naive STATIC background (no adaptation to lighting drift).
    static = VideoSurveillance(adaptive=False).fit(frames[:WARMUP])
    s_motion, _ = static.process(frames)
    s_acc = np.mean(s_motion == present)
    _, _, s_f1 = prf1(present, s_motion)

    # Localization: on correctly-detected intruder frames, how close is the
    # estimated centre to the truth vs. a "guess the frame centre" baseline.
    hit = (present == 1) & (motion == 1)
    err = np.linalg.norm(cxy[hit] - true_cxy[hit], axis=1)
    base_err = np.linalg.norm(np.array([W / 2, H / 2]) - true_cxy[hit], axis=1)

    print("Frames: %d (%dx%d)   Warm-up: %d   Intruder frames: %d"
          % (T, H, W, WARMUP, present.sum()))
    print("-" * 60)
    print("Detector (adaptive)  acc: %.4f  P: %.4f  R: %.4f  F1: %.4f"
          % (acc, prec, rec, f1))
    print("Baseline (majority)  acc: %.4f  P: ----    R: ----    F1: %.4f"
          % (base_acc, base_f1))
    print("Baseline (static bg) acc: %.4f  ......................  F1: %.4f"
          % (s_acc, s_f1))
    print("-" * 60)
    print("Localization error (px)  detector: %.2f   center-guess: %.2f"
          % (err.mean(), base_err.mean()))
    print("-" * 60)
    beats = (acc > base_acc and f1 > base_f1 and f1 > s_f1
             and err.mean() < base_err.mean())
    print("Surveillance system beats every baseline:", beats)
