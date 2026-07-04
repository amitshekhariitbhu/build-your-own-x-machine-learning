import numpy as np

# Human Pose Estimation System from scratch (keypoint regression MLP).
#
# Synthetic 24x24 grayscale images of a stick-figure "person": a vertical
# torso, a round head, two arms and two legs. The GLOBAL position of the body
# and every LIMB ANGLE are randomized, so the K=5 keypoints we must recover
# (head, left/right hand, left/right foot) move all over the frame -- a system
# has to READ the limb geometry, not memorize a fixed layout. Everything is
# hand-rolled:
#   1) a rasterizer that draws thick line-segments (distance-to-segment) and
#      plants exact ground-truth joint coordinates
#   2) a one-hidden-layer MLP (ReLU + sigmoid) trained by full-batch back-prop
#      that regresses all joint (x, y) coordinates directly from the pixels
# Reported on a held-out split: mean per-joint pixel error and PCK (percentage
# of correct keypoints within a tolerance) vs a mean-pose baseline.

S = 28                      # image side (pixels)
K = 5                       # keypoints: head, L-hand, R-hand, L-foot, R-foot
YY, XX = np.mgrid[0:S, 0:S]


def draw_segment(img, p0, p1, val, thick):
    # Paint a thick line segment onto img via point-to-segment distance.
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy + 1e-9
    t = np.clip(((XX - x0) * dx + (YY - y0) * dy) / L2, 0, 1)
    px, py = x0 + t * dx, y0 + t * dy
    dist = np.sqrt((XX - px) ** 2 + (YY - py) ** 2)
    img[dist <= thick] = val


def make_sample(rng):
    # Build one stick-figure image and its 5 ground-truth joint coordinates.
    img = np.full((S, S), 0.12) + rng.normal(0, 0.03, (S, S))    # dark textured bg
    cx = rng.uniform(9, 19)                                       # body shifts L/R
    ty = rng.uniform(-1, 1)                                       # body shifts up/down
    sy, hy = 9.0 + ty, 18.0 + ty                                 # shoulder / hip rows
    head = (cx, sy - 4)                                           # head center
    arm, leg = 8.0, 8.0
    aL = np.deg2rad(rng.uniform(120, 240))                        # left-arm angle
    aR = np.deg2rad(rng.uniform(-60, 60))                         # right-arm angle
    bL = np.deg2rad(rng.uniform(80, 150))                         # left-leg angle
    bR = np.deg2rad(rng.uniform(30, 100))                         # right-leg angle
    lh = (cx + arm * np.cos(aL), sy + arm * np.sin(aL))           # left hand
    rh = (cx + arm * np.cos(aR), sy + arm * np.sin(aR))           # right hand
    lf = (cx + leg * np.cos(bL), hy + leg * np.sin(bL))           # left foot
    rf = (cx + leg * np.cos(bR), hy + leg * np.sin(bR))           # right foot

    val = rng.uniform(0.85, 1.0)                                  # bright ink
    draw_segment(img, (cx, sy), (cx, hy), val, 1.3)              # torso
    draw_segment(img, (cx, sy), lh, val, 1.0)                    # arms
    draw_segment(img, (cx, sy), rh, val, 1.0)
    draw_segment(img, (cx, hy), lf, val, 1.0)                    # legs
    draw_segment(img, (cx, hy), rf, val, 1.0)
    img[(XX - head[0]) ** 2 + (YY - head[1]) ** 2 <= 4] = val    # round head
    img = np.clip(img, 0, 1)

    joints = np.array([head, lh, rh, lf, rf], float)             # (K, 2) = (x, y)
    return img.ravel(), joints.ravel()


class PoseNet:
    """One-hidden-layer MLP regressing normalized joint coords from pixels."""

    def __init__(self, n_in, n_hidden=96, n_out=2 * K, lr=0.35, epochs=1200, l2=1e-4, seed=0):
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


def joint_errors(pred, true):
    # pred/true: (N, 2K) normalized -> per-keypoint Euclidean pixel distances.
    p = pred.reshape(-1, K, 2) * S
    t = true.reshape(-1, K, 2) * S
    return np.sqrt(((p - t) ** 2).sum(-1))                       # (N, K)


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    n = 1000
    X = np.zeros((n, S * S)); Y = np.zeros((n, 2 * K))
    for i in range(n):
        X[i], Y[i] = make_sample(rng)
    Y /= S                                                        # normalize coords

    idx = rng.permutation(n); cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]

    net = PoseNet(S * S).fit(X[tr], Y[tr])
    pred = net.predict(X[te])

    err = joint_errors(pred, Y[te])                              # (Nte, K) pixels
    base_pose = Y[tr].mean(0)                                    # mean training pose
    base_err = joint_errors(np.tile(base_pose, (len(te), 1)), Y[te])

    tol = 3.0                                                    # PCK tolerance (px)
    pck, base_pck = (err < tol).mean(), (base_err < tol).mean()
    names = ["head", "L-hand", "R-hand", "L-foot", "R-foot"]

    print("Images: {}  ({}x{} px)   Keypoints: {}   Train/Test: {}/{}".format(
        n, S, S, K, len(tr), len(te)))
    print("-" * 56)
    print("Per-joint mean pixel error (held-out):")
    for k in range(K):
        print("  {:7s}  PoseNet {:5.2f} px   baseline {:5.2f} px".format(
            names[k], err[:, k].mean(), base_err[:, k].mean()))
    print("-" * 56)
    print("  Mean-pose baseline error : {:5.2f} px".format(base_err.mean()))
    print("  PoseNet mean error       : {:5.2f} px".format(err.mean()))
    print("  Baseline PCK@{:.0f}px        : {:.3f}".format(tol, base_pck))
    print("  PoseNet  PCK@{:.0f}px        : {:.3f}".format(tol, pck))
    print("-" * 56)
    ok = err.mean() < 0.7 * base_err.mean() and pck > base_pck + 0.25
    print("PASS" if ok else "FAIL")
