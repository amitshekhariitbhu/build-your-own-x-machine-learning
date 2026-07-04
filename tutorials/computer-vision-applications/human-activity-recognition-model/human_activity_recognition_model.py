import numpy as np

# Human activity recognition from scratch (video-based).
# Each sample is a tiny grayscale CLIP of T frames showing a stick-figure blob.
# The MOTION pattern is the activity:
#   0 standing -> body still
#   1 walking  -> body slides horizontally
#   2 jumping  -> body bobs vertically
#   3 waving   -> body still, a small arm bar oscillates near the top
# We turn each clip into motion features (centroid trajectory + frame-difference
# energy by region) and train a plain softmax classifier by gradient descent.

H, W, T = 16, 16, 8          # frame height, width, frames per clip

def draw_body(cx, cy):
    # Solid torso+legs rectangle centred at (cx, cy).
    img = np.zeros((H, W))
    r0, r1 = int(cy) - 3, int(cy) + 4          # 7 rows tall
    c0, c1 = int(cx) - 2, int(cx) + 2          # 4 cols wide
    img[max(0, r0):min(H, r1), max(0, c0):min(W, c1)] = 1.0
    return img

def make_clip(activity):
    # Render one T-frame clip for the given activity, with pose/sensor jitter.
    clip = np.zeros((T, H, W))
    cx0 = 8 + np.random.randint(-1, 2)         # base position jitter
    cy0 = 9 + np.random.randint(-1, 2)
    phase = np.random.uniform(0, 2 * np.pi)    # random motion phase
    amp = np.random.uniform(2.5, 3.5)          # motion amplitude jitter
    for t in range(T):
        cx, cy = cx0, cy0
        if activity == 1:                      # walking: horizontal drift
            cx = cx0 - 3 + 6.0 * t / (T - 1)
        elif activity == 2:                    # jumping: vertical bob
            cy = cy0 - amp * abs(np.sin(np.pi * t / (T - 1)))
        frame = draw_body(cx, cy)
        if activity == 3:                      # waving: oscillating arm bar
            ax = int(cx0 + amp * np.sin(2 * np.pi * t / (T - 1) + phase))
            frame[2:5, max(0, ax):min(W, ax + 2)] = 1.0
        frame += 0.15 * np.random.randn(H, W)  # sensor noise
        clip[t] = np.clip(frame, 0, 1)
    return clip

def make_data(n):
    # Balanced set of clips labelled by activity 0..3.
    X = np.zeros((n, T, H, W))
    y = np.arange(n) % 4
    for i in range(n):
        X[i] = make_clip(y[i])
    return X, y

def features(X):
    # Motion descriptors that make the four activities linearly separable.
    n = X.shape[0]
    rows = np.arange(H)[None, :]
    cols = np.arange(W)[None, :]
    F = np.zeros((n, 7))
    for i in range(n):
        clip = np.maximum(X[i], 0.0)
        tot = clip.sum(axis=(1, 2)) + 1e-8            # per-frame mass
        cy = (clip.sum(axis=2) * rows).sum(1) / tot   # vertical centroid path
        cx = (clip.sum(axis=1) * cols).sum(1) / tot   # horizontal centroid path
        diff = np.abs(np.diff(clip, axis=0))          # frame-to-frame motion
        upper = diff[:, :H // 3, :].sum()             # motion in top third
        lower = diff[:, 2 * H // 3:, :].sum()         # motion in bottom third
        total = diff.sum() + 1e-8
        F[i] = [cx.max() - cx.min(), cy.max() - cy.min(),
                cx.std(), cy.std(), total, upper / total, lower / total]
    return F

class SoftmaxClassifier:
    # Multinomial logistic regression via full-batch gradient descent.
    def __init__(self, n_classes, lr=0.3, l2=1e-3, iters=500):
        self.n_classes, self.lr, self.l2, self.iters = n_classes, lr, l2, iters

    def _softmax(self, Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xn = (X - self.mu) / self.sd
        n, d = Xn.shape
        self.W = np.zeros((d, self.n_classes))
        self.b = np.zeros(self.n_classes)
        Y = np.eye(self.n_classes)[y]                 # one-hot targets
        for _ in range(self.iters):
            P = self._softmax(Xn @ self.W + self.b)
            G = (P - Y) / n
            self.W -= self.lr * (Xn.T @ G + self.l2 * self.W)
            self.b -= self.lr * G.sum(0)
        return self

    def predict(self, X):
        Xn = (X - self.mu) / self.sd
        return self._softmax(Xn @ self.W + self.b).argmax(axis=1)


if __name__ == "__main__":
    np.random.seed(0)

    Xtr, ytr = make_data(600)
    Xte, yte = make_data(240)                          # held-out clips

    clf = SoftmaxClassifier(n_classes=4).fit(features(Xtr), ytr)
    pred = clf.predict(features(Xte))
    acc = np.mean(pred == yte)

    # Baseline: always guess the most common activity in the training set.
    baseline = np.bincount(ytr).max() / len(ytr)

    names = ["standing", "walking", "jumping", "waving"]
    print("Activities         :", ", ".join(names))
    print("Train / test clips :", len(ytr), "/", len(yte))
    print("Clip shape (T,H,W) :", (T, H, W))
    print("Majority baseline  :", round(float(baseline), 4))
    print("Recognition acc.   :", round(float(acc), 4))
    print("Beats baseline     :", bool(acc > baseline + 0.3))
    print("Per-activity recall:")
    for c, nm in enumerate(names):
        rec = np.mean(pred[yte == c] == c)
        print("   {:9s}: {:.3f}".format(nm, float(rec)))
