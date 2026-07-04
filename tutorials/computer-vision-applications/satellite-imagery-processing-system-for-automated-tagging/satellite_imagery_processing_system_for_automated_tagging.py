import numpy as np

# Satellite imagery auto-tagging built from scratch (multi-label).
# Each 16x16 tile is split into 4 quadrants; a land-cover TAG may be
# planted in each quadrant, so a tile carries ANY subset of 4 tags:
#   water  (top-left)     -> dark, smooth region
#   forest (top-right)    -> mid-tone, high-texture region
#   urban  (bottom-left)  -> bright grid -> lots of edges
#   field  (bottom-right) -> bright, uniform region
# We hand-craft global features (Sobel edges, local-variance texture,
# dark/bright fractions) and fit a from-scratch multi-label logistic
# regression (one sigmoid head per tag) by batch gradient descent.

TAGS = ["water", "forest", "urban", "field"]

def conv2d(imgs, k):
    # Valid, stride-1 2D convolution of (N,H,W) tiles with a small kernel.
    n, H, W = imgs.shape
    kh, kw = k.shape
    oh, ow = H - kh + 1, W - kw + 1
    out = np.zeros((n, oh, ow))
    for i in range(kh):
        for j in range(kw):
            out += imgs[:, i:i + oh, j:j + ow] * k[i, j]
    return out

def features(X):
    # Turn tiles into 8 hand-crafted land-cover descriptors.
    mean = X.mean((1, 2))
    std = X.std((1, 2))
    frac_dark = (X < 0.3).mean((1, 2))            # water signal
    frac_bright = (X > 0.7).mean((1, 2))          # urban/field signal
    sx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], float)
    gx, gy = conv2d(X, sx), conv2d(X, sx.T)
    edge = np.sqrt(gx ** 2 + gy ** 2).mean((1, 2))  # urban grid -> high
    hgrad, vgrad = np.abs(gx).mean((1, 2)), np.abs(gy).mean((1, 2))
    ones = np.ones((3, 3))                          # local 3x3 variance
    m = conv2d(X, ones) / 9.0
    m2 = conv2d(X ** 2, ones) / 9.0
    tex = np.clip(m2 - m ** 2, 0, None).mean((1, 2))  # forest -> high
    return np.stack([mean, std, frac_dark, frac_bright,
                     edge, hgrad, vgrad, tex], axis=1)

class MultiLabelLogReg:
    # Independent sigmoid per tag, shared linear features, BCE + GD.
    def __init__(self, n_feat, n_tags):
        self.W = np.zeros((n_feat, n_tags))
        self.b = np.zeros(n_tags)
    def _sig(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    def predict_proba(self, X):
        return self._sig(X @ self.W + self.b)
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
    def fit(self, X, Y, epochs=400, lr=0.5):
        n = X.shape[0]
        for _ in range(epochs):
            g = self.predict_proba(X) - Y          # dLoss/dlogit
            self.W -= lr * (X.T @ g) / n
            self.b -= lr * g.mean(0)
        return self

def make_tiles(n):
    # Draw n satellite tiles; return tiles and their (n,4) binary tags.
    H = 16
    X = np.zeros((n, H, H))
    Y = (np.random.rand(n, 4) > 0.5).astype(int)
    for i in range(n):
        img = 0.5 * np.ones((H, H))                        # bare ground
        if Y[i, 0]:                                         # water
            img[0:8, 0:8] = 0.1 + 0.03 * np.random.randn(8, 8)
        if Y[i, 1]:                                         # forest (texture)
            img[0:8, 8:16] = 0.5 + 0.28 * np.random.randn(8, 8)
        if Y[i, 2]:                                         # urban (grid edges)
            u = 0.65 * np.ones((8, 8))
            u[::2, :] = 0.97
            u[:, ::2] = 0.97
            img[8:16, 0:8] = u
        if Y[i, 3]:                                         # field (bright flat)
            img[8:16, 8:16] = 0.87 + 0.02 * np.random.randn(8, 8)
        X[i] = img
    X = np.clip(X + 0.03 * np.random.randn(n, H, H), 0, 1)  # sensor noise
    return X, Y

def micro_f1(Yt, Yp):
    tp = ((Yp == 1) & (Yt == 1)).sum()
    fp = ((Yp == 1) & (Yt == 0)).sum()
    fn = ((Yp == 0) & (Yt == 1)).sum()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9)

if __name__ == "__main__":
    np.random.seed(0)

    Xtr_img, Ytr = make_tiles(400)
    Xte_img, Yte = make_tiles(200)

    # Standardize hand-crafted features with TRAIN statistics only.
    Ftr, Fte = features(Xtr_img), features(Xte_img)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd

    model = MultiLabelLogReg(Ftr.shape[1], len(TAGS)).fit(Ftr, Ytr)
    Yp = model.predict(Fte)

    # Baseline: predict every label by its TRAIN majority (present if >0.5).
    maj = (Ytr.mean(0) > 0.5).astype(int)
    Ybase = np.tile(maj, (len(Yte), 1))

    f1 = micro_f1(Yte, Yp)
    f1_base = micro_f1(Yte, Ybase)
    exact = np.mean((Yp == Yte).all(1))            # all 4 tags correct
    exact_base = np.mean((Ybase == Yte).all(1))
    per_tag = (Yp == Yte).mean(0)

    print("Tags                : " + ", ".join(TAGS))
    print("Per-tag accuracy    : " +
          ", ".join(f"{t}={a:.2f}" for t, a in zip(TAGS, per_tag)))
    print("Majority micro-F1   :", round(float(f1_base), 4))
    print("Model    micro-F1   :", round(float(f1), 4))
    print("Majority exact-match:", round(float(exact_base), 4))
    print("Model    exact-match:", round(float(exact), 4))
    print("Beats baseline      :", bool(f1 > f1_base + 0.3 and exact > 0.7))
