import numpy as np

# Image colorization from scratch: predict COLOR from a grayscale image.
# We synthesize tiny RGB "scenes" of 4 materials, each with a fixed color and a
# luminance texture: sky (smooth blue), grass (vertical-stripe green),
# sand (smooth yellow), water (horizontal-stripe teal). Color is stored as
# chroma = RGB - Y, so texture/noise live only in the known luminance Y.
# A tiny per-pixel MLP maps a 3x3 grayscale patch -> (chroma_R, chroma_G, chroma_B).
# Reconstruct RGB = Y + predicted_chroma; texture/patch context disambiguates
# materials that overlap in single-pixel brightness.

MATERIALS = np.array([
    [0.50, 0.70, 0.95],   # 0 sky   - bright blue
    [0.25, 0.55, 0.20],   # 1 grass - green,  vertical stripes
    [0.60, 0.50, 0.25],   # 2 sand  - yellow, smooth
    [0.20, 0.42, 0.52],   # 3 water - teal,   horizontal stripes
])
YW = np.array([0.299, 0.587, 0.114])            # luminance weights

def make_scene(H=14, W=14):
    # Horizontal bands, each a random material; return RGB and per-pixel label.
    nb = np.random.randint(2, 5)
    cuts = np.sort(np.random.choice(np.arange(1, H), nb - 1, replace=False))
    bounds = np.concatenate([[0], cuts, [H]])
    labels = np.random.permutation(4)[:nb]
    lab = np.empty((H, W), dtype=int)
    for b in range(nb):
        lab[bounds[b]:bounds[b + 1]] = labels[b]
    rgb = MATERIALS[lab].astype(float)          # base color per pixel
    cols, rows = np.arange(W), np.arange(H)[:, None]
    tex = np.zeros((H, W))                       # luminance-only texture
    tex += (lab == 1) * 0.12 * ((cols % 2) * 2 - 1)      # grass: vertical
    tex += (lab == 3) * 0.12 * ((rows % 2) * 2 - 1)      # water: horizontal
    rgb += tex[:, :, None]                       # add equally -> chroma unchanged
    rgb += 0.02 * np.random.randn(H, W, 3)       # faint color noise
    return np.clip(rgb, 0, 1), lab

def patches(Y):
    # 3x3 grayscale patch per pixel (reflect-padded) -> (H*W, 9).
    H, W = Y.shape
    p = np.pad(Y, 1, mode="reflect")
    f = [p[i:i + H, j:j + W] for i in range(3) for j in range(3)]
    return np.stack(f, axis=-1).reshape(H * W, 9)

def build(n, H=14, W=14):
    # Stack n scenes into patch features X, targets chroma, plus Y and labels.
    X, C, Yl, L = [], [], [], []
    for _ in range(n):
        rgb, lab = make_scene(H, W)
        Y = rgb @ YW                             # grayscale = luminance
        X.append(patches(Y))
        C.append((rgb - Y[:, :, None]).reshape(-1, 3))
        Yl.append(Y.reshape(-1))
        L.append(lab.reshape(-1))
    return (np.vstack(X), np.vstack(C), np.concatenate(Yl), np.concatenate(L))

class ColorMLP:
    # 9 -> hidden (ReLU) -> 3, MSE-regressed chroma with mini-batch SGD.
    def __init__(self, nin=9, nh=32, nout=3):
        self.W1 = np.random.randn(nin, nh) * np.sqrt(2.0 / nin)
        self.b1 = np.zeros(nh)
        self.W2 = np.random.randn(nh, nout) * np.sqrt(2.0 / nh)
        self.b2 = np.zeros(nout)
    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(self.z1, 0)
        return self.a1 @ self.W2 + self.b2
    def predict(self, X):
        return self.forward(X)
    def fit(self, X, C, epochs=40, lr=0.05, batch=256):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - self.mu) / self.sd
        n = len(X)
        for _ in range(epochs):
            perm = np.random.permutation(n)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                xb, cb = Xn[idx], C[idx]
                out = self.forward(xb)
                d = 2 * (out - cb) / len(idx)          # MSE grad
                dW2 = self.a1.T @ d; db2 = d.sum(0)
                da1 = (d @ self.W2.T) * (self.z1 > 0)
                dW1 = xb.T @ da1; db1 = da1.sum(0)
                self.W1 -= lr * dW1; self.b1 -= lr * db1
                self.W2 -= lr * dW2; self.b2 -= lr * db2
        return self
    def transform(self, X):
        return self.forward((X - self.mu) / self.sd)   # colorize grayscale patches

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

if __name__ == "__main__":
    np.random.seed(0)

    Xtr, Ctr, Ytr, Ltr = build(60)
    Xte, Cte, Yte, Lte = build(30)
    rgb_true = Yte[:, None] + Cte                       # ground-truth color

    model = ColorMLP().fit(Xtr, Ctr, epochs=40, lr=0.05)
    rgb_pred = np.clip(Yte[:, None] + model.transform(Xte), 0, 1)

    # Baselines: grayscale (no color) and mean-chroma (uniform tint).
    rgb_gray = np.repeat(Yte[:, None], 3, axis=1)
    rgb_mean = Yte[:, None] + Ctr.mean(0)
    # Per-pixel linear baseline: least-squares  [1, Y] -> chroma (no patch context).
    A = np.column_stack([np.ones_like(Ytr), Ytr])
    coef = np.linalg.lstsq(A, Ctr, rcond=None)[0]
    rgb_lin = Yte[:, None] + np.column_stack([np.ones_like(Yte), Yte]) @ coef

    # Region-color accuracy: nearest material by predicted chroma vs true label.
    cent = MATERIALS - (MATERIALS @ YW)[:, None]
    pred_lab = np.argmin(((model.transform(Xte)[:, None] - cent) ** 2).sum(2), axis=1)
    acc = float(np.mean(pred_lab == Lte))
    majority = float(np.bincount(Lte).max() / len(Lte))

    print("Materials            : sky, grass, sand, water")
    print("Grayscale RMSE       :", round(rmse(rgb_gray, rgb_true), 4))
    print("Mean-chroma RMSE     :", round(rmse(rgb_mean, rgb_true), 4))
    print("Per-pixel lin RMSE   :", round(rmse(rgb_lin, rgb_true), 4))
    print("ColorMLP RMSE        :", round(rmse(rgb_pred, rgb_true), 4))
    print("Beats mean baseline  :", bool(rmse(rgb_pred, rgb_true) < 0.5 * rmse(rgb_mean, rgb_true)))
    print("Beats per-pixel lin  :", bool(rmse(rgb_pred, rgb_true) < rmse(rgb_lin, rgb_true)))
    print("Majority region acc  :", round(majority, 4))
    print("Colorized region acc :", round(acc, 4))
    print("Beats majority       :", bool(acc > majority + 0.3))
