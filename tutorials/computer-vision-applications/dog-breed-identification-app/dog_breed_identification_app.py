import numpy as np

# Dog breed identification from scratch: hand-written 2D convolution (Sobel) ->
# HOG-like oriented-gradient histograms -> multinomial logistic (softmax).
# Each tiny grayscale "photo" plants breed-specific latent structure:
#   0 = Dalmatian       tall body, SPOTTED coat  (many dark spots -> high texture)
#   1 = Poodle          round body, CURLY coat   (wavy edges, medium texture)
#   2 = Dachshund       very LONG low body        (wide aspect ratio, smooth)
#   3 = German Shepherd tall body + POINTY ears   (strong diagonal edges up top)
# HOG reads the ear/body edge structure, a texture term reads the coat, and an
# aspect term reads the dachshund. Softmax trained by gradient descent classifies.

def make_dogs(n, H=28):
    X = np.zeros((n, H, H)); y = np.arange(n) % 4
    rx = [7.0, 8.0, 11.0, 6.5]; ry = [9.0, 8.0, 5.0, 9.0]   # body semi-axes per breed
    yy, xx = np.mgrid[0:H, 0:H].astype(float)
    for i in range(n):
        k = y[i]
        cy, cx = H / 2.0 + 2 + np.random.randint(-1, 2), H / 2.0 + np.random.randint(-1, 2)
        ax, ay = rx[k] * (1 + 0.06 * np.random.randn()), ry[k] * (1 + 0.06 * np.random.randn())
        ang = np.arctan2(yy - cy, xx - cx)
        wob = 0.20 * np.cos(6 * ang) if k == 1 else 0.0     # poodle: curly boundary
        ell = ((xx - cx) / (ax * (1 + wob))) ** 2 + ((yy - cy) / (ay * (1 + wob))) ** 2
        body = ell <= 1.0
        X[i][body] = 0.80
        hx = cx + np.random.randint(-1, 2)                  # head above the body
        head = ((xx - hx) ** 2 + (yy - (cy - ay - 2)) ** 2) <= 9.0
        X[i][head] = 0.80
        if k == 3:                                          # shepherd: two pointy ears
            for s in (-3, 3):
                tip = (np.abs(xx - (hx + s)) + 0.7 * np.abs(yy - (cy - ay - 4))) <= \
                      np.maximum(0, (cy - ay - 1) - yy) * 0.9
                X[i][(yy < cy - ay - 1) & tip] = 0.80
        if k == 0:                                          # dalmatian: dark coat spots
            for _ in range(9):
                sy, sx = np.random.randint(-6, 7), np.random.randint(-5, 6)
                spot = ((xx - cx - sx) ** 2 + (yy - cy - sy) ** 2) <= 1.6
                X[i][spot & body] = 0.15
        elif k == 1:                                        # poodle: fuzzy curl texture
            X[i][body] += 0.12 * np.random.randn(int(body.sum()))
    X += 0.05 * np.random.randn(n, H, H)                    # sensor noise
    return np.clip(X, 0, 1), y

def _conv3(img, ker):
    # Hand-written 3x3 convolution via 9 shifted, edge-padded slices (vectorized).
    p = np.pad(img, ((0, 0), (1, 1), (1, 1)), mode="edge")
    out = np.zeros_like(img)
    for a in range(3):
        for b in range(3):
            out += ker[a, b] * p[:, a:a + img.shape[1], b:b + img.shape[2]]
    return out

def dog_features(X, cell=7, nbins=6):
    # Sobel gradients -> per-cell unsigned-orientation histograms (HOG-lite) + globals.
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    gx, gy = _conv3(X, Kx), _conv3(X, Kx.T)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ori = (np.arctan2(gy, gx) % np.pi)                      # unsigned edge orientation
    b = np.minimum((ori / (np.pi / nbins)).astype(int), nbins - 1)
    N, H, W = X.shape; g = H // cell
    feats = []
    for bi in range(nbins):                                 # (n, g, g) energy per bin
        contrib = (mag * (b == bi)).reshape(N, g, cell, g, cell).sum(axis=(2, 4))
        feats.append(contrib.reshape(N, g * g))
    hog = np.concatenate(feats, axis=1)
    hog = hog / (np.linalg.norm(hog, axis=1, keepdims=True) + 1e-8)   # L2 block norm
    mask = X > 0.4                                          # foreground = the dog
    cols = mask.any(axis=1); rows = mask.any(axis=2)
    w = cols.sum(1); h = rows.sum(1)
    aspect = (w / (h + 1e-6))[:, None]                      # dachshund -> large
    inside = np.where(mask, X, np.nan)
    texture = np.nan_to_num(np.nanstd(inside.reshape(N, -1), axis=1))[:, None]  # spots
    top = mag[:, : H // 3, :].reshape(N, -1).sum(1)[:, None]  # ear edge energy up top
    glob = np.concatenate([aspect, 4 * texture, top / (mag.reshape(N, -1).sum(1)[:, None] + 1e-8)], 1)
    return np.concatenate([hog, glob], axis=1)

class SoftmaxClassifier:
    # Multinomial logistic regression trained by full-batch gradient descent.
    def __init__(self, n_classes, lr=0.5, epochs=500, l2=1e-3):
        self.K, self.lr, self.epochs, self.l2 = n_classes, lr, epochs, l2

    def fit(self, X, y):
        N, D = X.shape
        self.W = np.zeros((D, self.K)); self.b = np.zeros(self.K)
        Y = np.eye(self.K)[y]
        for _ in range(self.epochs):
            P = self._softmax(X @ self.W + self.b)
            G = (P - Y) / N
            self.W -= self.lr * (X.T @ G + self.l2 * self.W)
            self.b -= self.lr * G.sum(axis=0)
        return self

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return (X @ self.W + self.b).argmax(axis=1)

if __name__ == "__main__":
    np.random.seed(0)
    names = ["Dalmatian", "Poodle", "Dachshund", "GermanShepherd"]

    Xtr_img, ytr = make_dogs(400)
    Xte_img, yte = make_dogs(200)                           # held-out split

    Ftr, Fte = dog_features(Xtr_img), dog_features(Xte_img)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8                 # standardize on train
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd

    clf = SoftmaxClassifier(n_classes=4).fit(Ftr, ytr)
    pred = clf.predict(Fte)
    acc = np.mean(pred == yte)
    baseline = np.bincount(ytr, minlength=4).max() / len(ytr)   # majority guess

    print("Dog breeds        :", ", ".join(names))
    for k in range(4):
        rec = np.mean(pred[yte == k] == k)
        print(f"  {names[k]:15s} recall {rec:.3f}")
    print("Feature dim       :", Ftr.shape[1], "(HOG cells x 6 orientations + 3 globals)")
    print("Majority baseline :", round(float(baseline), 4))
    print("Softmax accuracy  :", round(float(acc), 4))
    print("Beats baseline    :", bool(acc > baseline + 0.3))
