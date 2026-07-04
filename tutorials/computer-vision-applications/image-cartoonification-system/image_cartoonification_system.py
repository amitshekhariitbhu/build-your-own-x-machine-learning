import numpy as np

# Image cartoonification from scratch: turn a noisy "photo" into a flat,
# poster-like cartoon with bold black outlines. The classic recipe is
#   edge-preserving smoothing -> colour quantization -> edge overlay,
# each done by hand (median filter, k-means palette, Sobel edges).
# Synthetic scenes share a fixed 5-colour palette (sky, ground, sun, wall,
# roof), so a palette is LEARNED on training photos and applied to held-out
# ones. The latent structure is the clean flat regions; the photo hides them
# under a lighting gradient + sensor noise that cartoonifying should undo.

PALETTE = np.array([
    [0.55, 0.72, 0.95],   # sky  (kept < 1 to avoid clipping bias)
    [0.30, 0.60, 0.25],   # ground
    [0.95, 0.82, 0.20],   # sun
    [0.80, 0.35, 0.30],   # wall
    [0.45, 0.28, 0.20],   # roof
])
H = W = 24

def conv2d(gray, kernel):
    # Same-size correlation with edge padding (used for Sobel gradients).
    kh, kw = kernel.shape
    P = np.pad(gray, kh // 2, mode="edge")
    out = np.zeros_like(gray)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * P[i:i + gray.shape[0], j:j + gray.shape[1]]
    return out

def median_filter(img, k=3):
    # Per-channel k x k median: kills noise but preserves sharp region edges.
    pad = k // 2
    P = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    h, w, c = img.shape
    patches = np.empty((h, w, c, k * k))
    n = 0
    for i in range(k):
        for j in range(k):
            patches[..., n] = P[i:i + h, j:j + w, :]
            n += 1
    return np.median(patches, axis=-1)

def _kmeans_once(X, K, iters, rng):
    # k-means++ init (so rare bright colours each get a centroid) + Lloyd.
    C = [X[rng.randint(len(X))]]
    for _ in range(K - 1):
        d = np.min(((X[:, None, :] - np.array(C)[None]) ** 2).sum(2), axis=1)
        C.append(X[rng.choice(len(X), p=d / d.sum())])
    C = np.array(C, float)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        for k in range(K):
            if np.any(lab == k):
                C[k] = X[lab == k].mean(0)
    return C, d.min(1).sum()

def kmeans(X, K, iters=12, seed=0, restarts=6):
    # Keep the lowest-inertia run so all K well-separated colours are found.
    rng = np.random.RandomState(seed)
    best_C, best_inertia = None, np.inf
    for _ in range(restarts):
        C, inertia = _kmeans_once(X, K, iters, rng)
        if inertia < best_inertia:
            best_C, best_inertia = C, inertia
    return best_C

class Cartoonifier:
    # fit() learns a shared colour palette; transform() cartoonifies an image.
    def __init__(self, n_colors=5, edge_frac=0.13):
        self.n_colors = n_colors
        self.edge_frac = edge_frac
        self.SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
        self.SY = self.SX.T
    def fit(self, photos):
        pix = np.concatenate([median_filter(p, 3).reshape(-1, 3) for p in photos])
        self.palette = kmeans(pix, self.n_colors, seed=0)
        return self
    def quantize(self, img):
        flat = img.reshape(-1, 3)
        d = ((flat[:, None, :] - self.palette[None, :, :]) ** 2).sum(2)
        return self.palette[d.argmin(1)].reshape(img.shape)
    def detect_edges(self, gray):
        mag = np.sqrt(conv2d(gray, self.SX) ** 2 + conv2d(gray, self.SY) ** 2)
        return mag >= np.quantile(mag, 1.0 - self.edge_frac)
    def transform(self, photo):
        smooth = median_filter(photo, 3)               # edge-preserving denoise
        quant = self.quantize(smooth)                  # posterize to palette
        edges = self.detect_edges(smooth.mean(2))      # bold outlines
        cartoon = quant.copy()
        cartoon[edges] = 0.0
        return cartoon, quant, edges

def make_scene(rng):
    # Clean flat-region scene + its integer label map (the latent structure).
    labels = np.zeros((H, W), int)                     # 0 = sky
    horizon = rng.randint(12, 15)
    labels[horizon:, :] = 1                            # 1 = ground
    sy, sx, sr = rng.randint(4, 7), rng.randint(15, 20), rng.randint(4, 6)
    yy, xx = np.mgrid[0:H, 0:W]
    labels[(yy - sy) ** 2 + (xx - sx) ** 2 <= sr ** 2] = 2   # 2 = sun
    wx, ww = rng.randint(4, 8), rng.randint(6, 9)
    wtop = horizon - rng.randint(5, 8)
    labels[wtop:horizon, wx:wx + ww] = 3              # 3 = wall
    labels[wtop - 4:wtop, wx:wx + ww] = 4             # 4 = roof
    return PALETTE[labels], labels

def make_dataset(n, seed):
    # Corrupt each clean scene with a smooth lighting ramp + Gaussian noise.
    rng = np.random.RandomState(seed)
    photos, cleans, labs = [], [], []
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(n):
        clean, labels = make_scene(rng)
        ramp = 0.12 * (xx / W - 0.5) + 0.10 * (yy / H - 0.5)
        photo = clean + ramp[..., None] + rng.normal(0, 0.10, clean.shape)
        photos.append(np.clip(photo, 0, 1)); cleans.append(clean); labs.append(labels)
    return photos, cleans, labs

def true_boundaries(labels):
    # Pixels whose right/down neighbour has a different region label.
    b = np.zeros_like(labels, bool)
    b[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    b[:-1, :] |= labels[:-1, :] != labels[1:, :]
    return b

def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0

if __name__ == "__main__":
    np.random.seed(0)
    tr_photos, _, _ = make_dataset(16, seed=0)
    te_photos, te_cleans, te_labels = make_dataset(16, seed=1)

    model = Cartoonifier(n_colors=5, edge_frac=0.13).fit(tr_photos)

    base_mse = cart_mse = e_iou = r_iou = 0.0
    rng = np.random.RandomState(2)
    for photo, clean, labels in zip(te_photos, te_cleans, te_labels):
        cartoon, quant, edges = model.transform(photo)
        base_mse += np.mean((photo - clean) ** 2)      # baseline: raw photo
        cart_mse += np.mean((quant - clean) ** 2)      # recovered flat regions
        tb = true_boundaries(labels)
        e_iou += iou(edges, tb)
        r_iou += iou(rng.permutation(edges.ravel()).reshape(edges.shape), tb)  # same density
    n = len(te_photos)
    base_mse, cart_mse, e_iou, r_iou = base_mse / n, cart_mse / n, e_iou / n, r_iou / n

    ex = te_photos[0]
    n_photo = len(np.unique(np.round(ex.reshape(-1, 3), 2), axis=0))
    n_cart = len(np.unique(model.transform(ex)[1].reshape(-1, 3), axis=0))

    print("Palette colours learned :", model.palette.shape[0])
    print("Colours in photo (test) :", n_photo)
    print("Colours in cartoon      :", n_cart, "(posterized)")
    print("-- region recovery (MSE to clean cartoon, lower=better) --")
    print("Raw photo baseline MSE  :", round(float(base_mse), 5))
    print("Cartoon MSE             :", round(float(cart_mse), 5))
    print("MSE reduction (x)       :", round(float(base_mse / cart_mse), 2))
    print("Beats baseline          :", bool(cart_mse < base_mse * 0.5))
    print("-- edge outline detection (IoU to true boundaries) --")
    print("Random edges IoU        :", round(float(r_iou), 4))
    print("Sobel edges IoU         :", round(float(e_iou), 4))
    print("Beats random            :", bool(e_iou > r_iou + 0.2))
