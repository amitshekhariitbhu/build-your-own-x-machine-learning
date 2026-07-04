import numpy as np

# Neural style transfer from scratch (Gatys et al.), NumPy only.
# We stylize a CONTENT image so it keeps its layout but takes on the TEXTURE
# of a STYLE image. Content = low-frequency layout (a blur response); style =
# feature CORRELATIONS captured by a Gram matrix over a fixed bank of oriented
# 3x3 filters (Sobel-x/y + two diagonals) that plays the role of a conv layer.
# We minimise  alpha*content_loss + beta*style_loss  by gradient descent on the
# pixels, backpropping through the (linear) filter bank via its exact adjoint.
# Latent structure: styles are stripe textures with a planted ORIENTATION
# (vertical / horizontal / diagonal). A stylized image should adopt its style's
# orientation -> a nearest-Gram classifier recovers which style was applied,
# while the raw content image (no orientation) only scores at chance.

H = W = 24
# Oriented filter bank -> the "conv layer" whose Gram matrix encodes texture.
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)  # vertical edges
SOBEL_Y = SOBEL_X.T                                              # horizontal edges
DIAG_A = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], float)   # "\" edges
DIAG_B = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], float)   # "/" edges
FILTERS = [SOBEL_X, SOBEL_Y, DIAG_A, DIAG_B]
BLUR = np.ones((5, 5)) / 25.0                                    # content = layout


def corr_valid(x, k):
    # Valid 2-D cross-correlation (kernel loop; kernels are tiny 3x3/5x5).
    kh, kw = k.shape
    gh, gw = x.shape[0] - kh + 1, x.shape[1] - kw + 1
    y = np.zeros((gh, gw))
    for a in range(kh):
        for b in range(kw):
            y += k[a, b] * x[a:a + gh, b:b + gw]
    return y


def corr_valid_T(g, k, out_shape):
    # Exact adjoint of corr_valid: scatters gradient g back onto the input grid.
    kh, kw = k.shape
    gh, gw = g.shape
    gx = np.zeros(out_shape)
    for a in range(kh):
        for b in range(kw):
            gx[a:a + gh, b:b + gw] += k[a, b] * g
    return gx


def features(x):
    # Stack filter responses into a (C, M) matrix (C channels, M pixels).
    maps = [corr_valid(x, f) for f in FILTERS]
    shape = maps[0].shape
    return np.stack([m.ravel() for m in maps]), shape


def gram(x):
    # Normalised feature-correlation (Gram) matrix = the style descriptor.
    F, _ = features(x)
    return F @ F.T / F.shape[1]


class StyleTransfer:
    # transfer() runs pixel-space gradient descent on the Gatys objective.
    def __init__(self, alpha=5.0, beta=1.0, lr=0.02, iters=250):
        self.alpha, self.beta, self.lr, self.iters = alpha, beta, lr, iters

    def transfer(self, content, style):
        Gs = gram(style)                       # style target (Gram matrix)
        Bc = corr_valid(content, BLUR)         # content target (layout)
        x = content.copy()                     # init from content
        for _ in range(self.iters):
            # -- content gradient: keep the blurred layout close to content --
            Bx = corr_valid(x, BLUR)
            cdiff = Bx - Bc
            g_content = 2.0 * corr_valid_T(cdiff, BLUR, x.shape)
            # -- style gradient: match feature correlations (Gram) --
            F, shape = features(x)
            M = F.shape[1]
            Gdiff = F @ F.T / M - Gs
            dF = (4.0 / M) * (Gdiff @ F)        # dE/dF, derived analytically
            g_style = np.zeros_like(x)
            for k, f in enumerate(FILTERS):
                g_style += corr_valid_T(dF[k].reshape(shape), f, x.shape)
            x -= self.lr * (self.alpha * g_content + self.beta * g_style)
            np.clip(x, 0.0, 1.0, out=x)         # projected step (valid image)
        return x


def make_content(rng):
    # A low-frequency shape (square / cross / disk) -> the layout to preserve.
    img = np.full((H, W), 0.25)
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = rng.randint(8, 16), rng.randint(8, 16)
    kind = rng.randint(3)
    if kind == 0:                                            # filled square
        img[cy - 5:cy + 5, cx - 5:cx + 5] = 0.9
    elif kind == 1:                                          # cross
        img[cy - 6:cy + 6, cx - 2:cx + 2] = 0.9
        img[cy - 2:cy + 2, cx - 6:cx + 6] = 0.9
    else:                                                    # disk
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= 30] = 0.9
    return np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1)


def make_style(orientation, rng):
    # Stripe texture with a planted orientation (the latent style signal).
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    period = rng.uniform(2.6, 3.4)
    phase = rng.uniform(0, 2 * np.pi)
    coord = [xx, yy, xx + yy][orientation]                   # 0=vert 1=horiz 2=diag
    img = 0.5 + 0.4 * np.sin(2 * np.pi * coord / period + phase)
    return np.clip(img + rng.normal(0, 0.03, img.shape), 0, 1)


if __name__ == "__main__":
    np.random.seed(0)
    K = 3                                                    # style classes
    names = ["vertical", "horizontal", "diagonal"]

    # Learn a style prototype (average Gram) per orientation from train samples.
    rng = np.random.RandomState(0)
    protos = [np.mean([gram(make_style(o, rng)) for _ in range(4)], axis=0)
              for o in range(K)]

    def classify(img):                                       # nearest-Gram label
        G = gram(img)
        return int(np.argmin([np.sum((G - P) ** 2) for P in protos]))

    # Held-out test: fresh contents + fresh style samples of every class.
    rng = np.random.RandomState(1)
    model = StyleTransfer()
    hits = base_hits = 0
    style_out = style_in = 0.0
    keep = swap = 0.0
    pairs = 0
    for _ in range(5):
        content = make_content(rng)
        for o in range(K):
            style = make_style(o, rng)
            out = model.transfer(content, style)
            hits += (classify(out) == o)                     # stylized -> its style?
            base_hits += (classify(content) == o)            # raw content (chance)
            style_out += np.sum((gram(out) - gram(style)) ** 2)   # style match
            style_in += np.sum((gram(content) - gram(style)) ** 2)
            keep += np.corrcoef(corr_valid(out, BLUR).ravel(),    # layout kept
                                corr_valid(content, BLUR).ravel())[0, 1]
            swap += np.corrcoef(corr_valid(out, BLUR).ravel(),    # not just style
                                corr_valid(style, BLUR).ravel())[0, 1]
            pairs += 1

    acc = hits / pairs
    base = base_hits / pairs
    print("Style transfers run       :", pairs, "(%dx%d, %d iters each)" % (H, W, model.iters))
    print("-- did the output adopt its style's orientation? --")
    print("Random / chance level     :", round(1.0 / K, 3))
    print("Raw content baseline acc  :", round(base, 3))
    print("Stylized-output accuracy  :", round(acc, 3))
    print("Beats baseline            :", bool(acc > base + 0.4 and acc >= 0.9))
    print("-- style match (Gram dist to style, lower=better) --")
    print("Content->style baseline   :", round(style_in / pairs, 3))
    print("Output ->style            :", round(style_out / pairs, 3))
    print("Style-loss reduction (x)  :", round(style_in / style_out, 2))
    print("-- content preserved (blur-layout correlation) --")
    print("Output vs CONTENT layout  :", round(keep / pairs, 3), "(kept high)")
    print("Output vs STYLE layout    :", round(swap / pairs, 3), "(stays low)")
