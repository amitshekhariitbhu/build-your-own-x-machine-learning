import numpy as np

# Visual Question Answering from scratch: each tiny grayscale image has ONE
# planted shape whose identity / position / size are the latent attributes.
# A question (bag-of-words) selects WHICH attribute to report, so answering
# requires FUSING vision + language -- neither modality alone can solve it.

N = 16                                              # image is N x N grayscale
SHAPES = ["square", "circle", "triangle"]
ANSWERS = ["square", "circle", "triangle",          # answer vocabulary (9 words)
           "top", "bottom", "left", "right", "big", "small"]
AIDX = {a: i for i, a in enumerate(ANSWERS)}

# Several phrasings per question type -> the language encoder must generalize.
TEMPLATES = {
    "shape": [["what", "shape"], ["which", "shape"], ["name", "the", "shape"]],
    "vpos":  [["top", "or", "bottom"], ["vertical", "side"], ["up", "or", "down"]],
    "hpos":  [["left", "or", "right"], ["horizontal", "side"], ["which", "side"]],
    "size":  [["big", "or", "small"], ["what", "size"], ["large", "or", "tiny"]],
}
QTYPES = list(TEMPLATES)                             # ["shape","vpos","hpos","size"]
VOCAB = sorted({w for ph in TEMPLATES.values() for t in ph for w in t})
WIDX = {w: i for i, w in enumerate(VOCAB)}


def render(shape, cr, cc, r):
    # Paint a filled shape centered at (cr,cc) with half-extent r.
    Y, X = np.ogrid[:N, :N]
    if shape == 0:                                  # square
        mask = (np.abs(Y - cr) <= r) & (np.abs(X - cc) <= r)
    elif shape == 1:                                # circle (disk)
        mask = (Y - cr) ** 2 + (X - cc) ** 2 <= r * r
    else:                                           # triangle, apex up
        hw = ((Y - (cr - r)) / (2 * r)) * r         # half-width grows top->bottom
        mask = (Y >= cr - r) & (Y <= cr + r) & (np.abs(X - cc) <= hw)
    return mask.astype(float)


def sample_scene(rng):
    # Latent attributes -> a rendered, noisy image.
    shape, vpos, hpos, size = (rng.integers(3), rng.integers(2),
                               rng.integers(2), rng.integers(2))
    r = 2 if size == 0 else 4                        # small vs big
    cr = (5 if vpos == 0 else 10) + rng.integers(-1, 2)   # top vs bottom
    cc = (5 if hpos == 0 else 10) + rng.integers(-1, 2)   # left vs right
    img = render(shape, cr, cc, r) + 0.05 * rng.standard_normal((N, N))
    return np.clip(img, 0.0, 1.0), (shape, vpos, hpos, size)


def make_question(attrs, rng):
    shape, vpos, hpos, size = attrs
    qt = rng.integers(4)
    phrase = TEMPLATES[QTYPES[qt]][rng.integers(3)]
    ans = [SHAPES[shape], ["top", "bottom"][vpos],
           ["left", "right"][hpos], ["small", "big"][size]][qt]
    q = np.zeros(len(VOCAB))                         # bag-of-words encoding
    for w in phrase:
        q[WIDX[w]] += 1.0
    return q, qt, AIDX[ans]


def resize(crop, out=5):
    # Nearest-neighbor resize to a canonical grid (size/position invariant).
    H, W = crop.shape
    ri = ((np.arange(out) + 0.5) * H / out).astype(int)
    ci = ((np.arange(out) + 0.5) * W / out).astype(int)
    return crop[np.ix_(ri, ci)]


def image_features(img):
    # Hand-built visual encoder: canonical silhouette + centroid + extent.
    ink = (img > 0.5).astype(float)
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return np.zeros(5 * 5 + 3)
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    sil = resize(ink[r0:r1 + 1, c0:c1 + 1]).ravel()   # 25: what the shape looks like
    pos = [ys.mean() / N, xs.mean() / N,              # where it is (vertical, horiz.)
           max(r1 - r0 + 1, c1 - c0 + 1) / N]          # how big it is
    return np.concatenate([sil, pos])


def make_dataset(n, rng):
    F, Q, y, qt = [], [], [], []
    for _ in range(n):
        img, attrs = sample_scene(rng)
        q, t, a = make_question(attrs, rng)
        F.append(image_features(img)); Q.append(q); y.append(a); qt.append(t)
    return np.array(F), np.array(Q), np.array(y), np.array(qt)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class VQAModel:
    """Bilinear vision+language fusion: the joint feature is the outer product
    of the (bias-augmented) image and question vectors, so the classifier can
    ROUTE -- 'if the question asks about shape, read the silhouette', etc. --
    then a from-scratch softmax head is trained by gradient descent."""

    def __init__(self, lr=0.5, n_iters=500, reg=1e-3):
        self.lr, self.n_iters, self.reg = lr, n_iters, reg

    def _join(self, F, Q):
        Ai = np.hstack([F, np.ones((len(F), 1))])     # append bias to each modality
        Aq = np.hstack([Q, np.ones((len(Q), 1))])
        return (Ai[:, :, None] * Aq[:, None, :]).reshape(len(F), -1)

    def fit(self, F, Q, y):
        J = self._join(F, Q)
        n, d = J.shape
        self.k = int(y.max()) + 1
        Yoh = np.eye(self.k)[y]
        self.W = np.zeros((d, self.k)); self.b = np.zeros(self.k)
        for _ in range(self.n_iters):
            P = softmax(J @ self.W + self.b)
            dZ = (P - Yoh) / n
            self.W -= self.lr * (J.T @ dZ + self.reg * self.W)
            self.b -= self.lr * dZ.sum(axis=0)
        return self

    def predict(self, F, Q):
        return (self._join(F, Q) @ self.W + self.b).argmax(axis=1)


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.default_rng(0)

    Ftr, Qtr, ytr, qttr = make_dataset(1500, rng)     # planted image+question pairs
    Fte, Qte, yte, qtte = make_dataset(500, rng)       # held-out draw

    model = VQAModel(lr=0.5, n_iters=500).fit(Ftr, Qtr, ytr)
    pred = model.predict(Fte, Qte)
    acc = np.mean(pred == yte)

    # Baselines --------------------------------------------------------------
    rand_acc = 1.0 / len(ANSWERS)                       # blind guessing
    majority = np.bincount(ytr).argmax()               # single most-common answer
    maj_acc = np.mean(yte == majority)
    q_only = {t: np.bincount(ytr[qttr == t]).argmax() for t in range(4)}  # ignore image
    qpred = np.array([q_only[t] for t in qtte])
    qonly_acc = np.mean(qpred == yte)

    print("Answer vocabulary:    %s" % ", ".join(ANSWERS))
    print("Train / test pairs:   %d / %d  (%dx%d px images)" % (len(ytr), len(yte), N, N))
    print("Random baseline:      %.3f" % rand_acc)
    print("Majority baseline:    %.3f" % maj_acc)
    print("Question-only base:   %.3f   (image ignored)" % qonly_acc)
    print("VQA (vision+lang):    %.3f" % acc)
    print("Improvement vs Q-only:+%.1f pts" % (100 * (acc - qonly_acc)))
    print("Beats baselines:      %s" % (acc > max(qonly_acc, maj_acc, rand_acc) + 0.2))

    # Qualitative demo: one scene, all four questions ------------------------
    img, attrs = sample_scene(rng)
    F1 = image_features(img)[None, :]
    print("\nScene attrs -> shape=%s vpos=%s hpos=%s size=%s"
          % (SHAPES[attrs[0]], ["top", "bottom"][attrs[1]],
             ["left", "right"][attrs[2]], ["small", "big"][attrs[3]]))
    for t, name in enumerate(QTYPES):
        phrase = TEMPLATES[name][0]
        q = np.zeros(len(VOCAB))
        for w in phrase:
            q[WIDX[w]] += 1.0
        a = model.predict(F1, q[None, :])[0]
        print("  Q: %-18s A: %s" % (" ".join(phrase) + "?", ANSWERS[a]))
