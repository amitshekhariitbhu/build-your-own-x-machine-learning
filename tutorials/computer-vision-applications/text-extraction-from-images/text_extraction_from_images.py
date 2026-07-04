import numpy as np

# Text extraction = LAYOUT ANALYSIS (find lines / words / characters in a scene
# image via projection profiles) + character RECOGNITION (a from-scratch softmax
# classifier over the segmented glyphs).  Everything below is manual numpy.

# --- 5x7 bitmap font: 16 letters + 10 digits (each string is one scanline) ---
FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}
CHARS = list(FONT)                       # class index -> character
GH, GW = 7, 5                            # glyph height / width
CHAR_GAP, WORD_GAP = 1, 4                # empty cols between chars / words
SPACE_TH = 4                             # gap >= this many empty cols => a space


def render_glyph(ch):
    return np.array([[float(b) for b in row] for row in FONT[ch]])


def noisy_sample(ch, noise=0.18, flip_p=0.02):
    # A scanned glyph: variable ink darkness, speckle, and sensor noise.
    img = render_glyph(ch) * np.random.uniform(0.75, 1.0)
    flips = np.random.rand(GH, GW) < flip_p
    img[flips] = 1.0 - img[flips]
    img = img + noise * np.random.randn(GH, GW)
    return np.clip(img, 0.0, 1.0)


def resize_nn(patch, h=GH, w=GW):
    # Nearest-neighbour resample of a tight glyph crop to the fixed h x w grid.
    ph, pw = patch.shape
    if ph == 0 or pw == 0:
        return np.zeros((h, w))
    ri = np.clip(((np.arange(h) + 0.5) * ph / h).astype(int), 0, ph - 1)
    ci = np.clip(((np.arange(w) + 0.5) * pw / w).astype(int), 0, pw - 1)
    return patch[np.ix_(ri, ci)]


def normalize(gray):
    # Crop to the ink bounding box, then rescale to GH x GW -> flat feature vector.
    ink = gray > 0.5
    rows, cols = np.where(ink.any(1))[0], np.where(ink.any(0))[0]
    if rows.size and cols.size:
        gray = gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return resize_nn(gray).ravel()


def make_dataset(per_class):
    X, y = [], []
    for ci, ch in enumerate(CHARS):
        for _ in range(per_class):
            X.append(normalize(noisy_sample(ch)))
            y.append(ci)
    return np.array(X), np.array(y)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxClassifier:
    """From-scratch multinomial logistic regression trained by gradient descent."""

    def __init__(self, lr=0.5, n_iters=400, reg=1e-3):
        self.lr, self.n_iters, self.reg = lr, n_iters, reg

    def fit(self, X, y):
        n, d = X.shape
        k = int(y.max()) + 1
        Y = np.eye(k)[y]
        self.W, self.b = np.zeros((d, k)), np.zeros(k)
        for _ in range(self.n_iters):
            P = softmax(X @ self.W + self.b)
            dZ = (P - Y) / n
            self.W -= self.lr * (X.T @ dZ + self.reg * self.W)
            self.b -= self.lr * dZ.sum(axis=0)
        return self

    def predict(self, X):
        return softmax(X @ self.W + self.b).argmax(axis=1)


def runs(mask):
    # Maximal runs of True in a 1D mask -> list of (start, end) half-open spans.
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    parts = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return [(p[0], p[-1] + 1) for p in parts]


def build_document(lines):
    # Paint text lines onto one scene image at rediscoverable positions, add noise.
    laid = []                                            # (row0, col0, char)
    width = 0
    for li, text in enumerate(lines):
        y, x = 1 + li * (GH + 2), 1
        for ch in text:
            if ch == " ":
                x += WORD_GAP
                continue
            laid.append((y, x, ch))
            x += GW + CHAR_GAP
        width = max(width, x + 1)
    height = 1 + len(lines) * (GH + 2)
    scene = np.zeros((height, width))
    for y, x, ch in laid:
        scene[y:y + GH, x:x + GW] = render_glyph(ch) * np.random.uniform(0.8, 1.0)
    scene = np.clip(scene + 0.08 * np.random.randn(*scene.shape), 0.0, 1.0)
    return scene


def extract_text(model, scene):
    # 1) split rows into text lines, 2) split cols into chars, 3) recognise glyphs.
    ink = scene > 0.5
    out = []
    for r0, r1 in runs(ink.any(axis=1)):                 # line bands
        band = scene[r0:r1]
        col_ink = (band > 0.5).any(axis=0)
        segs = runs(col_ink)
        line, prev_end = "", None
        for c0, c1 in segs:
            if prev_end is not None and c0 - prev_end >= SPACE_TH:
                line += " "                              # wide gap => word break
            glyph = normalize(band[:, c0:c1])
            line += CHARS[model.predict(glyph[None])[0]]
            prev_end = c1
        out.append(line)
    return out


if __name__ == "__main__":
    np.random.seed(0)

    # --- Train the character recogniser on planted, noisy glyph templates ---
    Xtr, ytr = make_dataset(per_class=60)
    Xte, yte = make_dataset(per_class=30)                # held-out draw
    model = SoftmaxClassifier(lr=0.5, n_iters=400).fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = np.mean(pred == yte)
    rand_acc = 1.0 / len(CHARS)
    base_acc = np.mean(yte == np.bincount(ytr).argmax())  # majority guess

    # --- End-to-end text extraction from a synthetic multi-line document ---
    truth = ["HELLO WORLD", "MODEL 2024"]
    scene = build_document(truth)
    got = extract_text(model, scene)
    hits = tot = 0
    for t, g in zip(truth, got):
        m = max(len(t), len(g))
        hits += sum(a == b for a, b in zip(t.ljust(m), g.ljust(m)))
        tot += m
    char_acc = hits / tot

    print("Classes (chars):     %s" % "".join(CHARS))
    print("Train / test glyphs: %d / %d  (%dx%d px each)"
          % (len(ytr), len(yte), GH, GW))
    print("Held-out accuracy:   %.3f" % acc)
    print("Majority baseline:   %.3f" % base_acc)
    print("Random baseline:     %.3f" % rand_acc)
    print("Improvement vs rand: +%.1f pts" % (100 * (acc - rand_acc)))
    print("Scene image size:    %dx%d px" % scene.shape)
    print("Document truth:      %s" % " / ".join(truth))
    print("Extracted text:      %s" % " / ".join(got))
    print("Char-level accuracy: %.3f" % char_acc)
    print("Beats baseline:      %s" % bool(acc > max(base_acc, rand_acc) + 0.3
                                           and char_acc > 0.9))
