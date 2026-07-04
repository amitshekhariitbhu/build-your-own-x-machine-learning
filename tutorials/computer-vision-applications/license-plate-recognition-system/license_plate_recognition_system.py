import numpy as np

# --- 5x7 dot-matrix font: 10 digits + 10 letters (alphanumeric plate chars) ---
FONT = {
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
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}
DIGITS = list("0123456789")
LETTERS = list("ABCDEHLPTZ")
CHARS = DIGITS + LETTERS                  # class index -> character
CANVAS_H, CANVAS_W = 9, 7                 # per-char cell (1px pad -> allows jitter)
K = 6                                     # chars per plate: 3 letters + 3 digits
PH, PW = CANVAS_H, K * CANVAS_W           # plate box size (9 x 42)
SCENE_H, SCENE_W = 18, 54                 # full scene: room to localize the plate


def render_cell(ch, dy=0, dx=0):
    # Paint a clean 0/1 glyph onto the padded cell, translated by (dy, dx).
    cell = np.zeros((CANVAS_H, CANVAS_W))
    top, left = 1 + dy, 1 + dx
    for r, row in enumerate(FONT[ch]):
        for c, bit in enumerate(row):
            if bit == "1":
                cell[top + r, left + c] = 1.0
    return cell


def add_noise(img, noise=0.06, flip_p=0.01):
    # Shared sensor model so training cells and scene pixels match distributions.
    out = img.copy()
    flips = np.random.rand(*out.shape) < flip_p        # salt-and-pepper speckle
    out[flips] = 1.0 - out[flips]
    out += noise * np.random.randn(*out.shape)         # additive gray noise
    return np.clip(out, 0.0, 1.0)


def char_sample(ch):
    # One noisy training glyph with a random 1px shift (translation robustness).
    return add_noise(render_cell(ch, np.random.randint(-1, 2), np.random.randint(-1, 2)))


def make_char_dataset(per_class):
    X, y = [], []
    for ci, ch in enumerate(CHARS):
        for _ in range(per_class):
            X.append(char_sample(ch).ravel())
            y.append(ci)
    return np.array(X), np.array(y)


def random_plate():
    return "".join(np.random.choice(LETTERS, 3)) + "".join(np.random.choice(DIGITS, 3))


def make_scene(plate):
    # Textured mid-gray background + a distractor bar; bright plate glyphs planted in.
    bg = np.full((SCENE_H, SCENE_W), 0.30)
    bg += np.linspace(-0.05, 0.05, SCENE_W)[None, :]                   # lighting gradient
    br, bc = np.random.randint(0, SCENE_H - 4), np.random.randint(0, SCENE_W - 8)
    bg[br:br + 4, bc:bc + 8] = 0.60                                    # distractor block
    content = np.hstack([render_cell(ch) for ch in plate])            # 9 x 42, fixed pitch
    r0 = np.random.randint(0, SCENE_H - PH + 1)
    c0 = np.random.randint(0, SCENE_W - PW + 1)
    bg[r0:r0 + PH, c0:c0 + PW] = content                              # plant the plate
    return add_noise(bg), (r0, c0, r0 + PH, c0 + PW)


# ------------------------- Stage 1: plate localization -------------------------
def smooth(img):
    # 3x3 Gaussian blur (vectorized) to suppress speckle before edge finding.
    p = np.pad(img, 1, mode="edge")
    return (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:] +
            2 * p[1:-1, :-2] + 4 * p[1:-1, 1:-1] + 2 * p[1:-1, 2:] +
            p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) / 16.0


def edge_map(img):
    # Gradient magnitude (|dI/dx| + |dI/dy|); the char band has the densest edges.
    gx = np.zeros_like(img); gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    return np.abs(gx) + np.abs(gy)


def detect_plate(scene):
    # Slide a PHxPW window; pick the max total edge energy via an integral image.
    E = edge_map(smooth(scene))
    S = np.zeros((E.shape[0] + 1, E.shape[1] + 1))
    S[1:, 1:] = E.cumsum(0).cumsum(1)
    rr = np.arange(SCENE_H - PH + 1); cc = np.arange(SCENE_W - PW + 1)
    sums = (S[np.ix_(rr + PH, cc + PW)] - S[np.ix_(rr, cc + PW)]
            - S[np.ix_(rr + PH, cc)] + S[np.ix_(rr, cc)])
    r, c = np.unravel_index(sums.argmax(), sums.shape)
    return (r, c, r + PH, c + PW)


def iou(a, b):
    r0, c0 = max(a[0], b[0]), max(a[1], b[1])
    r1, c1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, r1 - r0) * max(0, c1 - c0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) +
                    (b[2] - b[0]) * (b[3] - b[1]) - inter)


# ------------------------- Stage 2: character reader ---------------------------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxReader:
    """From-scratch multinomial logistic regression over flat cell pixels,
    trained by full-batch gradient descent on the cross-entropy loss."""

    def __init__(self, lr=0.5, n_iters=1500, reg=1e-3):
        self.lr, self.n_iters, self.reg = lr, n_iters, reg

    def fit(self, X, y):
        n, d = X.shape
        self.k = int(y.max()) + 1
        Y = np.eye(self.k)[y]
        self.W = np.zeros((d, self.k)); self.b = np.zeros(self.k)
        for _ in range(self.n_iters):
            P = softmax(X @ self.W + self.b)
            dZ = (P - Y) / n                                # softmax + CE gradient
            self.W -= self.lr * (X.T @ dZ + self.reg * self.W)
            self.b -= self.lr * dZ.sum(axis=0)
        return self

    def predict(self, X):
        return softmax(X @ self.W + self.b).argmax(axis=1)


def read_plate(model, scene, box):
    # Crop the detected box, slice into K fixed-pitch cells, classify each.
    crop = scene[box[0]:box[2], box[1]:box[3]]
    cells = [crop[:, i * CANVAS_W:(i + 1) * CANVAS_W].ravel() for i in range(K)]
    return "".join(CHARS[i] for i in model.predict(np.array(cells)))


if __name__ == "__main__":
    np.random.seed(0)

    # Planted structure: 20 glyph templates + jitter/speckle/sensor noise.
    Xtr, ytr = make_char_dataset(per_class=150)
    model = SoftmaxReader().fit(Xtr, ytr)

    # Held-out scenes: each hides a plate at a random location on textured clutter.
    N = 300
    ious, rand_ious, char_hits, char_n, exact = [], [], 0, 0, 0
    for _ in range(N):
        truth = random_plate()
        scene, gt = make_scene(truth)
        box = detect_plate(scene)                            # Stage 1: localize
        ious.append(iou(box, gt))
        rb_r = np.random.randint(0, SCENE_H - PH + 1)
        rb_c = np.random.randint(0, SCENE_W - PW + 1)
        rand_ious.append(iou((rb_r, rb_c, rb_r + PH, rb_c + PW), gt))
        read = read_plate(model, scene, box)                 # Stage 2: read (end-to-end)
        char_hits += sum(a == b for a, b in zip(read, truth))
        char_n += K
        exact += (read == truth)

    det_iou = np.mean(ious)
    rand_iou = np.mean(rand_ious)
    char_acc = char_hits / char_n
    rand_char = 1.0 / len(CHARS)

    # One worked end-to-end example.
    np.random.seed(7)
    truth = random_plate(); scene, gt = make_scene(truth)
    box = detect_plate(scene); read = read_plate(model, scene, box)

    print("Classes (chars):        %s" % "".join(CHARS))
    print("Train cells / scenes:   %d / %d  (scene %dx%d, plate %dx%d)"
          % (len(ytr), N, SCENE_H, SCENE_W, PH, PW))
    print("--- Stage 1: plate localization ---")
    print("Detection IoU:          %.3f" % det_iou)
    print("Random-box IoU:         %.3f" % rand_iou)
    print("--- Stage 2: character recognition (on detected crop) ---")
    print("Char accuracy:          %.3f" % char_acc)
    print("Random baseline:        %.3f" % rand_char)
    print("Whole-plate exact:      %.3f" % (exact / N))
    print("Example truth / read:   %s / %s" % (truth, read))
    print("Detected box vs truth:  %s vs %s" % (tuple(int(v) for v in box), gt))
    print("Beats baseline:         %s"
          % bool(det_iou > rand_iou + 0.3 and char_acc > rand_char + 0.3))
