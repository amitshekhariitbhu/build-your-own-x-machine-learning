import numpy as np

# --- 5x7 bitmap font for digits 0-9 (each row is one scanline) ---
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
}
CHARS = list(FONT)                       # class index -> character
GLYPH_H, GLYPH_W = 7, 5
CANVAS_H, CANVAS_W = 9, 7                 # 1px frame so glyphs can be jittered


def render_glyph(ch, dy=0, dx=0):
    # Paint a clean 0/1 glyph onto the padded canvas, translated by (dy, dx).
    canvas = np.zeros((CANVAS_H, CANVAS_W))
    top, left = 1 + dy, 1 + dx
    for r, row in enumerate(FONT[ch]):
        for c, bit in enumerate(row):
            if bit == "1":
                canvas[top + r, left + c] = 1.0
    return canvas


def noisy_sample(ch, noise=0.10, flip_p=0.015):
    # A realistic scan: random shift, ink intensity, salt-pepper, blur-ish noise.
    img = render_glyph(ch, np.random.randint(-1, 2), np.random.randint(-1, 2))
    img *= np.random.uniform(0.7, 1.0)                       # ink darkness
    flips = np.random.rand(*img.shape) < flip_p              # speckle
    img[flips] = 1.0 - img[flips]
    img += noise * np.random.randn(*img.shape)               # sensor noise
    return np.clip(img, 0.0, 1.0)


def make_dataset(per_class):
    X, y = [], []
    for ci, ch in enumerate(CHARS):
        for _ in range(per_class):
            X.append(noisy_sample(ch).ravel())
            y.append(ci)
    return np.array(X), np.array(y)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxOCR:
    """From-scratch multinomial logistic regression (softmax) over flat pixels,
    trained by full-batch gradient descent on the cross-entropy loss."""

    def __init__(self, lr=0.5, n_iters=700, reg=1e-3):
        self.lr, self.n_iters, self.reg = lr, n_iters, reg

    def fit(self, X, y):
        n, d = X.shape
        self.k = int(y.max()) + 1
        Y = np.eye(self.k)[y]                                # one-hot targets
        self.W = np.zeros((d, self.k))
        self.b = np.zeros(self.k)
        for _ in range(self.n_iters):
            P = softmax(X @ self.W + self.b)
            dZ = (P - Y) / n                                 # softmax+CE gradient
            self.W -= self.lr * (X.T @ dZ + self.reg * self.W)
            self.b -= self.lr * dZ.sum(axis=0)
        return self

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def recognize_line(model, line_img):
    # OCR app: slice a fixed-pitch line image into character cells and read them.
    n = line_img.shape[1] // CANVAS_W
    cells = [line_img[:, i * CANVAS_W:(i + 1) * CANVAS_W].ravel() for i in range(n)]
    idx = model.predict(np.array(cells))
    return "".join(CHARS[i] for i in idx)


if __name__ == "__main__":
    np.random.seed(0)

    # Planted structure: 10 distinct glyph templates + jitter/speckle/sensor noise.
    Xtr, ytr = make_dataset(per_class=100)
    Xte, yte = make_dataset(per_class=40)                    # held-out draw

    model = SoftmaxOCR(lr=0.5, n_iters=700, reg=1e-3).fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = np.mean(pred == yte)

    # Baselines: always-guess-majority and uniform-random over 10 classes.
    counts = np.bincount(ytr)
    majority = counts.argmax()
    base_acc = np.mean(yte == majority)
    rand_acc = 1.0 / len(CHARS)

    # OCR app demo: render a full number as one jittered, noisy line image.
    text = "31415926"
    line = np.hstack([noisy_sample(ch) for ch in text])
    read = recognize_line(model, line)
    char_acc = np.mean([a == b for a, b in zip(read, text)])

    print("Classes (chars):      %s" % "".join(CHARS))
    print("Train / test images:  %d / %d  (%dx%d px each)"
          % (len(ytr), len(yte), CANVAS_H, CANVAS_W))
    print("Held-out accuracy:    %.3f" % acc)
    print("Majority baseline:    %.3f" % base_acc)
    print("Random baseline:      %.3f" % rand_acc)
    print("Improvement vs rand:  +%.1f pts" % (100 * (acc - rand_acc)))
    print("OCR app truth:        %s" % text)
    print("OCR app read:         %s" % read)
    print("Line char accuracy:   %.3f" % char_acc)
    print("Beats baseline:       %s" % (acc > max(base_acc, rand_acc) + 0.3))
