import numpy as np

# Handwriting recognition for form fields, from scratch.
# Forms carry hand-printed digit fields (ZIP, phone, dates). Each digit is a
# 7x5 ink bitmap; "handwriting" is that same glyph with random 1px pen shifts,
# ink dropout and sensor noise, so the latent per-digit shape stays recoverable.
# A from-scratch softmax classifier learns pixels -> digit, then we read whole
# form fields (sequences of glyphs) back into text -- the recognition payoff.

H, W = 7, 5                                   # glyph grid
GLYPHS = {                                    # hand-printed digit templates
    "0": [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    "3": ["####.", "....#", "....#", ".###.", "....#", "....#", "####."],
    "4": ["#...#", "#...#", "#...#", "#####", "....#", "....#", "....#"],
    "5": ["#####", "#....", "#....", "####.", "....#", "....#", "####."],
    "6": [".###.", "#....", "#....", "####.", "#...#", "#...#", ".###."],
    "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    "8": [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    "9": [".###.", "#...#", "#...#", ".####", "....#", "....#", ".###."],
}
DIGITS = list(GLYPHS)                          # class labels "0".."9"


def render(rows):
    # ASCII template -> float ink image (1 = ink, 0 = paper).
    return np.array([[1.0 if c == "#" else 0.0 for c in r] for r in rows])


TEMPLATES = {d: render(rows) for d, rows in GLYPHS.items()}


def scribble(img, rng, noise=0.3, drop=0.06, shift_p=0.4):
    # Simulate one handwritten instance: pen shift + ink dropout + Gaussian ink.
    g = img.copy()
    if rng.random() < shift_p:                 # random 1px pen offset
        dy, dx = rng.randint(-1, 2), rng.randint(-1, 2)
        g = np.roll(np.roll(g, dy, axis=0), dx, axis=1)
        if dy > 0: g[0, :] = 0
        elif dy < 0: g[-1, :] = 0
        if dx > 0: g[:, 0] = 0
        elif dx < 0: g[:, -1] = 0
    g[(rng.random(g.shape) < drop) & (g > 0)] = 0        # skipped ink
    g = g + rng.randn(*g.shape) * noise                   # smudge / sensor noise
    return np.clip(g, 0, None).ravel()


def make_forms(n_per=100, seed=0):
    # Plant noisy handwritten samples for every digit template.
    rng = np.random.RandomState(seed)
    X, y = [], []
    for c, d in enumerate(DIGITS):
        for _ in range(n_per):
            X.append(scribble(TEMPLATES[d], rng))
            y.append(c)
    X, y = np.array(X), np.array(y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


class SoftmaxClassifier:
    """Multinomial logistic regression via manual batch gradient descent."""

    def __init__(self, epochs=400, lr=0.5, reg=1e-3, seed=0):
        self.epochs, self.lr, self.reg, self.seed = epochs, lr, reg, seed

    def _standardize(self, X, fit=False):
        if fit:
            self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        return (X - self.mu) / self.sd

    @staticmethod
    def _softmax(Z):
        E = np.exp(Z - Z.max(1, keepdims=True))
        return E / E.sum(1, keepdims=True)

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        Xb = np.hstack([self._standardize(np.asarray(X, float), True),
                        np.ones((len(X), 1))])            # bias column
        self.K = int(y.max()) + 1
        Y = np.eye(self.K)[y]
        self.W = rng.randn(Xb.shape[1], self.K) * 0.01
        for _ in range(self.epochs):
            grad = Xb.T @ (self._softmax(Xb @ self.W) - Y) / len(Xb) + self.reg * self.W
            self.W -= self.lr * grad
        return self

    def predict(self, X):
        Xb = np.hstack([self._standardize(np.asarray(X, float)),
                        np.ones((len(X), 1))])
        return np.argmax(self._softmax(Xb @ self.W), axis=1)


def read_field(text, clf, rng):
    # Recognize a whole form field: one noisy glyph per character.
    imgs = np.array([scribble(TEMPLATES[ch], rng) for ch in text])
    return "".join(DIGITS[i] for i in clf.predict(imgs))


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_forms(n_per=100, seed=0)
    split = int(0.7 * len(y))                             # held-out split
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]

    clf = SoftmaxClassifier(epochs=400, lr=0.5, reg=1e-3).fit(Xtr, ytr)
    acc = np.mean(clf.predict(Xte) == yte)

    majority = np.bincount(ytr).argmax()                 # baseline: guess mode
    base_acc = np.mean(yte == majority)

    print("Handwriting-on-forms: %d glyphs, %d digit classes, %d pixels"
          % (len(y), len(DIGITS), H * W))
    print("Train: %d   Test: %d" % (len(ytr), len(yte)))
    print("-" * 58)
    print("Softmax recognizer accuracy: %.4f" % acc)
    print("Majority baseline accuracy:  %.4f  (chance = %.4f)"
          % (base_acc, 1.0 / len(DIGITS)))
    print("-" * 58)

    rng = np.random.RandomState(7)
    fields = {"ZIP": "90210", "PHONE": "5551212", "DATE": "07041776"}
    total, correct = 0, 0
    for name, value in fields.items():
        got = read_field(value, clf, rng)
        hits = sum(a == b for a, b in zip(got, value))
        total += len(value); correct += hits
        print("  %-6s wrote '%s' -> read '%s'   %d/%d chars"
              % (name, value, got, hits, len(value)))
    print("-" * 58)
    print("Field char accuracy: %.4f   (random = %.4f)"
          % (correct / total, 1.0 / len(DIGITS)))
    print("Beats majority baseline: %s" % (acc > base_acc))
