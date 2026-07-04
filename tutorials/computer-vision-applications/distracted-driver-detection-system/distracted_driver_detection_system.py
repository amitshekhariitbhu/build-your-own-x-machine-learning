import numpy as np

# Distracted Driver Detection System from scratch (multinomial softmax classifier).
#
# Each sample is a tiny 20x20 grayscale "dashcam" frame of a driver. Every frame
# shares a head (top) and a steering wheel (bottom center) so those are NOT
# discriminative -- the ACTIVITY is encoded purely by WHERE the bright hand/arm
# is, i.e. the driver's pose:
#   0 safe driving  : both hands gripping the wheel (bright at wheel's two sides)
#   1 phone call    : arm raised to the ear            (bright upper-right)
#   2 texting       : hands + phone glow in the lap    (bright lower-center)
#   3 operating radio: arm reaching to the console     (bright center-right)
# Everything is hand-rolled: we generate the frames, then train a softmax
# (multinomial logistic) classifier by full-batch gradient descent on the raw
# normalized pixels and report held-out accuracy vs a majority-class baseline.

H = W = 20                                   # frame size
CLASSES = ["safe_driving", "phone_call", "texting", "operating_radio"]
YY, XX = np.mgrid[0:H, 0:W]                  # reusable pixel coordinate grids


def blob(img, cy, cx, r, val, rng):
    # Paint a soft bright disc (a hand / phone glow) at (cy, cx).
    m = (XX - cx) ** 2 + (YY - cy) ** 2 <= r * r
    img[m] = np.clip(val + rng.normal(0, 0.04, int(m.sum())), 0, 1)


def make_driver(cls, rng):
    # Draw one driver frame whose pose encodes the activity `cls`.
    img = np.clip(0.40 + rng.normal(0, 0.05, (H, W)), 0, 1)      # cabin background
    # Shared, non-discriminative anatomy: head (top) + steering-wheel ring (bottom).
    img[(XX - 10) ** 2 + (YY - 3) ** 2 <= 2.5 ** 2] = 0.75       # head
    ring = np.abs(np.sqrt((XX - 10) ** 2 + (YY - 15) ** 2) - 5.0) <= 0.9
    img[ring] = 0.20                                             # dark wheel rim
    jy, jx = rng.randint(-1, 2), rng.randint(-1, 2)             # pose jitter
    if cls == 0:                                                 # safe: both hands on wheel
        blob(img, 15 + jy, 5 + jx, 1.8, 0.92, rng)
        blob(img, 15 + jy, 15 + jx, 1.8, 0.92, rng)
    elif cls == 1:                                               # phone call: arm to ear
        blob(img, 4 + jy, 15 + jx, 2.2, 0.92, rng)
    elif cls == 2:                                               # texting: phone in lap
        blob(img, 18 + jy, 10 + jx, 2.2, 0.92, rng)
    else:                                                        # radio: reach to console
        blob(img, 11 + jy, 17 + jx, 2.2, 0.92, rng)
    return img.ravel()


def make_dataset(n_per, rng):
    X, y = [], []
    for cls in range(len(CLASSES)):
        for _ in range(n_per):
            X.append(make_driver(cls, rng))
            y.append(cls)
    return np.array(X), np.array(y)


class SoftmaxClassifier:
    """Multinomial logistic regression trained by full-batch gradient descent."""

    def __init__(self, n_classes, lr=0.5, epochs=400, reg=1e-3):
        self.k, self.lr, self.epochs, self.reg = n_classes, lr, epochs, reg

    def _softmax(self, Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - self.mu) / self.sd
        n, d = Xs.shape
        self.W = np.zeros((d, self.k))
        self.b = np.zeros(self.k)
        Y = np.eye(self.k)[y]                                    # one-hot targets
        self.hist = []
        for _ in range(self.epochs):
            P = self._softmax(Xs @ self.W + self.b)
            self.hist.append(-np.mean(np.log(P[np.arange(n), y] + 1e-9)))
            G = (P - Y) / n
            self.W -= self.lr * (Xs.T @ G + self.reg * self.W)
            self.b -= self.lr * G.sum(0)
        return self

    def predict_proba(self, X):
        return self._softmax((X - self.mu) / self.sd @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.RandomState(0)

    X, y = make_dataset(n_per=120, rng=rng)                      # 4 * 120 = 480 frames
    idx = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]

    clf = SoftmaxClassifier(n_classes=len(CLASSES)).fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    acc = (pred == y[te]).mean()

    # Majority-class baseline (classes are balanced -> ~1/4).
    counts = np.bincount(y[tr], minlength=len(CLASSES))
    majority = counts.max() / len(tr)

    print("Activities           :", ", ".join(CLASSES))
    print("Train / test frames  : {} / {}".format(len(tr), len(te)))
    print("Start training loss  : {:.4f}".format(clf.hist[0]))
    print("Final training loss  : {:.4f}".format(clf.hist[-1]))
    print("-" * 54)
    print("Per-activity recall (held-out):")
    for c, name in enumerate(CLASSES):
        m = y[te] == c
        print("  {:<16}: {:.3f}".format(name, (pred[m] == c).mean()))
    print("-" * 54)
    print("Majority baseline acc: {:.3f}".format(majority))
    print("Softmax test accuracy: {:.3f}".format(acc))
    print("Beats baseline       :", bool(acc > majority + 0.3))
