import numpy as np

# Real-time hand gesture recognition from scratch.
# Each frame is a tiny grayscale hand SILHOUETTE: a solid palm at the bottom
# with 0..4 raised finger bars. The number of raised fingers is the gesture:
#   0 = fist, 1 = point, 2 = peace, 3 = three, 4 = open hand.
# We extract a fast, translation-tolerant feature (column ink profile) and
# train a plain softmax classifier by gradient descent -- light enough to run
# per frame in "real time".

H, W = 18, 18          # frame size
SLOTS = [3, 6, 9, 12]  # x-centers of the 4 possible fingers

def draw_hand(n_fingers):
    # Render one hand silhouette with n_fingers raised, plus pose/sensor noise.
    img = np.zeros((H, W))
    dx = np.random.randint(-1, 2)                    # whole-hand shift
    palm_top = 10 + np.random.randint(-1, 2)
    img[palm_top:H, 2 + dx:15 + dx] = 1.0            # solid palm block
    for k in range(n_fingers):                       # raise the first k fingers
        cx = SLOTS[k] + dx
        length = np.random.randint(4, 7)             # finger length jitter
        img[palm_top - length:palm_top, cx:cx + 2] = 1.0
    img += 0.18 * np.random.randn(H, W)              # sensor noise
    return np.clip(img, 0, 1)

def make_frames(n):
    # Balanced dataset of gesture frames labelled by finger count 0..4.
    X = np.zeros((n, H, W))
    y = np.arange(n) % 5
    for i in range(n):
        X[i] = draw_hand(y[i])
    return X, y

def features(X):
    # Column ink profile: how much "hand" is in each column, above vs below.
    # Fingers add ink to upper columns, so this cleanly counts raised fingers
    # while staying robust to small vertical shifts. Cheap = real-time.
    upper = X[:, :H // 2, :].sum(axis=1)             # (n, W) upper-half ink
    lower = X[:, H // 2:, :].sum(axis=1)             # (n, W) lower-half ink
    total_upper = upper.sum(axis=1, keepdims=True)   # overall finger ink
    F = np.concatenate([upper, lower, total_upper], axis=1)
    return F

class SoftmaxClassifier:
    # Multinomial logistic regression trained with full-batch gradient descent.
    def __init__(self, n_classes, lr=0.15, l2=1e-3, iters=400):
        self.n_classes, self.lr, self.l2, self.iters = n_classes, lr, l2, iters

    def _softmax(self, Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        mu, sd = X.mean(0), X.std(0) + 1e-8
        self.mu, self.sd = mu, sd
        Xn = (X - mu) / sd
        n, d = Xn.shape
        self.W = np.zeros((d, self.n_classes))
        self.b = np.zeros(self.n_classes)
        Y = np.eye(self.n_classes)[y]                # one-hot targets
        for _ in range(self.iters):
            P = self._softmax(Xn @ self.W + self.b)
            G = (P - Y) / n
            self.W -= self.lr * (Xn.T @ G + self.l2 * self.W)
            self.b -= self.lr * G.sum(0)
        return self

    def predict_proba(self, X):
        Xn = (X - self.mu) / self.sd
        return self._softmax(Xn @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


if __name__ == "__main__":
    np.random.seed(0)

    Xtr_img, ytr = make_frames(600)
    Xte_img, yte = make_frames(250)                  # held-out frames

    clf = SoftmaxClassifier(n_classes=5).fit(features(Xtr_img), ytr)
    pred = clf.predict(features(Xte_img))
    acc = np.mean(pred == yte)

    # Baseline: always guess the most common gesture in the training set.
    baseline = np.bincount(ytr).max() / len(ytr)

    # Per-class recall to show every gesture is recognised, not just one.
    names = ["fist", "point", "peace", "three", "open"]
    print("Gestures            :", ", ".join(names))
    print("Train / test frames :", len(ytr), "/", len(yte))
    print("Majority baseline   :", round(float(baseline), 4))
    print("Gesture accuracy    :", round(float(acc), 4))
    print("Beats baseline      :", bool(acc > baseline + 0.3))
    print("Per-gesture recall  :")
    for c, nm in enumerate(names):
        rec = np.mean(pred[yte == c] == c)
        print("   {:5s} : {:.3f}".format(nm, float(rec)))

    # Simulate a real-time stream: classify frames one at a time.
    hits = sum(int(clf.predict(features(draw_hand(g)[None]))[0] == g)
               for g in np.random.randint(0, 5, 100))
    print("Live-stream 100 frames accuracy :", round(hits / 100.0, 4))
