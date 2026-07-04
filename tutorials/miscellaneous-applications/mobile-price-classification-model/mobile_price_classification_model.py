import numpy as np


class MobilePriceClassifier:
    """Softmax (multinomial logistic) regression trained from scratch.

    Standardizes phone-spec features, learns a weight matrix + bias by
    full-batch gradient descent on the L2-regularized cross-entropy loss,
    and predicts one of 4 price ranges (0=low ... 3=very high).
    No ML libraries -- just numpy math.
    """

    def __init__(self, n_classes=4, lr=0.5, n_iter=800, l2=1e-3):
        self.n_classes = n_classes
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2  # L2 regularization strength
        self.W = None
        self.b = None
        self.mu = None
        self.sigma = None

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)  # stability shift
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def _standardize(self, X):
        return (X - self.mu) / self.sigma

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).reshape(-1)
        # scaling stats computed on training data only (no test leakage)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Xs = self._standardize(X)
        n, d = Xs.shape
        k = self.n_classes
        Y = np.eye(k)[y]  # one-hot targets
        self.W = np.zeros((d, k))
        self.b = np.zeros(k)
        for _ in range(self.n_iter):
            P = self._softmax(Xs @ self.W + self.b)  # class probabilities
            err = P - Y
            grad_W = Xs.T @ err / n + self.l2 * self.W
            grad_b = err.mean(axis=0)
            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        Xs = self._standardize(np.asarray(X, dtype=float))
        return self._softmax(Xs @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def make_phone_data(n=1200):
    """Synthetic phone specs with a planted price-range signal.

    Features: battery_power, ram, px_width, px_height, int_memory,
    clock_speed, mobile_wt, n_cores.  A latent "value" score (dominated by
    RAM, then resolution / battery / storage) is split into 4 balanced
    quantile bins -> price ranges 0..3, plus feature noise so classes overlap.
    """
    battery = np.random.uniform(500, 2000, n)
    ram = np.random.uniform(256, 4000, n)
    px_w = np.random.uniform(500, 2000, n)
    px_h = np.random.uniform(400, 1900, n)
    int_mem = np.random.uniform(2, 64, n)
    clock = np.random.uniform(0.5, 3.0, n)
    weight = np.random.uniform(80, 200, n)       # heavier ~ cheaper build
    cores = np.random.randint(1, 9, n)
    X = np.column_stack([battery, ram, px_w, px_h,
                         int_mem, clock, weight, cores])

    # planted latent price score: RAM dominates (like the real dataset)
    score = (0.0016 * ram + 0.0006 * battery + 0.0005 * px_w
             + 0.0004 * px_h + 0.020 * int_mem + 0.25 * clock
             - 0.004 * weight + 0.10 * cores)
    score += np.random.normal(0, 0.35, n)        # overlap noise
    # balanced quantile bins -> 4 ordered price ranges
    q = np.quantile(score, [0.25, 0.50, 0.75])
    y = np.digitize(score, q)
    return X, y


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_phone_data(1200)

    # held-out train/test split
    idx = np.random.permutation(len(y))
    cut = int(0.75 * len(y))
    tr, te = idx[:cut], idx[cut:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    model = MobilePriceClassifier(n_classes=4, lr=0.5, n_iter=800).fit(Xtr, ytr)
    pred = model.predict(Xte)

    acc = np.mean(pred == yte)
    # majority-class baseline (predict most common training label)
    majority = np.bincount(ytr).argmax()
    base_acc = np.mean(yte == majority)

    # macro-averaged F1 across the 4 price ranges
    f1s = []
    for c in range(4):
        tp = np.sum((pred == c) & (yte == c))
        fp = np.sum((pred == c) & (yte != c))
        fn = np.sum((pred != c) & (yte == c))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    macro_f1 = float(np.mean(f1s))

    feats = ["battery", "ram", "px_w", "px_h",
             "int_mem", "clock", "weight", "cores"]
    # mean |weight| per feature shows RAM drives the prediction
    importance = dict(zip(feats, np.round(np.abs(model.W).mean(axis=1), 3)))

    print("test samples          :", len(yte))
    print("num price ranges      :", 4)
    print("majority baseline acc :", round(float(base_acc), 3))
    print("model accuracy        :", round(float(acc), 3))
    print("macro F1 score        :", round(macro_f1, 3))
    print("top feature (|w|)     :", max(importance, key=importance.get))
    print("feature importance    :", importance)
    print("BEATS baseline        :", bool(acc > base_acc))
