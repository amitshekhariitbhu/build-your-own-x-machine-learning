import numpy as np

# Cats vs Dogs classifier built from scratch with a tiny CNN.
# Each pet is a tiny grayscale "face" whose planted EARS encode the class:
#   0 = cat -> two upright pointy triangular ears on top,
#   1 = dog -> two floppy rounded ears drooping down the sides.
# conv -> relu -> maxpool -> flatten -> dense (softmax), trained by SGD.

def im2col(X, k):
    # Turn (N, C, H, W) into patch columns for a stride-1, valid conv.
    N, C, H, W = X.shape
    oh, ow = H - k + 1, W - k + 1
    cols = np.empty((N, C, k, k, oh, ow), dtype=X.dtype)
    for i in range(k):
        for j in range(k):
            cols[:, :, i, j] = X[:, :, i:i + oh, j:j + ow]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N, oh, ow, -1), oh, ow

class Conv2D:
    # Valid, stride-1 convolution with F filters over C input channels.
    def __init__(self, C, F, k):
        self.k = k
        self.W = np.random.randn(F, C * k * k) * np.sqrt(2.0 / (C * k * k))
        self.b = np.zeros(F)
    def forward(self, X):
        self.X = X
        cols, oh, ow = im2col(X, self.k)          # (N, oh, ow, C*k*k)
        self.cols, self.oh, self.ow = cols, oh, ow
        out = cols @ self.W.T + self.b            # (N, oh, ow, F)
        return out.transpose(0, 3, 1, 2)         # (N, F, oh, ow)
    def backward(self, d, lr):
        N = d.shape[0]
        d = d.transpose(0, 2, 3, 1).reshape(-1, d.shape[1])
        cflat = self.cols.reshape(-1, self.cols.shape[-1])
        dW = d.T @ cflat
        db = d.sum(axis=0)
        dcols = (d @ self.W).reshape(self.cols.shape)
        C, k = self.X.shape[1], self.k
        dcols = dcols.reshape(N, self.oh, self.ow, C, k, k).transpose(0, 3, 4, 5, 1, 2)
        dX = np.zeros_like(self.X)
        for i in range(k):
            for j in range(k):
                dX[:, :, i:i + self.oh, j:j + self.ow] += dcols[:, :, i, j]
        self.W -= lr * dW
        self.b -= lr * db
        return dX

class ReLU:
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask
    def backward(self, d, lr):
        return d * self.mask

class MaxPool2D:
    # Non-overlapping p x p max pooling.
    def __init__(self, p=2):
        self.p = p
    def forward(self, X):
        N, C, H, W = X.shape
        p = self.p
        oh, ow = H // p, W // p
        Xr = X[:, :, :oh * p, :ow * p].reshape(N, C, oh, p, ow, p)
        Xr = Xr.transpose(0, 1, 2, 4, 3, 5).reshape(N, C, oh, ow, p * p)
        self.arg = Xr.argmax(axis=-1)
        return Xr.max(axis=-1)
    def backward(self, d, lr):
        N, C, oh, ow = d.shape
        p = self.p
        dX = np.zeros((N, C, oh, ow, p * p))
        n, c, i, j = np.indices((N, C, oh, ow))
        dX[n, c, i, j, self.arg] = d
        dX = dX.reshape(N, C, oh, ow, p, p).transpose(0, 1, 2, 4, 3, 5)
        return dX.reshape(N, C, oh * p, ow * p)

class Flatten:
    def forward(self, X):
        self.shape = X.shape
        return X.reshape(X.shape[0], -1)
    def backward(self, d, lr):
        return d.reshape(self.shape)

class Dense:
    def __init__(self, nin, nout):
        self.W = np.random.randn(nin, nout) * np.sqrt(2.0 / nin)
        self.b = np.zeros(nout)
    def forward(self, X):
        self.X = X
        return X @ self.W + self.b
    def backward(self, d, lr):
        dW = self.X.T @ d
        db = d.sum(axis=0)
        dX = d @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dX

def softmax_ce(logits, y):
    # Softmax cross-entropy loss and gradient w.r.t. logits (already /N).
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    loss = -np.log(p[np.arange(N), y] + 1e-9).mean()
    grad = p.copy()
    grad[np.arange(N), y] -= 1
    return loss, grad / N

class CNN:
    # conv -> relu -> maxpool -> flatten -> dense (softmax).
    def __init__(self, in_shape, n_filters, k, pool, n_classes):
        C, H, W = in_shape
        self.conv = Conv2D(C, n_filters, k)
        self.relu = ReLU()
        self.pool = MaxPool2D(pool)
        self.flat = Flatten()
        oh, ow = (H - k + 1) // pool, (W - k + 1) // pool
        self.fc = Dense(n_filters * oh * ow, n_classes)
        self.layers = [self.conv, self.relu, self.pool, self.flat, self.fc]
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    def predict(self, X):
        return self.forward(X).argmax(axis=1)
    def fit(self, X, y, epochs=12, lr=0.15, batch=16):
        n = X.shape[0]
        for _ in range(epochs):
            perm = np.random.permutation(n)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                loss, grad = softmax_ce(self.forward(X[idx]), y[idx])
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, lr)
        return self
    def loss(self, X, y):
        return softmax_ce(self.forward(X), y)[0]

def make_pets(n):
    # Draw cat/dog faces on 16x16 grayscale tiles with jitter + noise.
    H = 16
    X = np.zeros((n, 1, H, H))
    y = np.arange(n) % 2
    for i in range(n):
        oy, ox = np.random.randint(-1, 2, 2)           # small position jitter
        cy, cx = 8.5 + oy, 7.5 + ox
        yy, xx = np.mgrid[0:H, 0:H]
        img = np.zeros((H, H))
        img[(xx - cx) ** 2 + (yy - cy) ** 2 <= 16] = 1.0   # round face
        if y[i] == 0:                                  # cat: pointy ears up
            for ex in (cx - 4, cx + 4):
                dy, dx = yy - (cy - 5), xx - ex
                img[(dy >= -3) & (dy <= 0) & (np.abs(dx) <= (dy + 3) * 0.7)] = 1.0
        else:                                          # dog: floppy ears down
            for ex in (cx - 5, cx + 5):
                img[(xx - ex) ** 2 + (yy - (cy + 1)) ** 2 <= 5] = 1.0
        X[i, 0] = img
    X += 0.15 * np.random.randn(n, 1, H, H)            # sensor noise
    return X, y

if __name__ == "__main__":
    np.random.seed(0)

    Xtr, ytr = make_pets(240)                          # held-out split
    Xte, yte = make_pets(120)

    model = CNN(in_shape=(1, 16, 16), n_filters=6, k=3, pool=2, n_classes=2)
    start_loss = model.loss(Xtr, ytr)
    model.fit(Xtr, ytr, epochs=12, lr=0.15, batch=16)
    final_loss = model.loss(Xtr, ytr)

    acc = np.mean(model.predict(Xte) == yte)
    # Baseline: always guess the most common class (2 balanced classes).
    counts = np.bincount(ytr, minlength=2)
    baseline = counts.max() / len(ytr)

    print("Classes             : cat, dog")
    print("Start training loss :", round(float(start_loss), 4))
    print("Final training loss :", round(float(final_loss), 4))
    print("Loss decreased      :", bool(final_loss < start_loss * 0.5))
    print("Majority baseline   :", round(float(baseline), 4))
    print("CNN test accuracy   :", round(float(acc), 4))
    print("Beats baseline      :", bool(acc > baseline + 0.2))
