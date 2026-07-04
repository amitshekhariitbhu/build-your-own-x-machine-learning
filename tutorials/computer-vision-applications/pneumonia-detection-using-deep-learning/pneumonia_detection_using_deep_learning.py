import numpy as np

# Pneumonia detection from chest X-rays with a tiny from-scratch CNN.
# Synthetic 12x12 grayscale X-rays: NORMAL lungs are clear, PNEUMONIA lungs
# carry bright localized opacities (infiltrates). A conv->relu->pool->dense
# network learns to spot those planted patches.

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
    # Valid, stride-1 convolution: F filters over C input channels.
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
        d = d.transpose(0, 2, 3, 1).reshape(-1, d.shape[1])   # (N*oh*ow, F)
        cflat = self.cols.reshape(-1, self.cols.shape[-1])
        dW = d.T @ cflat
        db = d.sum(axis=0)
        dcols = (d @ self.W).reshape(self.cols.shape)         # (N, oh, ow, C*k*k)
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
    # Softmax cross-entropy loss and gradient w.r.t. logits (mean over batch).
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    loss = -np.log(p[np.arange(N), y] + 1e-9).mean()
    grad = p.copy(); grad[np.arange(N), y] -= 1
    return loss, grad / N

class PneumoniaCNN:
    # conv -> relu -> maxpool -> flatten -> dense(softmax), trained by SGD.
    def __init__(self, in_shape, n_filters=6, k=3, pool=2, n_classes=2):
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
    def fit(self, X, y, epochs=15, lr=0.1, batch=16):
        n = X.shape[0]
        for _ in range(epochs):
            perm = np.random.permutation(n)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                loss, grad = softmax_ce(self.forward(X[idx]), y[idx])
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, lr)
        return self

def make_xrays(n, size=12, p_pneu=0.6):
    # Build synthetic chest X-rays. Base = two dark lung fields with faint rib
    # texture; pneumonia adds 1-2 bright Gaussian opacities inside a lung.
    yy, xx = np.mgrid[0:size, 0:size]
    X = np.zeros((n, 1, size, size)); y = np.zeros(n, dtype=int)
    lung_cx = [size * 0.3, size * 0.7]                 # left / right lung centers
    for i in range(n):
        img = 0.35 + 0.05 * np.sin(yy * 1.2)           # faint horizontal ribs
        img[:, size // 2 - 1:size // 2 + 1] += 0.25    # bright mediastinum column
        if np.random.rand() < p_pneu:
            y[i] = 1
            for _ in range(np.random.randint(1, 3)):   # plant opacities
                cx = np.random.choice(lung_cx) + np.random.randn()
                cy = size * 0.5 + np.random.randn() * 2
                blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.6 ** 2))
                img += (0.5 + 0.2 * np.random.rand()) * blob
        X[i, 0] = img
    X += 0.05 * np.random.randn(*X.shape)              # sensor noise
    return X, y

if __name__ == "__main__":
    np.random.seed(0)

    Xtr, ytr = make_xrays(200)
    Xte, yte = make_xrays(80)

    model = PneumoniaCNN(in_shape=(1, 12, 12))
    model.fit(Xtr, ytr, epochs=15, lr=0.1, batch=16)

    pred = model.predict(Xte)
    acc = np.mean(pred == yte)
    majority = np.bincount(ytr).argmax()               # baseline: always predict it
    base_acc = np.mean(yte == majority)

    tp = int(np.sum((pred == 1) & (yte == 1)))
    fp = int(np.sum((pred == 1) & (yte == 0)))
    fn = int(np.sum((pred == 0) & (yte == 1)))
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    print("Test images           :", len(yte))
    print("Majority baseline acc :", round(float(base_acc), 4))
    print("CNN test accuracy     :", round(float(acc), 4))
    print("Pneumonia precision   :", round(float(prec), 4))
    print("Pneumonia recall      :", round(float(rec), 4))
    print("Pneumonia F1          :", round(float(f1), 4))
    print("Beats baseline        :", acc > base_acc)
