import numpy as np

def im2col(X, kh, kw):
    # Turn (N, C, H, W) into patch columns for a stride-1, valid conv.
    N, C, H, W = X.shape
    oh, ow = H - kh + 1, W - kw + 1
    cols = np.empty((N, C, kh, kw, oh, ow), dtype=X.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j] = X[:, :, i:i + oh, j:j + ow]
    # (N, oh, ow, C*kh*kw)
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N, oh, ow, -1), oh, ow

class Conv2D:
    # Valid, stride-1 convolution with F filters over C input channels.
    def __init__(self, C, F, k):
        self.k = k
        # He init: weights shaped (F, C*k*k) to match im2col columns.
        self.W = np.random.randn(F, C * k * k) * np.sqrt(2.0 / (C * k * k))
        self.b = np.zeros(F)

    def forward(self, X):
        self.X = X
        cols, oh, ow = im2col(X, self.k, self.k)   # (N, oh, ow, C*k*k)
        self.cols, self.oh, self.ow = cols, oh, ow
        out = cols @ self.W.T + self.b             # (N, oh, ow, F)
        return out.transpose(0, 3, 1, 2)          # (N, F, oh, ow)

    def backward(self, d, lr):
        N = d.shape[0]
        d = d.transpose(0, 2, 3, 1)               # (N, oh, ow, F)
        dflat = d.reshape(-1, d.shape[-1])        # (N*oh*ow, F)
        cflat = self.cols.reshape(-1, self.cols.shape[-1])
        # softmax_ce already divides the loss gradient by N, so no extra /N here.
        dW = dflat.T @ cflat
        db = dflat.sum(axis=0)
        # Backprop into inputs by scattering column gradients back.
        dcols = (dflat @ self.W).reshape(self.cols.shape)  # (N, oh, ow, C*k*k)
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
        self.shape, self.oh, self.ow = X.shape, oh, ow
        return Xr.max(axis=-1)
    def backward(self, d, lr):
        N, C, oh, ow = d.shape
        p = self.p
        dX = np.zeros((N, C, oh, ow, p * p))
        idx = self.arg
        n, c, i, j = np.indices((N, C, oh, ow))
        dX[n, c, i, j, idx] = d
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
        # d is already the mean-loss gradient (softmax_ce divides by N).
        dW = self.X.T @ d
        db = d.sum(axis=0)
        dX = d @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dX

def softmax_ce(logits, y):
    # Softmax cross-entropy loss and its gradient w.r.t. logits.
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    loss = -np.log(p[np.arange(N), y] + 1e-9).mean()
    grad = p.copy()
    grad[np.arange(N), y] -= 1
    return loss, grad / N

class CNN:
    # conv -> relu -> maxpool -> flatten -> dense (softmax) trained by SGD.
    def __init__(self, in_shape, n_filters, k, pool, n_classes):
        C, H, W = in_shape
        self.conv = Conv2D(C, n_filters, k)
        self.relu = ReLU()
        self.pool = MaxPool2D(pool)
        self.flat = Flatten()
        oh = (H - k + 1) // pool
        ow = (W - k + 1) // pool
        self.fc = Dense(n_filters * oh * ow, n_classes)
        self.layers = [self.conv, self.relu, self.pool, self.flat, self.fc]

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def fit(self, X, y, epochs=8, lr=0.1, batch=16):
        n = X.shape[0]
        for ep in range(epochs):
            perm = np.random.permutation(n)
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                logits = self.forward(X[idx])
                loss, grad = softmax_ce(logits, y[idx])
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, lr)
        return self

    def loss(self, X, y):
        return softmax_ce(self.forward(X), y)[0]

if __name__ == "__main__":
    np.random.seed(0)

    # Tiny synthetic 8x8 images: vertical bars (class 0) vs horizontal bars (1).
    def make(n):
        X = np.zeros((n, 1, 8, 8))
        y = np.zeros(n, dtype=int)
        for i in range(n):
            if i % 2 == 0:                       # vertical bar
                cols = np.random.choice(8, 2, replace=False)
                X[i, 0, :, cols] = 1.0
                y[i] = 0
            else:                                # horizontal bar
                rows = np.random.choice(8, 2, replace=False)
                X[i, 0, rows, :] = 1.0
                y[i] = 1
        X += 0.1 * np.random.randn(n, 1, 8, 8)   # small noise
        return X, y

    Xtr, ytr = make(100)
    Xte, yte = make(40)

    model = CNN(in_shape=(1, 8, 8), n_filters=4, k=3, pool=2, n_classes=2)

    start_loss = model.loss(Xtr, ytr)
    model.fit(Xtr, ytr, epochs=30, lr=0.2, batch=16)
    final_loss = model.loss(Xtr, ytr)

    acc = np.mean(model.predict(Xte) == yte)
    print("Start training loss :", round(float(start_loss), 4))
    print("Final training loss :", round(float(final_loss), 4))
    print("Loss decreased      :", final_loss < start_loss * 0.5)
    print("Test accuracy       :", round(float(acc), 4))
