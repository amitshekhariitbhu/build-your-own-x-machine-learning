import numpy as np

def im2col(X, kh, kw, stride, pad):
    # (N, C, H, W) -> patch columns for a strided, padded conv.
    N, C, H, W = X.shape
    if pad:
        X = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Hp, Wp = X.shape[2], X.shape[3]
    oh = (Hp - kh) // stride + 1
    ow = (Wp - kw) // stride + 1
    cols = np.empty((N, C, kh, kw, oh, ow), dtype=X.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j] = X[:, :, i:i + stride * oh:stride, j:j + stride * ow:stride]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N, oh, ow, -1), oh, ow

def col2im(cols, Xshape, kh, kw, stride, pad, oh, ow):
    # Scatter column gradients back to input positions (overlaps accumulate).
    N, C, H, W = Xshape
    cols = cols.reshape(N, oh, ow, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    dX = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)
    for i in range(kh):
        for j in range(kw):
            dX[:, :, i:i + stride * oh:stride, j:j + stride * ow:stride] += cols[:, :, i, j]
    return dX[:, :, pad:pad + H, pad:pad + W] if pad else dX

class Conv2D:
    # Strided, padded convolution with F filters over C input channels.
    def __init__(self, C, F, k, stride=1, pad=0):
        self.C, self.F, self.k, self.stride, self.pad = C, F, k, stride, pad
        self.W = (np.random.randn(F, C * k * k) * np.sqrt(2.0 / (C * k * k))).astype(np.float32)
        self.b = np.zeros(F, dtype=np.float32)

    def forward(self, X):
        self.Xshape = X.shape
        cols, oh, ow = im2col(X, self.k, self.k, self.stride, self.pad)
        self.cols, self.oh, self.ow = cols, oh, ow
        out = cols @ self.W.T + self.b          # (N, oh, ow, F)
        return out.transpose(0, 3, 1, 2)        # (N, F, oh, ow)

    def backward(self, d, lr):
        N = d.shape[0]
        d = d.transpose(0, 2, 3, 1).reshape(-1, self.F)    # (N*oh*ow, F)
        cflat = self.cols.reshape(-1, self.cols.shape[-1])
        dW = d.T @ cflat / N
        db = d.mean(axis=0)
        dcols = d @ self.W                                 # (N*oh*ow, C*k*k)
        dX = col2im(dcols, self.Xshape, self.k, self.k, self.stride, self.pad, self.oh, self.ow)
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
    # k x k max pooling with a stride (AlexNet uses overlapping 3x3 / stride 2).
    def __init__(self, k, stride):
        self.k, self.stride = k, stride

    def forward(self, X):
        N, C, H, W = X.shape
        k, s = self.k, self.stride
        oh, ow = (H - k) // s + 1, (W - k) // s + 1
        win = np.empty((N, C, k, k, oh, ow), dtype=X.dtype)
        for i in range(k):
            for j in range(k):
                win[:, :, i, j] = X[:, :, i:i + s * oh:s, j:j + s * ow:s]
        win = win.reshape(N, C, k * k, oh, ow)
        self.arg = win.argmax(axis=2)                      # winning index per window
        self.Xshape = X.shape
        return win.max(axis=2)

    def backward(self, d, lr):
        N, C, oh, ow = d.shape
        k, s = self.k, self.stride
        dX = np.zeros(self.Xshape, dtype=d.dtype)
        ii, jj = self.arg // k, self.arg % k
        n, c, oy, ox = np.indices((N, C, oh, ow))
        np.add.at(dX, (n, c, oy * s + ii, ox * s + jj), d)  # overlaps sum
        return dX

class Flatten:
    def forward(self, X):
        self.shape = X.shape
        return X.reshape(X.shape[0], -1)
    def backward(self, d, lr):
        return d.reshape(self.shape)

class Dense:
    def __init__(self, nin, nout):
        self.W = (np.random.randn(nin, nout) * np.sqrt(2.0 / nin)).astype(np.float32)
        self.b = np.zeros(nout, dtype=np.float32)
    def forward(self, X):
        self.X = X
        return X @ self.W + self.b
    def backward(self, d, lr):
        dW = self.X.T @ d / d.shape[0]
        db = d.mean(axis=0)
        dX = d @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dX

def softmax_ce(logits, y):
    # Softmax cross-entropy loss and gradient w.r.t. logits.
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    loss = -np.log(p[np.arange(N), y] + 1e-9).mean()
    grad = p.copy()
    grad[np.arange(N), y] -= 1
    return loss, grad / N

class AlexNet:
    # Canonical AlexNet: 5 conv (+ReLU/pool) -> 3 FC, 1000-way output.
    def __init__(self, n_classes=1000):
        self.layers = [
            ("conv1", Conv2D(3, 96, 11, stride=4, pad=0)), ("relu1", ReLU()),
            ("pool1", MaxPool2D(3, 2)),
            ("conv2", Conv2D(96, 256, 5, stride=1, pad=2)), ("relu2", ReLU()),
            ("pool2", MaxPool2D(3, 2)),
            ("conv3", Conv2D(256, 384, 3, stride=1, pad=1)), ("relu3", ReLU()),
            ("conv4", Conv2D(384, 384, 3, stride=1, pad=1)), ("relu4", ReLU()),
            ("conv5", Conv2D(384, 256, 3, stride=1, pad=1)), ("relu5", ReLU()),
            ("pool3", MaxPool2D(3, 2)),
            ("flat", Flatten()),
            ("fc6", Dense(256 * 6 * 6, 4096)), ("relu6", ReLU()),
            ("fc7", Dense(4096, 4096)), ("relu7", ReLU()),
            ("fc8", Dense(4096, n_classes)),
        ]

    def forward(self, X, verbose=False):
        for name, layer in self.layers:
            X = layer.forward(X)
            if verbose:
                print("  {:6s} -> {}".format(name, X.shape))
        return X

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def n_params(self):
        return sum(l.W.size + l.b.size for _, l in self.layers if hasattr(l, "W"))

    def fit(self, X, y, steps=12, lr=0.01):
        for _ in range(steps):
            loss, grad = softmax_ce(self.forward(X), y)
            for _, layer in reversed(self.layers):
                grad = layer.backward(grad, lr)
        return self

    def loss(self, X, y):
        return softmax_ce(self.forward(X), y)[0]

if __name__ == "__main__":
    np.random.seed(0)
    net = AlexNet(n_classes=1000)

    # (1) Architecture check: one forward pass on a synthetic 227x227x3 image.
    print("=== AlexNet forward pass (architecture wiring) ===")
    x = np.random.randn(1, 3, 227, 227).astype(np.float32)
    print("  input  -> {}".format(x.shape))
    out = net.forward(x, verbose=True)
    print("Output shape       :", out.shape, " (expected (1, 1000))")
    print("Total parameters   : {:,}".format(net.n_params()))

    # (2) Learning check: overfit a tiny synthetic batch to prove backprop works.
    print("\n=== Training (overfit a tiny synthetic batch) ===")
    n = 4
    X = 0.1 * np.random.randn(n, 3, 227, 227).astype(np.float32)
    y = np.array([0, 1, 0, 1])
    for i in range(n):                      # class 0 = vertical, 1 = horizontal cue
        if y[i] == 0:
            X[i, :, :, 100:127] += 1.0
        else:
            X[i, :, 100:127, :] += 1.0

    start = net.loss(X, y)
    net.fit(X, y, steps=12, lr=0.01)
    final = net.loss(X, y)
    acc = np.mean(net.predict(X) == y)
    print("Start loss         : {:.4f}".format(float(start)))
    print("Final loss         : {:.4f}".format(float(final)))
    print("Loss decreased 5x+ :", bool(final < start / 5))
    print("Train accuracy     : {:.2f}".format(float(acc)))
