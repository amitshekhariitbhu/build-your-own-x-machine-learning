import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    """Multilayer perceptron: dense layers, ReLU hidden units, softmax output.
    Trained by manual backprop with mini-batch gradient descent."""

    def __init__(self, sizes, lr=0.1, seed=0):
        # sizes = [n_in, h1, h2, ..., n_out]
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W = [rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, X):
        # cache pre/post activations for backprop; ReLU on hidden, softmax on last
        self.a = [X]
        self.z = []
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            self.z.append(z)
            h = softmax(z) if i == len(self.W) - 1 else relu(z)
            self.a.append(h)
        return h

    def loss(self, X, Y):
        # cross-entropy; Y is one-hot
        P = self.forward(X)
        return -np.mean(np.sum(Y * np.log(P + 1e-9), axis=1))

    def backward(self, Y):
        n = Y.shape[0]
        # dL/dz for softmax + cross-entropy
        dz = (self.a[-1] - Y) / n
        for i in reversed(range(len(self.W))):
            dW = self.a[i].T @ dz
            db = dz.sum(axis=0)
            if i > 0:  # propagate through ReLU of previous layer
                dz = (dz @ self.W[i].T) * (self.z[i - 1] > 0)
            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * db

    def fit(self, X, Y, epochs=200, batch=32, verbose=False):
        n = X.shape[0]
        for ep in range(epochs):
            idx = np.random.permutation(n)
            for s in range(0, n, batch):
                b = idx[s:s + batch]
                self.forward(X[b])
                self.backward(Y[b])
            if verbose and ep % 50 == 0:
                print(f"epoch {ep:3d}  loss {self.loss(X, Y):.4f}")

    def predict(self, X):
        return self.forward(X).argmax(axis=1)


if __name__ == "__main__":
    np.random.seed(0)
    rng = np.random.default_rng(0)

    # tiny synthetic 3-class problem: gaussian blobs around 3 centers
    K, per, d = 3, 80, 2
    centers = np.array([[0, 0], [3, 3], [-3, 3]], dtype=float)
    X = np.vstack([rng.standard_normal((per, d)) * 0.7 + c for c in centers])
    y = np.repeat(np.arange(K), per)
    Y = np.eye(K)[y]

    net = MLP([d, 16, 16, K], lr=0.2, seed=0)
    start = net.loss(X, Y)
    net.fit(X, Y, epochs=200, batch=32, verbose=True)
    final = net.loss(X, Y)
    acc = (net.predict(X) == y).mean()

    print(f"start loss {start:.4f} -> final loss {final:.4f}")
    print(f"train accuracy {acc:.3f}")
