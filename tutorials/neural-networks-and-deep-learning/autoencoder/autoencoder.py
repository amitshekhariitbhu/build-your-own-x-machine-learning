import numpy as np


class Autoencoder:
    """Fully-connected autoencoder (encoder -> bottleneck -> decoder).

    Trained by manual forward/backprop to reconstruct its own input. The
    middle layer is a narrow bottleneck, forcing a compressed code.
    """

    def __init__(self, dims, lr=0.01, seed=0):
        # dims is symmetric, e.g. [8, 5, 2, 5, 8]: the 2 is the bottleneck.
        rng = np.random.RandomState(seed)
        self.dims = dims
        self.lr = lr
        self.n_layers = len(dims) - 1
        self.mid = self.n_layers // 2  # encoder ends here (bottleneck)
        # Weights + biases; scale init for tanh units.
        self.W = [rng.randn(dims[i], dims[i + 1]) * np.sqrt(1.0 / dims[i])
                  for i in range(self.n_layers)]
        self.b = [np.zeros(dims[i + 1]) for i in range(self.n_layers)]
        # Adam optimizer state.
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, X):
        # Cache layer activations; tanh on hidden layers, linear on output.
        self.a = [X]
        h = X
        for i in range(self.n_layers):
            z = h @ self.W[i] + self.b[i]
            h = z if i == self.n_layers - 1 else np.tanh(z)
            self.a.append(h)
        return h

    def encode(self, X):
        # Run the encoder half only, returning the bottleneck code.
        h = X
        for i in range(self.mid):
            h = np.tanh(h @ self.W[i] + self.b[i])
        return h

    def backward(self, recon, X):
        # MSE loss gradient; output layer is linear so delta starts here.
        m = X.shape[0]
        delta = 2.0 * (recon - X) / m  # dL/dz at the output layer
        gW, gb = [None] * self.n_layers, [None] * self.n_layers
        for i in reversed(range(self.n_layers)):
            gW[i] = self.a[i].T @ delta
            gb[i] = delta.sum(axis=0)
            dA = delta @ self.W[i].T  # dL/d(activation of layer i)
            if i > 0:
                # Backprop through tanh: derivative is 1 - tanh^2.
                delta = dA * (1 - self.a[i] ** 2)
        return gW, gb

    def _adam(self, gW, gb, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for i in range(self.n_layers):
            for p, g, mm, vv in ((self.W, gW, self.mW, self.vW),
                                 (self.b, gb, self.mb, self.vb)):
                mm[i] = b1 * mm[i] + (1 - b1) * g[i]
                vv[i] = b2 * vv[i] + (1 - b2) * g[i] ** 2
                m_hat = mm[i] / (1 - b1 ** self.t)
                v_hat = vv[i] / (1 - b2 ** self.t)
                p[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(self, X, epochs=1500):
        losses = []
        for _ in range(epochs):
            recon = self.forward(X)
            losses.append(np.mean((recon - X) ** 2))
            gW, gb = self.backward(recon, X)
            self._adam(gW, gb)
        return losses

    def predict(self, X):
        # Reconstruct inputs (encode then decode).
        return self.forward(X)


if __name__ == "__main__":
    np.random.seed(0)

    # Synthetic 8-dim data that intrinsically lives on a 2D manifold:
    # a 2-dim autoencoder bottleneck can therefore reconstruct it well.
    n = 128
    latent = np.random.randn(n, 2)
    mixing = np.random.randn(2, 8)
    X = latent @ mixing
    X = (X - X.mean(0)) / X.std(0)  # standardize columns

    ae = Autoencoder(dims=[8, 5, 2, 5, 8], lr=0.02)
    losses = ae.fit(X, epochs=1500)

    recon = ae.predict(X)
    code = ae.encode(X)
    final_mse = np.mean((recon - X) ** 2)
    var = np.var(X)

    print("Input dim: {}  ->  bottleneck dim: {}".format(X.shape[1], code.shape[1]))
    print("Start reconstruction MSE: {:.6f}".format(losses[0]))
    print("Final reconstruction MSE: {:.6f}".format(final_mse))
    print("Reduction factor: {:.1f}x".format(losses[0] / final_mse))
    print("Variance of data: {:.4f}  (MSE far below this = good)".format(var))
    print("Fraction of variance unexplained: {:.4f}".format(final_mse / var))
