import numpy as np


class VAE:
    """Variational Autoencoder with a single hidden layer per half.

    Encoder maps x -> (mu, logvar); we sample z = mu + sigma * eps
    (the reparameterization trick) and the decoder reconstructs x.
    Loss = Gaussian reconstruction + KL(q(z|x) || N(0, I)).
    Forward, backward and Adam are all hand-rolled.
    """

    def __init__(self, in_dim, hidden, latent, lr=0.01, seed=0):
        self.rng = np.random.RandomState(seed)
        self.lr = lr
        r = self.rng.randn
        s = np.sqrt
        # Encoder: x -> h1 -> (mu, logvar).
        self.P = {
            "W1": r(in_dim, hidden) * s(1.0 / in_dim), "b1": np.zeros(hidden),
            "Wm": r(hidden, latent) * s(1.0 / hidden), "bm": np.zeros(latent),
            "Wv": r(hidden, latent) * s(1.0 / hidden), "bv": np.zeros(latent),
            # Decoder: z -> h2 -> xhat.
            "W2": r(latent, hidden) * s(1.0 / latent), "b2": np.zeros(hidden),
            "Wo": r(hidden, in_dim) * s(1.0 / hidden), "bo": np.zeros(in_dim),
        }
        # Adam state.
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0

    def encode(self, X):
        h1 = np.tanh(X @ self.P["W1"] + self.P["b1"])
        mu = h1 @ self.P["Wm"] + self.P["bm"]
        logvar = h1 @ self.P["Wv"] + self.P["bv"]
        return h1, mu, logvar

    def decode(self, z):
        h2 = np.tanh(z @ self.P["W2"] + self.P["b2"])
        xhat = h2 @ self.P["Wo"] + self.P["bo"]
        return h2, xhat

    def forward(self, X):
        # Full pass; caches everything backward() needs.
        h1, mu, logvar = self.encode(X)
        eps = self.rng.randn(*mu.shape)
        std = np.exp(0.5 * logvar)
        z = mu + std * eps                       # reparameterization trick
        h2, xhat = self.decode(z)
        self.cache = (X, h1, mu, logvar, eps, std, z, h2, xhat)
        return xhat

    def loss(self, X):
        _, _, mu, logvar, _, _, _, _, xhat = self.cache
        m = X.shape[0]
        recon = 0.5 * np.sum((xhat - X) ** 2) / m          # Gaussian NLL
        kl = -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar)) / m
        return recon + kl, recon, kl

    def backward(self):
        X, h1, mu, logvar, eps, std, z, h2, xhat = self.cache
        m = X.shape[0]
        P, g = self.P, {}
        # --- decoder (reconstruction term) ---
        dxhat = (xhat - X) / m
        g["Wo"] = h2.T @ dxhat
        g["bo"] = dxhat.sum(0)
        dh2 = (dxhat @ P["Wo"].T) * (1 - h2 ** 2)          # through tanh
        g["W2"] = z.T @ dh2
        g["b2"] = dh2.sum(0)
        dz = dh2 @ P["W2"].T                               # dLoss_recon / dz
        # --- reparameterization: z = mu + std * eps ---
        dmu = dz.copy()
        dlogvar = dz * eps * std * 0.5                     # dstd/dlogvar = 0.5*std
        # --- add KL-divergence gradients ---
        dmu += mu / m
        dlogvar += 0.5 * (np.exp(logvar) - 1) / m
        # --- encoder ---
        g["Wm"] = h1.T @ dmu
        g["bm"] = dmu.sum(0)
        g["Wv"] = h1.T @ dlogvar
        g["bv"] = dlogvar.sum(0)
        dh1 = (dmu @ P["Wm"].T + dlogvar @ P["Wv"].T) * (1 - h1 ** 2)
        g["W1"] = X.T @ dh1
        g["b1"] = dh1.sum(0)
        return g

    def _adam(self, g, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self.P:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.P[k] -= self.lr * mh / (np.sqrt(vh) + eps)

    def fit(self, X, epochs=2000):
        history = []
        for _ in range(epochs):
            self.forward(X)
            history.append(self.loss(X))
            self._adam(self.backward())
        return history

    def predict(self, X):
        # Deterministic reconstruction: use the mean code, no sampling.
        _, mu, _ = self.encode(X)
        return self.decode(mu)[1]

    def sample(self, n):
        # Generate fresh data by decoding samples from the N(0, I) prior.
        return self.decode(self.rng.randn(n, self.P["Wm"].shape[1]))[1]


if __name__ == "__main__":
    np.random.seed(0)

    # 8-dim data intrinsically living on a 2D manifold, then standardized.
    n = 256
    latent = np.random.randn(n, 2)
    mixing = np.random.randn(2, 8)
    X = latent @ mixing
    X = (X - X.mean(0)) / X.std(0)

    vae = VAE(in_dim=8, hidden=16, latent=2, lr=0.01)
    hist = vae.fit(X, epochs=2000)

    (l0, r0, k0), (lN, rN, kN) = hist[0], hist[-1]
    print("Start loss: {:.4f}  (recon {:.4f} + KL {:.4f})".format(l0, r0, k0))
    print("Final loss: {:.4f}  (recon {:.4f} + KL {:.4f})".format(lN, rN, kN))
    print("Loss reduction: {:.1f}x".format(l0 / lN))

    recon = vae.predict(X)
    print("Reconstruction MSE: {:.4f}  (data variance {:.4f})".format(
        np.mean((recon - X) ** 2), np.var(X)))

    # Samples from the prior should roughly match the real data statistics.
    gen = vae.sample(1000)
    print("Real mean/std:  {:+.3f} / {:.3f}".format(X.mean(), X.std()))
    print("Gen  mean/std:  {:+.3f} / {:.3f}".format(gen.mean(), gen.std()))
