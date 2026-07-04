import numpy as np


class DiffusionModel:
    """Minimal DDPM diffusion model on low-dim (2D) data.

    A linear variance schedule (betas) defines the forward noising process
    x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps. A small MLP
    eps_theta(x_t, t), trained by manual backprop to predict the added noise,
    then drives the reverse loop that denoises Gaussian samples back into data.
    """

    def __init__(self, dim=2, T=50, hidden=64, n_freq=4, lr=0.002, seed=0):
        rng = np.random.RandomState(seed)
        self.rng = rng
        self.dim = dim
        self.T = T
        self.lr = lr
        self.n_freq = n_freq
        # Linear beta schedule; max-beta large enough to fully noise data by t=T.
        self.beta = np.linspace(1e-4, 0.2, T)
        self.alpha = 1.0 - self.beta
        self.abar = np.cumprod(self.alpha)              # alpha_bar_t
        # Noise-predictor MLP: [x, time_emb] -> hidden -> hidden -> eps, tanh hidden.
        dims = [dim + 2 * n_freq, hidden, hidden, dim]
        self.dims = dims
        self.nL = len(dims) - 1
        self.W = [rng.randn(dims[i], dims[i + 1]) * np.sqrt(1.0 / dims[i])
                  for i in range(self.nL)]
        self.b = [np.zeros(dims[i + 1]) for i in range(self.nL)]
        # Adam state.
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(x) for x in self.b]
        self.vb = [np.zeros_like(x) for x in self.b]
        self.t_adam = 0

    def time_embed(self, t):
        # Sinusoidal features of the normalized timestep -> (m, 2*n_freq).
        tt = (t[:, None] / self.T) * (2.0 ** np.arange(self.n_freq)) * np.pi
        return np.concatenate([np.sin(tt), np.cos(tt)], axis=1)

    def forward(self, x, t):
        # Predict noise eps for noised inputs x at timesteps t; cache activations.
        self.a = [np.concatenate([x, self.time_embed(t)], axis=1)]
        h = self.a[0]
        for i in range(self.nL):
            z = h @ self.W[i] + self.b[i]
            h = z if i == self.nL - 1 else np.tanh(z)   # linear output head
            self.a.append(h)
        return h

    def backward(self, pred, target):
        # Gradient of MSE(pred, target); output layer is linear.
        m = target.shape[0]
        delta = 2.0 * (pred - target) / m
        gW, gb = [None] * self.nL, [None] * self.nL
        for i in reversed(range(self.nL)):
            gW[i] = self.a[i].T @ delta
            gb[i] = delta.sum(0)
            dA = delta @ self.W[i].T
            if i > 0:
                delta = dA * (1 - self.a[i] ** 2)       # backprop through tanh
        return gW, gb

    def _adam(self, gW, gb, b1=0.9, b2=0.999, eps=1e-8):
        self.t_adam += 1
        for i in range(self.nL):
            for p, g, mm, vv in ((self.W, gW, self.mW, self.vW),
                                 (self.b, gb, self.mb, self.vb)):
                mm[i] = b1 * mm[i] + (1 - b1) * g[i]
                vv[i] = b2 * vv[i] + (1 - b2) * g[i] ** 2
                m_hat = mm[i] / (1 - b1 ** self.t_adam)
                v_hat = vv[i] / (1 - b2 ** self.t_adam)
                p[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def train(self, X, epochs=3000, batch=256):
        n = X.shape[0]
        losses = []
        for _ in range(epochs):
            idx = self.rng.randint(0, n, batch)
            t = self.rng.randint(0, self.T, batch)
            noise = self.rng.randn(batch, self.dim)
            ab = self.abar[t][:, None]
            # Forward noising in closed form, then predict the noise back.
            xt = np.sqrt(ab) * X[idx] + np.sqrt(1 - ab) * noise
            pred = self.forward(xt, t)
            losses.append(np.mean((pred - noise) ** 2))
            self._adam(*self.backward(pred, noise))
        return losses

    def sample(self, n):
        # Reverse diffusion: denoise pure Gaussian noise step by step (DDPM).
        x = self.rng.randn(n, self.dim)
        for t in reversed(range(self.T)):
            eps = self.forward(x, np.full(n, t))
            a, ab, beta = self.alpha[t], self.abar[t], self.beta[t]
            mean = (x - beta / np.sqrt(1 - ab) * eps) / np.sqrt(a)
            noise = self.rng.randn(n, self.dim) if t > 0 else 0.0
            x = mean + np.sqrt(beta) * noise
        return x


if __name__ == "__main__":
    np.random.seed(0)

    # Real data: a noisy unit ring in 2D (mean ~ 0, radius ~ 1).
    n = 1024
    theta = np.random.uniform(0, 2 * np.pi, n)
    r = 1.0 + 0.05 * np.random.randn(n)
    X = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    model = DiffusionModel(dim=2, T=50, hidden=64, lr=0.002)
    losses = model.train(X, epochs=3000, batch=256)

    start = np.mean(losses[:5])             # near-untrained loss (~E[eps^2]=1)
    final = np.mean(losses[-50:])           # smooth the stochastic tail
    samples = model.sample(1000)

    real_r = np.sqrt((X ** 2).sum(1))
    gen_r = np.sqrt((samples ** 2).sum(1))

    print("Start loss (avg first 5):  {:.4f}".format(start))
    print("Final loss (avg last 50):  {:.4f}".format(final))
    print("Loss reduction factor:     {:.1f}x".format(start / final))
    print("Real  mean/std : {} / {}".format(np.round(X.mean(0), 3), np.round(X.std(0), 3)))
    print("Gen   mean/std : {} / {}".format(np.round(samples.mean(0), 3), np.round(samples.std(0), 3)))
    print("Real  radius mean/std: {:.3f} / {:.3f}".format(real_r.mean(), real_r.std()))
    print("Gen   radius mean/std: {:.3f} / {:.3f}".format(gen_r.mean(), gen_r.std()))
    print("Radius-mean abs error: {:.3f}  (small = samples match the ring)".format(
        abs(gen_r.mean() - real_r.mean())))
