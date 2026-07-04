import numpy as np


def sigmoid(z):
    # Numerically-stable sigmoid, mapped to (0, 1)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


class MLP:
    """Small MLP: ReLU hidden layers, linear output, trained with Adam.

    forward() caches activations; backward() takes the gradient on the output
    and returns the gradient on the *input*, so gradients can flow G <- D.
    """

    def __init__(self, dims, lr, seed):
        rng = np.random.RandomState(seed)
        self.n = len(dims) - 1
        self.lr = lr
        self.W = [rng.randn(dims[i], dims[i + 1]) * np.sqrt(2.0 / dims[i])
                  for i in range(self.n)]
        self.b = [np.zeros(dims[i + 1]) for i in range(self.n)]
        # Adam moments.
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, X):
        self.a = [X]
        h = X
        for i in range(self.n):
            z = h @ self.W[i] + self.b[i]
            h = z if i == self.n - 1 else np.maximum(0, z)  # relu hidden, linear out
            self.a.append(h)
        return h

    def backward(self, grad):
        # grad = dL/d(output); stashes param grads, returns dL/d(input).
        self.gW, self.gb = [None] * self.n, [None] * self.n
        for i in reversed(range(self.n)):
            self.gW[i] = self.a[i].T @ grad
            self.gb[i] = grad.sum(axis=0)
            grad = grad @ self.W[i].T
            if i > 0:
                grad = grad * (self.a[i] > 0)  # backprop through relu
        return grad

    def step(self, b1=0.9, b2=0.999, eps=1e-8):
        # Adam update using the grads stashed by the last backward().
        self.t += 1
        for i in range(self.n):
            for p, g, mm, vv in ((self.W, self.gW, self.mW, self.vW),
                                 (self.b, self.gb, self.mb, self.vb)):
                mm[i] = b1 * mm[i] + (1 - b1) * g[i]
                vv[i] = b2 * vv[i] + (1 - b2) * g[i] ** 2
                m_hat = mm[i] / (1 - b1 ** self.t)
                v_hat = vv[i] / (1 - b2 ** self.t)
                p[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)


class GAN:
    """Minimal GAN: an MLP generator vs an MLP discriminator.

    D scores real-vs-fake (sigmoid on a linear logit); G maps latent noise to
    data and learns to fool D with the non-saturating loss -log D(G(z)). Both
    nets are trained by manual backprop; D's backward also hands G the gradient
    of D's output w.r.t. the fake samples. As in the original GAN algorithm we
    take k discriminator updates per generator update to keep D near-optimal,
    which lets a plain GAN match the target's mean *and* variance.
    """

    def __init__(self, data_dim=2, latent_dim=2, g_hidden=16, d_hidden=64,
                 lr=0.002, k=5, seed=0):
        self.latent_dim = latent_dim
        self.k = k
        self.G = MLP([latent_dim, g_hidden, data_dim], lr, seed)
        self.D = MLP([data_dim, d_hidden, 1], lr, seed + 1)

    def latent(self, n):
        return np.random.randn(n, self.latent_dim)

    def generate(self, n):
        return self.G.forward(self.latent(n))

    def _d_step(self, real):
        # Train D: real -> 1, fake -> 0, in one combined batch.
        m = real.shape[0]
        fake = self.G.forward(self.latent(m))
        X = np.vstack([real, fake])
        y = np.vstack([np.ones((m, 1)), np.zeros((m, 1))])
        p = sigmoid(self.D.forward(X))
        # d(BCE + sigmoid)/d(logit) = (p - y), averaged over the batch.
        self.D.backward((p - y) / (2 * m))
        self.D.step()
        eps = 1e-9
        return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    def _g_step(self, m):
        # Train G: push D's score on fakes toward 1 (non-saturating loss).
        fake = self.G.forward(self.latent(m))
        p = sigmoid(self.D.forward(fake))
        d_logit = (p - 1.0) / m            # grad of -log D(G(z)) w.r.t. logit
        d_fake = self.D.backward(d_logit)  # grad w.r.t. the fake samples
        self.G.backward(d_fake)            # ...continue into G (D not updated)
        self.G.step()
        return -np.mean(np.log(p + 1e-9))

    def fit(self, real, steps=1000, batch=256):
        hist = []
        for _ in range(steps):
            for _ in range(self.k):
                idx = np.random.randint(0, real.shape[0], batch)
                loss_d = self._d_step(real[idx])
            loss_g = self._g_step(batch)
            hist.append((loss_d, loss_g))
        return np.array(hist)


if __name__ == "__main__":
    np.random.seed(0)

    # Real data: a 2D Gaussian (distinct per-axis mean and std) the GAN learns.
    real_mean = np.array([3.0, -1.0])
    real_std = np.array([1.5, 0.7])
    real = np.random.randn(3000, 2) * real_std + real_mean

    def stats(x):
        return x.mean(0), x.std(0)

    gan = GAN(data_dim=2, latent_dim=2, g_hidden=16, d_hidden=64, lr=0.002, k=5)

    # Generator statistics before any training (baseline for the metric).
    m0, s0 = stats(gan.generate(4000))
    err0 = np.abs(m0 - real_mean).mean() + np.abs(s0 - real_std).mean()

    hist = gan.fit(real, steps=1000, batch=256)

    m1, s1 = stats(gan.generate(4000))
    err1 = np.abs(m1 - real_mean).mean() + np.abs(s1 - real_std).mean()

    print("Real  mean: [{:.2f} {:.2f}]  std: [{:.2f} {:.2f}]".format(*real_mean, *real_std))
    print("Init  mean: [{:.2f} {:.2f}]  std: [{:.2f} {:.2f}]".format(*m0, *s0))
    print("Final mean: [{:.2f} {:.2f}]  std: [{:.2f} {:.2f}]".format(*m1, *s1))
    print("Distribution error  start: {:.3f}  ->  final: {:.3f}".format(err0, err1))
    print("Error reduction: {:.1f}x".format(err0 / err1))
    print("D loss  start: {:.3f}  ->  final: {:.3f}  (~0.69 = D can't tell real from fake)".format(
        hist[0, 0], hist[-50:, 0].mean()))
    print("G loss  start: {:.3f}  ->  final: {:.3f}".format(hist[0, 1], hist[-50:, 1].mean()))
    print("PASS" if err1 < 0.2 and err1 < err0 / 5 else "FAIL")
