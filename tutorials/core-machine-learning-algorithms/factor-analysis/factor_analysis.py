import numpy as np


class FactorAnalysis:
    """Factor Analysis via EM: x = W z + mu + noise, z ~ N(0, I), noise ~ N(0, diag(Psi))."""

    def __init__(self, n_components, n_iter=100, tol=1e-4):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol

    def _log_likelihood(self, Xc, W, psi):
        # Marginal: x ~ N(mu, C) with C = W W^T + diag(psi).
        n, d = Xc.shape
        C = W @ W.T + np.diag(psi)
        sign, logdet = np.linalg.slogdet(C)
        Cinv = np.linalg.inv(C)
        S = (Xc.T @ Xc) / n  # sample covariance
        return -0.5 * n * (d * np.log(2 * np.pi) + logdet + np.trace(Cinv @ S))

    def fit(self, X):
        n, d = X.shape
        k = self.n_components
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # Initialise loadings randomly, noise from feature variances.
        W = np.random.randn(d, k) * 0.1
        psi = Xc.var(axis=0) + 1e-6
        I = np.eye(k)

        self.loglik_ = []
        for _ in range(self.n_iter):
            # E-step: Gaussian posterior of z given x.
            Wt_Pinv = W.T / psi                      # (k, d)
            G = np.linalg.inv(I + Wt_Pinv @ W)       # posterior covariance (k, k)
            Ez = Xc @ (G @ Wt_Pinv).T                # posterior means (n, k)
            Ezz = n * G + Ez.T @ Ez                  # summed E[z z^T] (k, k)

            # M-step: update loadings and diagonal noise.
            W = (Xc.T @ Ez) @ np.linalg.inv(Ezz)
            psi = (np.sum(Xc ** 2, axis=0) - np.sum(W * (Xc.T @ Ez), axis=1)) / n
            psi = np.maximum(psi, 1e-6)              # keep noise positive

            self.loglik_.append(self._log_likelihood(Xc, W, psi))
            if len(self.loglik_) > 1 and abs(self.loglik_[-1] - self.loglik_[-2]) < self.tol:
                break

        self.components_ = W
        self.noise_variance_ = psi
        return self

    def transform(self, X):
        # Posterior mean of the latent factors for new data.
        Wt_Pinv = self.components_.T / self.noise_variance_
        G = np.linalg.inv(np.eye(self.n_components) + Wt_Pinv @ self.components_)
        return (X - self.mean_) @ (G @ Wt_Pinv).T

    def fit_transform(self, X):
        return self.fit(X).transform(X)


if __name__ == "__main__":
    np.random.seed(0)

    # Low-rank signal (2 factors in 6-D) plus per-feature diagonal noise.
    n, d, k = 500, 6, 2
    W_true = np.random.randn(d, k)
    Z = np.random.randn(n, k)
    mu_true = np.array([1.0, -2.0, 0.5, 0.0, 3.0, -1.0])
    noise = np.random.randn(n, d) * np.array([0.3, 0.5, 0.2, 0.4, 0.3, 0.6])
    X = Z @ W_true.T + mu_true + noise

    fa = FactorAnalysis(n_components=k, n_iter=200).fit(X)

    print("Recovered loading matrix shape:", fa.components_.shape)
    ll = np.array(fa.loglik_)
    print("Log-likelihood (first 5): ", np.round(ll[:5], 2))
    print("Log-likelihood (last 5):  ", np.round(ll[-5:], 2))
    print("Log-likelihood increased monotonically:", bool(np.all(np.diff(ll) > -1e-6)))

    # Correctness: does W W^T + Psi reconstruct the sample covariance?
    S = np.cov(X - X.mean(axis=0), rowvar=False)
    C = fa.components_ @ fa.components_.T + np.diag(fa.noise_variance_)
    rel_err = np.linalg.norm(S - C) / np.linalg.norm(S)
    print("Covariance reconstruction relative error: {:.4f}".format(rel_err))
