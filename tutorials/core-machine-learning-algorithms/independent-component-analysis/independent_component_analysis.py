import numpy as np


class FastICA:
    """FastICA: recover independent sources via fixed-point iteration (tanh)."""

    def __init__(self, n_components, max_iter=200, tol=1e-5):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _sym_decorrelate(W):
        # W <- (W W^T)^{-1/2} W, so the rows become orthonormal.
        d, E = np.linalg.eigh(W @ W.T)
        return (E / np.sqrt(d)) @ E.T @ W

    def _whiten(self, X):
        # Center each feature to zero mean.
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # PCA whitening: cov^{-1/2} decorrelates and scales to unit variance.
        cov = np.cov(Xc, rowvar=False)
        d, E = np.linalg.eigh(cov)
        self.whiten_ = (E / np.sqrt(d)) @ E.T
        return Xc @ self.whiten_

    def fit(self, X):
        n = self.n_components
        Z = self._whiten(X).T                     # (n, samples), signals in rows

        # Random unmixing matrix, symmetrically decorrelated.
        W = self._sym_decorrelate(np.random.randn(n, n))

        for _ in range(self.max_iter):
            g = np.tanh(W @ Z)                    # nonlinearity g
            g_prime = 1.0 - g ** 2                # derivative g'
            # Fixed-point update: E[Z g(WZ)] - E[g'] W, then decorrelate.
            W_new = (g @ Z.T) / Z.shape[1] - g_prime.mean(axis=1)[:, None] * W
            W_new = self._sym_decorrelate(W_new)

            # Converged when the directions stop rotating.
            delta = np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1))
            W = W_new
            if delta < self.tol:
                break

        self.components_ = W
        return self

    def transform(self, X):
        # Unmix into recovered sources (columns).
        Xw = (X - self.mean_) @ self.whiten_
        return (self.components_ @ Xw.T).T

    def fit_transform(self, X):
        return self.fit(X).transform(X)


if __name__ == "__main__":
    np.random.seed(0)

    # Two independent source signals: a sine and a sign/square wave.
    t = np.linspace(0, 8, 2000)
    s1 = np.sin(2 * t)
    s2 = np.sign(np.sin(3 * t))
    S = np.c_[s1, s2] + 0.02 * np.random.randn(2000, 2)
    S -= S.mean(axis=0)

    # Mix the sources with a 2x2 mixing matrix.
    A = np.array([[1.0, 1.0], [0.5, 2.0]])
    X = S @ A.T

    ica = FastICA(n_components=2)
    S_est = ica.fit_transform(X)

    # Match each recovered signal to its best-correlated true source.
    corr = np.abs(np.corrcoef(S_est.T, S.T)[:2, 2:])   # 2x2 recovered-vs-true
    best = np.max(corr, axis=1)
    print("Mixing matrix A:\n", A)
    print("Recovered vs true |correlation| matrix:\n", np.round(corr, 3))
    print("Max |correlation| per recovered signal:", np.round(best, 4))
    print("Mean max |correlation| (near 1.0 = success): {:.4f}".format(best.mean()))
