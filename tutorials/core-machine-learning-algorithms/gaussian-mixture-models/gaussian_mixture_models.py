import numpy as np


class GaussianMixture:
    """Gaussian Mixture Model fit with Expectation-Maximization."""

    def __init__(self, n_components, max_iter=100, tol=1e-6, reg=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg  # ridge added to covariance diagonal for stability

    def _log_gaussian(self, X, mean, cov):
        # Log pdf of a multivariate Gaussian evaluated at every row of X.
        d = X.shape[1]
        diff = X - mean
        cov = cov + self.reg * np.eye(d)
        inv = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        maha = np.sum((diff @ inv) * diff, axis=1)  # Mahalanobis distance squared
        return -0.5 * (d * np.log(2 * np.pi) + logdet + maha)

    def _log_prob(self, X):
        # Weighted log responsibilities, shape (n_samples, n_components).
        logp = np.array([self._log_gaussian(X, self.means_[k], self.covs_[k])
                         for k in range(self.n_components)]).T
        return logp + np.log(self.weights_)

    def _init_means(self, X):
        # k-means++ style seeding: spread initial means to avoid bad local optima.
        n = X.shape[0]
        means = [X[np.random.randint(n)]]
        for _ in range(1, self.n_components):
            d2 = np.min([np.sum((X - m) ** 2, axis=1) for m in means], axis=0)
            probs = d2 / d2.sum()
            means.append(X[np.random.choice(n, p=probs)])
        return np.array(means, dtype=float)

    def fit(self, X):
        n, d = X.shape
        # Init: spread means, identity covariances, uniform weights.
        self.means_ = self._init_means(X)
        self.covs_ = np.array([np.eye(d) for _ in range(self.n_components)])
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)

        self.log_likelihoods_ = []
        prev_ll = -np.inf
        for _ in range(self.max_iter):
            # E-step: responsibilities via log-sum-exp for numerical stability.
            weighted = self._log_prob(X)
            m = weighted.max(axis=1, keepdims=True)
            log_norm = m[:, 0] + np.log(np.exp(weighted - m).sum(axis=1))
            resp = np.exp(weighted - log_norm[:, None])

            ll = log_norm.sum()
            self.log_likelihoods_.append(ll)

            # M-step: update weights, means, covariances.
            Nk = resp.sum(axis=0)
            self.weights_ = Nk / n
            self.means_ = (resp.T @ X) / Nk[:, None]
            for k in range(self.n_components):
                diff = X - self.means_[k]
                self.covs_[k] = (resp[:, k, None] * diff).T @ diff / Nk[k]

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return self

    def predict_proba(self, X):
        weighted = self._log_prob(X)
        m = weighted.max(axis=1, keepdims=True)
        log_norm = m + np.log(np.exp(weighted - m).sum(axis=1, keepdims=True))
        return np.exp(weighted - log_norm)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


if __name__ == "__main__":
    np.random.seed(0)

    # 3 Gaussian blobs with different means.
    true_means = np.array([[0.0, 0.0], [6.0, 6.0], [0.0, 7.0]])
    X = np.vstack([m + np.random.randn(60, 2) for m in true_means])
    y = np.repeat(np.arange(3), 60)

    gmm = GaussianMixture(n_components=3).fit(X)

    print("Converged means:")
    print(np.round(gmm.means_, 3))
    print("Mixing weights:", np.round(gmm.weights_, 3))

    lls = gmm.log_likelihoods_
    print("Log-likelihood: {:.2f} -> {:.2f} over {} iters".format(lls[0], lls[-1], len(lls)))
    print("Log-likelihood monotonically increasing:", all(np.diff(lls) >= -1e-6))

    # Correctness signal: clustering accuracy (labels are permutation-invariant).
    pred = gmm.predict(X)
    acc = max(
        np.mean(np.array(perm)[pred] == y)
        for perm in __import__("itertools").permutations(range(3))
    )
    print("Clustering accuracy (best label matching): {:.3f}".format(acc))
