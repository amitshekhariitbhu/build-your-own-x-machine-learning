import numpy as np


def _pairwise_dist(X):
    """Euclidean distance matrix."""
    sq = np.sum(X ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    return np.sqrt(np.maximum(d2, 0.0))


def _smooth_knn(knn_d, k, iters=64):
    """Per-point rho (nearest-neighbor dist) and sigma solving sum exp(-(d-rho)/sigma)=log2(k)."""
    target = np.log2(k)
    n = knn_d.shape[0]
    rho = knn_d[:, 0].copy()
    sigma = np.ones(n)
    for i in range(n):
        lo, hi, mid = 0.0, np.inf, 1.0
        d = np.maximum(knn_d[i] - rho[i], 0.0)
        for _ in range(iters):
            # Binary-search sigma so the fuzzy memberships sum to log2(k).
            psum = np.sum(np.exp(-d / mid))
            if psum > target:
                hi, mid = mid, (lo + mid) / 2
            else:
                lo = mid
                mid = mid * 2 if hi == np.inf else (lo + hi) / 2
        sigma[i] = mid
    return rho, sigma


def _fit_ab(min_dist=0.1, spread=1.0):
    """Fit a, b so 1/(1+a*x^(2b)) matches UMAP's target membership curve."""
    x = np.linspace(0, 3 * spread, 300)
    y = np.where(x < min_dist, 1.0, np.exp(-(x - min_dist) / spread))
    best, err = (1.577, 0.895), np.inf
    for a in np.linspace(0.1, 3.0, 80):
        for b in np.linspace(0.3, 1.5, 80):
            f = 1.0 / (1.0 + a * np.power(x, 2 * b))
            e = np.sum((f - y) ** 2)
            if e < err:
                err, best = e, (a, b)
    return best


class UMAP:
    """UMAP: fuzzy kNN graph in high-D, cross-entropy layout in low-D via SGD."""

    def __init__(self, n_components=2, n_neighbors=15, n_epochs=250,
                 lr=1.0, n_neg=5, min_dist=0.1):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.n_epochs = n_epochs
        self.lr = lr
        self.n_neg = n_neg
        self.min_dist = min_dist

    def _fuzzy_graph(self, X):
        # k-NN graph (self excluded) with smooth fuzzy membership strengths.
        D = _pairwise_dist(X)
        k = self.n_neighbors
        knn_idx = np.argsort(D, axis=1)[:, 1:k + 1]
        knn_d = np.take_along_axis(D, knn_idx, axis=1)
        rho, sigma = _smooth_knn(knn_d, k)
        P = np.zeros_like(D)
        vals = np.exp(-np.maximum(knn_d - rho[:, None], 0.0) / sigma[:, None])
        np.put_along_axis(P, knn_idx, vals, axis=1)
        # Symmetrize via probabilistic t-conorm: p + p^T - p*p^T.
        return P + P.T - P * P.T

    def _spectral_init(self, G):
        # Initialize from eigenvectors of the normalized graph Laplacian.
        deg = G.sum(axis=1)
        di = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        L = np.eye(len(G)) - di[:, None] * G * di[None, :]
        _, vecs = np.linalg.eigh(L)
        Y = vecs[:, 1:1 + self.n_components].copy()
        Y = Y / (Y.std(axis=0) + 1e-12) * 10.0
        return Y + 1e-4 * np.random.randn(*Y.shape)

    def fit_transform(self, X):
        a, b = _fit_ab(self.min_dist)
        self.a_, self.b_ = a, b
        G = self._fuzzy_graph(X)
        Y = self._spectral_init(G)
        n = len(X)

        # Edge list (upper triangle) with normalized sampling weights.
        i, j = np.where(np.triu(G, 1) > 1e-3)
        w = G[i, j] / G[i, j].max()

        for ep in range(self.n_epochs):
            alpha = self.lr * (1.0 - ep / self.n_epochs)
            m = np.random.rand(len(i)) < w          # sample edges by weight
            h, t = i[m], j[m]

            # Attraction along sampled edges.
            diff = Y[h] - Y[t]
            d2 = np.maximum(np.sum(diff ** 2, axis=1), 1e-10)
            co = (-2 * a * b * np.power(d2, b - 1)) / (1 + a * np.power(d2, b))
            g = np.clip(co[:, None] * diff, -4, 4) * alpha
            np.add.at(Y, h, g)
            np.add.at(Y, t, -g)

            # Repulsion via random negative samples.
            reps = np.repeat(h, self.n_neg)
            negs = np.random.randint(0, n, size=len(reps))
            diffr = Y[reps] - Y[negs]
            d2r = np.maximum(np.sum(diffr ** 2, axis=1), 1e-10)
            cor = (2 * b) / ((0.001 + d2r) * (1 + a * np.power(d2r, b)))
            gr = np.clip(cor[:, None] * diffr, -4, 4) * alpha
            np.add.at(Y, reps, gr)

        return Y


def _knn_loo_acc(Z, y, k=10):
    """Leave-one-out kNN label accuracy in the given space."""
    D = _pairwise_dist(Z)
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1)[:, :k]
    preds = np.array([np.bincount(y[row]).argmax() for row in nn])
    return (preds == y).mean()


if __name__ == "__main__":
    np.random.seed(0)

    # Synthetic data: 3 well-separated Gaussian blobs living in 10-D space.
    n_per, dim, n_cls = 60, 10, 3
    centers = np.random.randn(n_cls, dim) * 6.0
    X = np.vstack([centers[c] + np.random.randn(n_per, dim) for c in range(n_cls)])
    y = np.repeat(np.arange(n_cls), n_per)

    Y = UMAP(n_components=2, n_neighbors=15).fit_transform(X)

    emb_acc = _knn_loo_acc(Y, y)
    rand_acc = _knn_loo_acc(np.random.randn(*Y.shape), y)
    majority = np.bincount(y).max() / len(y)

    print("Fitted low-dim params a={:.3f} b={:.3f}".format(*_fit_ab()))
    print("Embedded 180 points from 10-D -> 2-D")
    print("Majority-class baseline accuracy:      {:.3f}".format(majority))
    print("Random-embedding kNN accuracy:         {:.3f}".format(rand_acc))
    print("UMAP-embedding kNN accuracy:           {:.3f}".format(emb_acc))
    print("Structure recovered (beats baseline):", emb_acc > majority + 0.3)
