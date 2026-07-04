import numpy as np


def _kmeans(X, k, n_iter=100, n_restarts=10):
    """Tiny KMeans on the spectral embedding; returns the best (lowest inertia) labels."""
    best_labels, best_inertia = None, np.inf
    for _ in range(n_restarts):
        centroids = X[np.random.choice(len(X), k, replace=False)].astype(float)
        labels = np.zeros(len(X), dtype=int)
        for _ in range(n_iter):
            # Assign each row to the nearest centroid.
            d = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(d, axis=1)
            centroids = np.array([
                X[new_labels == c].mean(axis=0) if np.any(new_labels == c) else centroids[c]
                for c in range(k)
            ])
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
        inertia = np.sum((X - centroids[labels]) ** 2)
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia
    return best_labels


class SpectralClustering:
    """Spectral clustering via the symmetric normalized graph Laplacian (Ng-Jordan-Weiss)."""

    def __init__(self, n_clusters=2, gamma=1.0):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.labels_ = None

    def fit(self, X):
        # RBF affinity matrix: W_ij = exp(-gamma * ||x_i - x_j||^2).
        sq = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
        W = np.exp(-self.gamma * sq)

        # Degree matrix and symmetric normalized Laplacian L = I - D^-1/2 W D^-1/2.
        d = W.sum(axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(d)
        L = np.eye(len(X)) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])

        # Embed using the k eigenvectors of the smallest eigenvalues.
        eigvals, eigvecs = np.linalg.eigh(L)
        U = eigvecs[:, : self.n_clusters]

        # Row-normalize the embedding, then cluster with KMeans.
        U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-10)
        self.labels_ = _kmeans(U, self.n_clusters)
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_


if __name__ == "__main__":
    np.random.seed(0)

    # Two concentric circles: non-convex, so plain KMeans on raw points fails.
    def circle(r, n):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([r * np.cos(t), r * np.sin(t)], axis=1) + 0.05 * np.random.randn(n, 2)

    n = 100
    X = np.vstack([circle(1.0, n), circle(3.0, n)])
    y_true = np.array([0] * n + [1] * n)

    labels = SpectralClustering(n_clusters=2, gamma=5.0).fit_predict(X)

    # Clustering is label-invariant, so score against both label orderings.
    acc = max((labels == y_true).mean(), (labels != y_true).mean())
    print("Cluster sizes:", np.bincount(labels))
    print("Inner-circle labels (first 20):", labels[:20])
    print("Outer-circle labels (first 20):", labels[n:n + 20])
    print("Clustering accuracy vs true circles: {:.3f}".format(acc))
