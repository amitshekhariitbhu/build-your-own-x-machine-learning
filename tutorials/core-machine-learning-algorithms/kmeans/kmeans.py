import numpy as np

class KMeans:
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
        self.inertia = None

    def _assign(self, X):
        # Squared Euclidean distance from each point to each centroid, pick nearest
        dists = np.sum((X[:, None, :] - self.centroids[None, :, :]) ** 2, axis=2)
        return np.argmin(dists, axis=1)

    def fit(self, X):
        # Seed k centroids from random distinct points
        idx = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[idx].astype(float)

        for _ in range(self.max_iter):
            labels = self._assign(X)
            # Recompute centroids as cluster means (keep old centroid if cluster empty)
            new_centroids = np.array([
                X[labels == c].mean(axis=0) if np.any(labels == c) else self.centroids[c]
                for c in range(self.k)
            ])
            if np.allclose(new_centroids, self.centroids):
                self.centroids = new_centroids
                break
            self.centroids = new_centroids

        self.labels = self._assign(X)
        # Inertia = sum of squared distances of points to their assigned centroid
        self.inertia = np.sum((X - self.centroids[self.labels]) ** 2)
        return self

    def predict(self, X):
        return self._assign(X)


# Example usage
if __name__ == "__main__":
    np.random.seed(0)

    # Generate 3 Gaussian blobs
    true_centers = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 6.0]])
    X = np.vstack([c + np.random.randn(50, 2) for c in true_centers])

    # Cluster
    model = KMeans(k=3).fit(X)

    print("Final centroids:")
    print(np.round(model.centroids, 3))
    print("Inertia:", round(model.inertia, 3))

    # Correctness signal: each recovered centroid should be near a true center
    matched = [np.min(np.sum((c - true_centers) ** 2, axis=1)) ** 0.5 for c in model.centroids]
    print("Max distance from a true center:", round(max(matched), 3))
