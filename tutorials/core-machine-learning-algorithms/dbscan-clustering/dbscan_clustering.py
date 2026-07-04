import numpy as np

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps                  # neighborhood radius
        self.min_samples = min_samples  # core-point density threshold (incl. self)
        self.labels = None

    def _region_query(self, X, i):
        # Indices of all points within eps of point i (Euclidean)
        dists = np.linalg.norm(X - X[i], axis=1)
        return np.where(dists <= self.eps)[0]

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        labels = np.full(n, -1)         # -1 = noise / unassigned
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                continue                # not a core point -> leave as noise (for now)

            # Start a new cluster and grow it by expanding through core points
            labels[i] = cluster_id
            seeds = list(neighbors)
            k = 0
            while k < len(seeds):
                j = seeds[k]
                k += 1
                if labels[j] == -1:     # noise reachable from a core -> border point
                    labels[j] = cluster_id
                if visited[j]:
                    continue
                visited[j] = True
                j_neighbors = self._region_query(X, j)
                if len(j_neighbors) >= self.min_samples:
                    seeds.extend(j_neighbors)   # j is core -> absorb its neighborhood
                labels[j] = cluster_id
            cluster_id += 1

        self.labels = labels
        return self

    def fit_predict(self, X):
        return self.fit(X).labels


# Example usage
if __name__ == "__main__":
    np.random.seed(0)

    # Two dense blobs plus uniformly scattered noise points
    blob1 = np.array([0.0, 0.0]) + 0.4 * np.random.randn(60, 2)
    blob2 = np.array([6.0, 6.0]) + 0.4 * np.random.randn(60, 2)
    noise = np.random.uniform(-3, 9, size=(20, 2))
    X = np.vstack([blob1, blob2, noise])

    model = DBSCAN(eps=0.6, min_samples=5).fit(X)
    labels = model.labels

    clusters = sorted(l for l in set(labels) if l != -1)
    n_noise = int(np.sum(labels == -1))

    print("Cluster labels:", labels)
    print("Number of clusters found:", len(clusters))
    print("Noise points:", n_noise)

    # Correctness signal: two dense blobs should be recovered as 2 clusters
    print("Recovered 2 blobs:", len(clusters) == 2)
