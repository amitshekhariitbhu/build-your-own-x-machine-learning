import numpy as np

class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage="average"):
        self.n_clusters = n_clusters
        self.linkage = linkage          # 'single' | 'complete' | 'average'
        self.labels = None

    def _cluster_distance(self, X, a, b):
        # Pairwise point distances between the two clusters, reduced by linkage
        d = np.linalg.norm(X[a][:, None, :] - X[b][None, :, :], axis=2)
        if self.linkage == "single":
            return d.min()             # closest pair
        if self.linkage == "complete":
            return d.max()             # farthest pair
        return d.mean()                # average linkage

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        # Start with each point as its own singleton cluster
        clusters = [[i] for i in range(n)]

        # Repeatedly merge the two closest clusters until n_clusters remain
        while len(clusters) > self.n_clusters:
            best, pair = np.inf, (0, 1)
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._cluster_distance(X, clusters[i], clusters[j])
                    if dist < best:
                        best, pair = dist, (i, j)
            i, j = pair
            clusters[i] += clusters[j]  # absorb j into i
            clusters.pop(j)

        # Assign a label per point from its final cluster
        self.labels = np.empty(n, dtype=int)
        for c_idx, members in enumerate(clusters):
            self.labels[members] = c_idx
        return self

    def fit_predict(self, X):
        return self.fit(X).labels


# Example usage
if __name__ == "__main__":
    np.random.seed(0)

    # Generate 3 well-separated Gaussian blobs
    true_centers = np.array([[0.0, 0.0], [8.0, 8.0], [0.0, 9.0]])
    X = np.vstack([c + np.random.randn(20, 2) for c in true_centers])
    y_true = np.repeat([0, 1, 2], 20)

    # Cluster bottom-up with average linkage
    model = AgglomerativeClustering(n_clusters=3, linkage="average").fit(X)
    print("Cluster labels:")
    print(model.labels)

    # Correctness signal: purity = fraction agreeing with the true blob after
    # matching each predicted cluster to its majority true group
    correct = 0
    for c in np.unique(model.labels):
        groups = y_true[model.labels == c]
        correct += np.sum(groups == np.bincount(groups).argmax())
    print("Clustering purity:", round(correct / len(X), 3))
