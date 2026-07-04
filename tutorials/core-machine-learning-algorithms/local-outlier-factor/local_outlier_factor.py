import numpy as np

class LocalOutlierFactor:
    def __init__(self, k=20):
        self.k = k          # number of neighbors
        self.X = None
        self.lof = None     # LOF score per point

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        k = min(self.k, n - 1)  # can't have more neighbors than other points

        # Pairwise Euclidean distances
        dists = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)     # exclude self from neighbors

        # k-nearest neighbors and k-distance (distance to the k-th neighbor)
        neighbors = np.argsort(dists, axis=1)[:, :k]
        k_distance = dists[np.arange(n), neighbors[:, -1]]

        # reach_dist(a, b) = max(k_distance(b), dist(a, b))
        # Local reachability density: inverse of mean reach-dist to neighbors
        lrd = np.zeros(n)
        for i in range(n):
            reach = np.maximum(k_distance[neighbors[i]], dists[i, neighbors[i]])
            lrd[i] = 1.0 / (reach.mean() + 1e-10)

        # LOF = mean(lrd of neighbors) / lrd(point); ~1 inlier, >>1 outlier
        self.lof = np.array([lrd[neighbors[i]].mean() / lrd[i] for i in range(n)])
        self.X = X
        return self

    def fit_predict(self, X, threshold=2.0):
        # Returns -1 for outliers (LOF > threshold), 1 for inliers
        self.fit(X)
        return np.where(self.lof > threshold, -1, 1)


# Example usage
if __name__ == "__main__":
    np.random.seed(0)

    # Dense cluster of inliers plus a few far-away outliers
    inliers = np.array([0.0, 0.0]) + 0.5 * np.random.randn(60, 2)
    outliers = np.array([[6.0, 6.0], [-5.0, 5.0], [5.0, -5.0]])
    X = np.vstack([inliers, outliers])

    model = LocalOutlierFactor(k=20)
    labels = model.fit_predict(X, threshold=2.0)

    print("LOF scores (last 3 = injected outliers):")
    print(np.round(model.lof[-3:], 2))
    print("Mean LOF of inliers: %.2f" % model.lof[:60].mean())

    flagged = np.where(labels == -1)[0]
    print("Flagged outlier indices:", flagged)

    # Correctness signal: exactly the 3 injected points (indices 60,61,62) flagged
    print("Recovered 3 injected outliers:", set(flagged) == {60, 61, 62})
