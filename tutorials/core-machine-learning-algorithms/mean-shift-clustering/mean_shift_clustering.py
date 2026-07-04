import numpy as np

class MeanShift:
    def __init__(self, bandwidth=1.0, max_iter=300, tol=1e-4):
        self.bandwidth = bandwidth      # flat-kernel radius
        self.max_iter = max_iter
        self.tol = tol                  # convergence + mode-merge tolerance
        self.cluster_centers = None
        self.labels = None

    def _shift(self, point, X):
        # Move point to the mean of all samples within bandwidth (flat kernel)
        dists = np.linalg.norm(X - point, axis=1)
        neighbors = X[dists <= self.bandwidth]
        if len(neighbors) == 0:
            return point
        return neighbors.mean(axis=0)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        # Shift every point uphill to its density mode
        modes = np.empty_like(X)
        for i, p in enumerate(X):
            for _ in range(self.max_iter):
                new_p = self._shift(p, X)
                if np.linalg.norm(new_p - p) < self.tol:
                    p = new_p
                    break
                p = new_p
            modes[i] = p

        # Merge modes that landed within tol into unique cluster centers
        centers = []
        labels = np.empty(len(X), dtype=int)
        for i, m in enumerate(modes):
            for c_idx, c in enumerate(centers):
                if np.linalg.norm(m - c) < self.bandwidth * 0.5:
                    labels[i] = c_idx
                    break
            else:
                centers.append(m)
                labels[i] = len(centers) - 1

        self.cluster_centers = np.array(centers)
        self.labels = labels
        return self

    def predict(self, X):
        # Assign each point to the nearest discovered center
        X = np.asarray(X, dtype=float)
        dists = np.linalg.norm(X[:, None, :] - self.cluster_centers[None, :, :], axis=2)
        return np.argmin(dists, axis=1)


# Example usage
if __name__ == "__main__":
    np.random.seed(0)

    # Generate 3 Gaussian blobs
    true_centers = np.array([[0.0, 0.0], [8.0, 8.0], [0.0, 9.0]])
    X = np.vstack([c + np.random.randn(60, 2) for c in true_centers])

    # Cluster (bandwidth chosen relative to blob spread/separation)
    model = MeanShift(bandwidth=3.0).fit(X)

    print("Clusters discovered:", len(model.cluster_centers))
    print("Cluster centers:")
    print(np.round(model.cluster_centers, 3))

    # Correctness signal: recovered centers should match the 3 true centers
    matched = [np.min(np.linalg.norm(true_centers - c, axis=1)) for c in model.cluster_centers]
    print("Max distance from a true center:", round(max(matched), 3))
