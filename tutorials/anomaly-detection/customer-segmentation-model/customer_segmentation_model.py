import numpy as np


class KMeans:
    """K-Means clustering from scratch with k-means++ seeding (Lloyd's algorithm)."""

    def __init__(self, n_clusters, max_iter=100, n_init=8, tol=1e-6, seed=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.seed = seed

    def _kpp_init(self, X, rng):
        # k-means++: pick spread-out seeds so Lloyd's rarely gets stuck.
        n = X.shape[0]
        centers = [X[rng.randint(n)]]
        d2 = ((X - centers[0]) ** 2).sum(axis=1)
        for _ in range(1, self.n_clusters):
            probs = d2 / d2.sum()                 # sample proportional to distance^2
            idx = rng.choice(n, p=probs)
            centers.append(X[idx])
            d2 = np.minimum(d2, ((X - X[idx]) ** 2).sum(axis=1))
        return np.array(centers)

    def _run_once(self, X, rng):
        C = self._kpp_init(X, rng)
        for _ in range(self.max_iter):
            # Assign each point to its nearest center (squared Euclidean).
            dists = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            labels = dists.argmin(axis=1)
            # Recompute each center as the mean of its members.
            newC = np.array([X[labels == k].mean(axis=0) if np.any(labels == k)
                             else C[k] for k in range(self.n_clusters)])
            shift = np.abs(newC - C).max()
            C = newC
            if shift < self.tol:
                break
        inertia = ((X - C[labels]) ** 2).sum()    # total within-cluster distance
        return C, labels, inertia

    def fit(self, X):
        rng = np.random.RandomState(self.seed)
        best = None
        # Restart several times, keep the lowest-inertia solution.
        for _ in range(self.n_init):
            C, labels, inertia = self._run_once(X, rng)
            if best is None or inertia < best[2]:
                best = (C, labels, inertia)
        self.centers_, self.labels_, self.inertia_ = best
        return self

    def predict(self, X):
        dists = ((X[:, None, :] - self.centers_[None, :, :]) ** 2).sum(axis=2)
        return dists.argmin(axis=1)


def make_customers(n_per=120, seed=0):
    """Synthetic customers with 4 planted segments in RFM-like feature space."""
    rng = np.random.RandomState(seed)
    # Each segment = (recency_days, frequency_orders, monetary_spend).
    protos = {
        "Champions":     [10, 40, 900],   # recent, frequent, big spenders
        "Loyal":         [30, 22, 400],   # steady mid-value buyers
        "At-Risk":       [120, 6, 250],   # slipping away, used to buy
        "New/Low-Value": [20, 3, 60],     # recent but tiny baskets
    }
    names = list(protos)
    X, y = [], []
    for c, name in enumerate(names):
        mu = np.array(protos[name], dtype=float)
        # Per-feature spread scaled to each prototype's magnitude.
        sd = np.array([8.0, 4.0, 0.18 * mu[2]])
        X.append(rng.normal(mu, sd, size=(n_per, 3)))
        y += [c] * n_per
    X = np.vstack(X)
    return X, np.array(y), names


def best_match_accuracy(y_true, labels, K):
    # Clusters are unlabeled: score the best cluster->segment permutation.
    from itertools import permutations
    best = 0.0
    for perm in permutations(range(K)):
        mapped = np.array([perm[l] for l in labels])
        best = max(best, np.mean(mapped == y_true))
    return best


def standardize(X):
    # Features live on wildly different scales; z-score so distance is fair.
    return (X - X.mean(axis=0)) / X.std(axis=0)


if __name__ == "__main__":
    np.random.seed(0)

    X, y, names = make_customers(n_per=120, seed=0)
    K = len(names)
    Xz = standardize(X)

    km = KMeans(n_clusters=K, seed=0).fit(Xz)
    acc = best_match_accuracy(y, km.labels_, K)
    random_level = 1.0 / K                          # chance assignment recovers ~1/K

    print("Customers: %d   Features: %d   Segments: %d"
          % (X.shape[0], X.shape[1], K))
    print("-" * 58)
    print("Segment sizes (true): %s"
          % {names[c]: int(np.sum(y == c)) for c in range(K)})
    print("-" * 58)
    print("Cluster centers (un-standardized R/F/M):")
    centers_raw = km.centers_ * X.std(axis=0) + X.mean(axis=0)
    for k in range(K):
        r, f, m = centers_raw[k]
        print("  cluster %d -> recency=%5.1fd  freq=%4.1f  spend=$%6.1f"
              % (k, r, f, m))
    print("-" * 58)
    print("Segmentation accuracy (best cluster match): %.4f" % acc)
    print("Random-assignment baseline               : %.4f" % random_level)
    print("-" * 58)
    print("K-Means beats random segmentation: %s" % (acc > random_level + 0.30))
