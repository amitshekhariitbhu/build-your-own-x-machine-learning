import numpy as np


class SelfOrganizingMap:
    """Kohonen Self-Organizing Map, from scratch.

    A 2-D grid of neurons, each holding a weight vector in input space, learns a
    topology-preserving map of the data by competitive learning: for every sample
    the Best Matching Unit (nearest neuron) and its grid neighbors are pulled
    toward the sample, with the learning rate and neighborhood radius annealed."""

    def __init__(self, grid=(8, 8), n_iter=300, lr0=0.5, sigma0=None, seed=0):
        self.rows, self.cols = grid
        self.n_iter = n_iter
        self.lr0 = lr0
        self.sigma0 = sigma0 if sigma0 else max(grid) / 2.0
        self.seed = seed
        # Fixed (row, col) coordinates of every neuron for neighborhood distances.
        r, c = np.meshgrid(np.arange(self.rows), np.arange(self.cols), indexing="ij")
        self.coords = np.stack([r.ravel(), c.ravel()], axis=1).astype(float)

    def _bmu(self, x):
        # Best Matching Unit = neuron whose weight is closest to sample x.
        d = np.sum((self.W - x) ** 2, axis=1)
        return int(np.argmin(d))

    def fit(self, X):
        rng = np.random.RandomState(self.seed)
        n_neurons = self.rows * self.cols
        # Init weights from the data's range so neurons start inside the manifold.
        lo, hi = X.min(axis=0), X.max(axis=0)
        self.W = lo + (hi - lo) * rng.rand(n_neurons, X.shape[1])

        tau = self.n_iter / np.log(self.sigma0)  # radius decay time constant
        for t in range(self.n_iter):
            x = X[rng.randint(len(X))]
            lr = self.lr0 * np.exp(-t / self.n_iter)          # anneal learning rate
            sigma = self.sigma0 * np.exp(-t / tau)            # shrink neighborhood
            b = self._bmu(x)
            # Gaussian neighborhood on the GRID pulls nearby neurons together too,
            # which is what makes the final map topology-preserving.
            grid_d2 = np.sum((self.coords - self.coords[b]) ** 2, axis=1)
            h = np.exp(-grid_d2 / (2.0 * sigma ** 2))[:, None]
            self.W += lr * h * (x - self.W)
        return self

    def transform(self, X):
        # Map each sample to its BMU index (its coordinate on the map).
        return np.array([self._bmu(x) for x in X])

    def label_neurons(self, X, y):
        # Give every neuron the majority label of the training points that land on
        # it; empty neurons inherit the label of the nearest labeled neuron.
        bmu = self.transform(X)
        n_classes = int(y.max()) + 1
        self.neuron_label = np.full(self.rows * self.cols, -1)
        for k in range(self.rows * self.cols):
            members = y[bmu == k]
            if len(members):
                self.neuron_label[k] = np.bincount(members, minlength=n_classes).argmax()
        labeled = np.where(self.neuron_label >= 0)[0]
        for k in range(self.rows * self.cols):
            if self.neuron_label[k] < 0:
                d = np.sum((self.coords[labeled] - self.coords[k]) ** 2, axis=1)
                self.neuron_label[k] = self.neuron_label[labeled[np.argmin(d)]]
        return self

    def predict(self, X):
        # Classify a sample by the label of the neuron it activates.
        return self.neuron_label[self.transform(X)]


if __name__ == "__main__":
    np.random.seed(0)

    # PLANTED STRUCTURE: 4 well-separated Gaussian blobs in a 12-dim space. Each
    # blob is a latent class the SOM must organize onto distinct map regions.
    n_per, dim, k = 60, 12, 4
    centers = np.random.randn(k, dim) * 6.0
    X = np.vstack([c + np.random.randn(n_per, dim) for c in centers])
    y = np.repeat(np.arange(k), n_per)

    # Held-out split.
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    cut = int(0.7 * len(X))
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]

    som = SelfOrganizingMap(grid=(8, 8), n_iter=2000, lr0=0.5).fit(Xtr)
    som.label_neurons(Xtr, ytr)

    acc = np.mean(som.predict(Xte) == yte)

    # BASELINE: always predict the most common training class.
    majority = np.bincount(ytr).argmax()
    base_acc = np.mean(yte == majority)

    # Topographic quality: fraction of samples whose 1st and 2nd BMUs are grid
    # neighbors. High values mean the map preserved input-space topology.
    def topographic_error(model, data):
        errs = 0
        for x in data:
            d = np.sum((model.W - x) ** 2, axis=1)
            b1, b2 = np.argsort(d)[:2]
            if np.sum((model.coords[b1] - model.coords[b2]) ** 2) > 2.0:
                errs += 1
        return errs / len(data)

    te = topographic_error(som, Xte)

    print("Samples: {}  dim: {}  classes: {}  map: {}x{}".format(
        len(X), dim, k, som.rows, som.cols))
    print("Majority-class baseline accuracy: {:.3f}".format(base_acc))
    print("SOM held-out classification acc:  {:.3f}".format(acc))
    print("Improvement over baseline:        +{:.3f}".format(acc - base_acc))
    print("Topographic error (lower=better): {:.3f}".format(te))
    assert acc > 0.9, "SOM failed to organize the classes"
    assert acc > base_acc + 0.3, "SOM did not beat the majority baseline"
    print("PASS: SOM organized the planted classes and crushed the baseline.")
