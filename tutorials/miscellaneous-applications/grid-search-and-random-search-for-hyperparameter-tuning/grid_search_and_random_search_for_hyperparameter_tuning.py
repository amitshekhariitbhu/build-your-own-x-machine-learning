import numpy as np
from itertools import product

# Grid Search and Random Search for hyperparameter tuning, from scratch.
# The estimator is a hand-written k-Nearest-Neighbors classifier whose
# hyperparameters (k = neighbors, p = Minkowski distance power) trade bias for
# variance: k=1 overfits label noise, huge k underfits. We build K-fold
# cross-validation and two tuners -- an exhaustive grid search and a randomized
# search -- that hunt the (k, p) space for the best mean CV accuracy. The search
# loops and CV are all written by hand; no scikit-learn.


class KNNClassifier:
    """k-NN with Minkowski distance. k and p are the tuned hyperparameters."""

    def __init__(self, k=5, p=2):
        self.k, self.p = k, p

    def fit(self, X, y):
        self.X, self.y = X, y
        return self

    def predict(self, X):
        # Pairwise Minkowski^p distances (n_query, n_train), then majority vote.
        D = np.sum(np.abs(X[:, None, :] - self.X[None, :, :]) ** self.p, axis=2)
        nn = np.argpartition(D, self.k - 1, axis=1)[:, :self.k]
        neigh = self.y[nn]
        return np.array([np.bincount(row).argmax() for row in neigh])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))


def cross_val_score(params, X, y, k=5):
    """Mean held-out accuracy of a fresh estimator across k CV folds."""
    folds = np.array_split(np.arange(len(y)), k)
    scores = []
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        model = KNNClassifier(**params).fit(X[train], y[train])
        scores.append(model.score(X[val], y[val]))
    return float(np.mean(scores))


def grid_search(param_grid, X, y, cv=5):
    """Exhaustively evaluate every combination in the parameter grid via CV."""
    keys = list(param_grid)
    best, best_score, n = None, -np.inf, 0
    for values in product(*[param_grid[key] for key in keys]):
        params = dict(zip(keys, values))
        s = cross_val_score(params, X, y, cv)
        n += 1
        if s > best_score:
            best_score, best = s, params
    return best, best_score, n


def random_search(param_dist, n_iter, X, y, cv=5):
    """Sample n_iter random combinations from per-parameter samplers."""
    best, best_score = None, -np.inf
    for _ in range(n_iter):
        params = {key: sampler() for key, sampler in param_dist.items()}
        s = cross_val_score(params, X, y, cv)
        if s > best_score:
            best_score, best = s, params
    return best, best_score, n_iter


if __name__ == "__main__":
    np.random.seed(0)

    # PLANTED STRUCTURE: an XOR-like layout -- four Gaussian blobs whose class is
    # the diagonal (top-left & bottom-right = 1, others = 0). Overlapping blobs
    # plus 10% flipped labels make k=1 chase noise, so the tuner must discover a
    # larger, smoothing k to recover the true decision regions.
    n = 600
    centers = np.array([[-2, -2], [2, 2], [-2, 2], [2, -2]], dtype=float)
    center_label = np.array([0, 0, 1, 1])
    c = np.random.randint(0, 4, size=n)
    X = centers[c] + np.random.randn(n, 2) * 1.6
    y = center_label[c]
    flip = np.random.rand(n) < 0.10          # 10% label noise
    y[flip] = 1 - y[flip]

    # Held-out test split: the tuners only ever see the training portion.
    ntr = 150
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]

    # Grid search over a hand-picked grid (8 x 2 = 16 combinations).
    grid = {"k": [1, 3, 5, 9, 15, 25, 41, 61], "p": [1, 2]}
    g_params, g_cv, g_n = grid_search(grid, Xtr, ytr)
    g_test = KNNClassifier(**g_params).fit(Xtr, ytr).score(Xte, yte)

    # Random search: draw k uniformly in [1, 61] and p in {1, 2} (15 samples).
    dist = {
        "k": lambda: int(np.random.randint(1, 62)),
        "p": lambda: int(np.random.choice([1, 2])),
    }
    r_params, r_cv, r_n = random_search(dist, 15, Xtr, ytr)
    r_test = KNNClassifier(**r_params).fit(Xtr, ytr).score(Xte, yte)

    # BASELINES: majority-class guess, and the untuned default k=1 classifier.
    majority = float(max(ytr.mean(), 1 - ytr.mean()))
    default = KNNClassifier(k=1, p=2).fit(Xtr, ytr).score(Xte, yte)

    print("Test-set size: {}, class counts: {}".format(len(yte), np.bincount(yte)))
    print("Baseline majority-class test acc: {:.3f}".format(majority))
    print("Untuned default (k=1) test acc:   {:.3f}".format(default))
    print("-" * 54)
    print("Grid   search ({} CV fits): best={}  CV={:.3f}".format(g_n * 5, g_params, g_cv))
    print("   -> tuned test acc: {:.3f}".format(g_test))
    print("Random search ({} CV fits): best={}  CV={:.3f}".format(r_n * 5, r_params, r_cv))
    print("   -> tuned test acc: {:.3f}".format(r_test))
    print("-" * 54)
    print("Grid   improvement over default:  +{:.3f}".format(g_test - default))
    print("Random improvement over default:  +{:.3f}".format(r_test - default))

    assert g_test > default, "Grid search did not beat the untuned default"
    assert g_test > majority, "Grid search did not beat the majority baseline"
    assert r_test > majority, "Random search did not beat the majority baseline"
    assert g_params["k"] > 1, "Tuning should prefer a smoothing k > 1"
    print("PASS: both searches tuned k>1 and beat the default & majority baselines.")
