import numpy as np


class FilterRecommender:
    """Content-based image-filter recommender (Instagram / Pinterest style).

    Each photo is a visual feature vector (brightness, warmth, saturation,
    contrast, ...). From a history of (image-features, chosen-filter) pairs it
    recommends a filter for a NEW image via distance-weighted k-NN: find the k
    most visually similar past photos and vote on the filter they were given."""

    def __init__(self, k=15, n_filters=None):
        self.k = k                 # neighbors to poll
        self.n_filters = n_filters

    def fit(self, features, filters):
        # Store the history; standardize features so every visual axis counts equally.
        X = np.asarray(features, dtype=float)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-12
        self.X = (X - self.mean) / self.std
        self.filters = np.asarray(filters)
        self.n_filters = self.n_filters or int(self.filters.max()) + 1
        return self

    def _rank(self, x):
        # Distance-weighted vote over the k nearest history photos.
        z = (np.asarray(x, dtype=float) - self.mean) / self.std
        d = np.linalg.norm(self.X - z, axis=1)          # euclidean distances
        nn = np.argsort(d)[:self.k]                      # k closest photos
        w = 1.0 / (d[nn] + 1e-9)                          # closer -> louder vote
        votes = np.zeros(self.n_filters)
        np.add.at(votes, self.filters[nn], w)
        return votes

    def recommend(self, x, top_n=3):
        # Return the top_n filters, best first.
        return np.argsort(self._rank(x))[::-1][:top_n]

    def predict(self, x):
        # Single best filter for one image.
        return int(np.argmax(self._rank(x)))


if __name__ == "__main__":
    np.random.seed(0)

    # ---- Synthetic photos with PLANTED visual clusters ----------------------
    # D visual attributes; each of C clusters (e.g. "beach", "food", "night")
    # is a blob around its own center, and every cluster has ONE preferred
    # filter its users tend to apply. Same-cluster photos sit close together.
    D, C, F = 6, 8, 6                 # dims, visual clusters, distinct filters
    per_cluster = 150
    centers = np.random.randn(C, D) * 2.5             # well-separated blobs
    pref_filter = np.random.randint(0, F, size=C)     # cluster -> preferred filter

    feats, filt, clus = [], [], []
    for c in range(C):
        pts = centers[c] + 0.7 * np.random.randn(per_cluster, D)   # visual noise
        for p in pts:
            # Users mostly pick the cluster's filter, but 25% of the time stray.
            f = pref_filter[c] if np.random.rand() < 0.75 else np.random.randint(F)
            feats.append(p); filt.append(f); clus.append(c)
    feats = np.array(feats); filt = np.array(filt); clus = np.array(clus)

    # ---- Train / held-out split ---------------------------------------------
    n = len(feats)
    idx = np.random.permutation(n)
    cut = int(0.75 * n)
    tr, te = idx[:cut], idx[cut:]

    model = FilterRecommender(k=15, n_filters=F).fit(feats[tr], filt[tr])

    # ---- Correctness: does the recommended filter match the cluster's
    #      planted preferred filter for HELD-OUT photos? ----------------------
    pred = np.array([model.predict(feats[i]) for i in te])
    truth = pref_filter[clus[te]]                      # ground-truth preference
    acc = np.mean(pred == truth)
    rand = 1.0 / F                                      # blind guessing
    # Majority baseline: always shout the globally most common preferred filter.
    majority = np.bincount(truth, minlength=F).max() / len(truth)

    # Top-3 hit rate: is the true preferred filter among our 3 suggestions?
    top3 = np.mean([truth[j] in model.recommend(feats[i], top_n=3)
                    for j, i in enumerate(te)])

    q = te[0]
    print("Photos: %d | visual clusters: %d | filters: %d | dims: %d"
          % (n, C, F, D))
    print("Query photo -> cluster %d (prefers filter %d)"
          % (clus[q], pref_filter[clus[q]]))
    print("  recommended filters:", model.recommend(feats[q], top_n=3).tolist())
    print("-" * 60)
    print("Held-out top-1 accuracy: %.3f" % acc)
    print("Held-out top-3 hitrate:  %.3f" % top3)
    print("Random baseline:         %.3f" % rand)
    print("Majority baseline:       %.3f" % majority)
    print("Improvement over random: %.1fx" % (acc / rand))
    print("Beats baselines:", bool(acc > 2 * rand and acc > majority))
