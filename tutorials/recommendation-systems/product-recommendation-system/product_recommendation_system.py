import numpy as np


class ProductRecommender:
    """Item-based collaborative filtering via item-item cosine similarity."""

    def __init__(self, k=10):
        self.k = k              # neighbors are implicit: all items, weighted by sim

    def fit(self, M):
        # M: user x product binary interaction matrix (1 = purchased).
        self.M = M
        # Cosine similarity between product columns: normalize then dot.
        norms = np.linalg.norm(M, axis=0)               # per-product L2 norm
        norms[norms == 0] = 1.0                         # avoid divide-by-zero
        Mn = M / norms                                  # unit-norm columns
        self.S = Mn.T @ Mn                              # product x product cosine
        np.fill_diagonal(self.S, 0.0)                   # a product is not its own neighbor
        return self

    def score(self, history):
        # Score every product by summing its similarity to the user's history.
        return self.S @ history                         # weighted by planted structure

    def recommend(self, history, n=5):
        # Top-n products not already in the user's history.
        scores = self.score(history).copy()
        scores[history > 0] = -np.inf                   # hide already-purchased items
        return np.argsort(scores)[::-1][:n]


if __name__ == "__main__":
    np.random.seed(0)

    # Planted structure: products fall into clusters; users prefer a few clusters
    # and mostly purchase within them, so co-purchase reveals the clusters.
    n_users, n_items, n_clusters = 400, 120, 8
    item_cluster = np.repeat(np.arange(n_clusters), n_items // n_clusters)

    M = np.zeros((n_users, n_items))
    for u in range(n_users):
        prefs = np.random.choice(n_clusters, size=2, replace=False)   # user's taste
        n_buy = np.random.randint(6, 12)
        for _ in range(n_buy):
            if np.random.rand() < 0.85:                 # 85% in-taste
                c = prefs[np.random.randint(2)]
            else:                                       # 15% cross-cluster noise
                c = np.random.randint(n_clusters)
            items = np.where(item_cluster == c)[0]
            M[u, np.random.choice(items)] = 1.0

    # Leave-one-out split: hide one purchased product per eligible user.
    train = M.copy()
    heldout = -np.ones(n_users, dtype=int)
    for u in range(n_users):
        bought = np.where(M[u] > 0)[0]
        if len(bought) >= 4:                            # keep enough history to learn
            h = np.random.choice(bought)
            train[u, h] = 0.0
            heldout[u] = h

    model = ProductRecommender().fit(train)

    # Evaluate Hit-Rate@K and cluster purity on the held-out purchases.
    K = 10
    users = np.where(heldout >= 0)[0]
    hits, purity, n_reco, cand_frac = 0, 0.0, 0, 0.0
    for u in users:
        reco = model.recommend(train[u], n=K)
        h = heldout[u]
        hits += int(h in reco)                          # did we recover the hidden item?
        purity += np.mean(item_cluster[reco] == item_cluster[h])  # top-K in same cluster
        n_reco += 1
        cand_frac += K / (n_items - int(train[u].sum()))   # random top-K hit prob

    hit_rate = hits / n_reco
    rand_hit = cand_frac / n_reco                        # baseline: guessing top-K
    clust_acc = purity / n_reco
    rand_clust = 1.0 / n_clusters                        # baseline: guessing a cluster

    print("Users x Products:      %d x %d  (%d clusters)" % (n_users, n_items, n_clusters))
    print("Held-out users:        %d" % n_reco)
    print("Hit-Rate@%d (CF):       %.4f" % (K, hit_rate))
    print("Hit-Rate@%d (random):   %.4f" % (K, rand_hit))
    print("Improvement:           %.1fx" % (hit_rate / rand_hit))
    print("Top-%d cluster purity:  %.4f" % (K, clust_acc))
    print("Cluster purity (random):%.4f" % rand_clust)
    print("CF beats random:       %s" % (hit_rate > rand_hit and clust_acc > rand_clust))
