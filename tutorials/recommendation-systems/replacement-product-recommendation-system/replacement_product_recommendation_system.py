import numpy as np


class ReplacementRecommender:
    """Substitute-product recommender: for a query product, rank other products
    in the SAME category by content-feature cosine similarity. The closest,
    same-category, different products are the best replacements."""

    def __init__(self, top_k=5):
        self.top_k = top_k

    def fit(self, features, categories, brands=None):
        X = np.asarray(features, dtype=float)
        # L2-normalize rows so a dot product equals cosine similarity.
        self.X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        self.categories = np.asarray(categories)
        self.brands = None if brands is None else np.asarray(brands)
        # Full item x item cosine-similarity matrix.
        self.sim = self.X @ self.X.T
        return self

    def recommend(self, item, top_k=None, same_category=True):
        # Best replacements: highest similarity, excluding the item itself.
        k = top_k or self.top_k
        sims = self.sim[item].copy()
        sims[item] = -np.inf  # a product is never its own replacement
        if same_category:
            # A substitute must belong to the query's category.
            sims[self.categories != self.categories[item]] = -np.inf
        order = np.argsort(sims)[::-1]
        order = order[np.isfinite(sims[order])]  # drop masked-out items
        return order[:k]

    def neighbors(self, item, k):
        # Unrestricted nearest neighbors (self excluded) — used to test whether
        # the content features alone already recover category structure.
        sims = self.sim[item].copy()
        sims[item] = -np.inf
        return np.argsort(sims)[::-1][:k]


if __name__ == "__main__":
    np.random.seed(0)

    n_cat = 5        # product categories
    n_clu = 4        # planted substitute clusters per category
    per_clu = 6      # products per substitute cluster
    dim = 30         # content feature dimensions
    n_brands = 8

    # Planted signal: each product = shared category vector + cluster vector + noise.
    # Same category -> shared category vector; same substitute cluster -> near-identical.
    cat_base = np.random.randn(n_cat, dim)            # per-category signal
    clu_off = np.random.randn(n_cat, n_clu, dim)      # per-cluster signal
    feats, cats, clus = [], [], []
    for c in range(n_cat):
        for s in range(n_clu):
            proto = 1.0 * cat_base[c] + 1.5 * clu_off[c, s]
            for _ in range(per_clu):
                feats.append(proto + 0.35 * np.random.randn(dim))
                cats.append(c)
                clus.append(c * n_clu + s)            # global substitute-cluster id
    feats = np.array(feats)
    cats = np.array(cats)
    clus = np.array(clus)
    brands = np.random.randint(0, n_brands, size=len(cats))

    model = ReplacementRecommender(top_k=5).fit(feats, cats, brands)

    n = len(cats)
    K = 5
    per_cat = n_clu * per_clu  # items per category

    # Metric 1: do raw content features recover CATEGORY? Check unrestricted
    # nearest neighbors and measure how often they share the query's category.
    cat_hits = sum(np.sum(cats[model.neighbors(i, K)] == cats[i]) for i in range(n))
    cat_acc = cat_hits / (n * K)
    cat_base_rate = (per_cat - 1) / (n - 1)  # chance a random other item matches

    # Metric 2: do same-category replacements recover the planted SUBSTITUTE
    # cluster? Precision@K = fraction of recommendations from the same cluster.
    clu_hits, clu_tot = 0, 0
    for i in range(n):
        rec = model.recommend(i, top_k=K)  # within-category ranking
        clu_hits += np.sum(clus[rec] == clus[i])
        clu_tot += len(rec)
    clu_prec = clu_hits / clu_tot
    clu_base_rate = (per_clu - 1) / (per_cat - 1)  # random within-category chance

    # Show one concrete query and its top replacements.
    q = 0
    rec = model.recommend(q, top_k=K)
    print("Products: %d | categories: %d | substitute clusters: %d"
          % (n, n_cat, n_cat * n_clu))
    print("Query product %d  ->  category=%d cluster=%d brand=%d"
          % (q, cats[q], clus[q], brands[q]))
    print("Top replacements:", rec.tolist())
    print("  their categories:", cats[rec].tolist(), "(query cat=%d)" % cats[q])
    print("  their clusters:  ", clus[rec].tolist(), "(query cluster=%d)" % clus[q])
    print("-" * 60)
    print("Category recovery @K=%d (unrestricted NN): %.3f  (random %.3f)"
          % (K, cat_acc, cat_base_rate))
    print("Substitute-cluster Precision@%d (in-category): %.3f  (random %.3f)"
          % (K, clu_prec, clu_base_rate))
    print("Beats random baseline:",
          bool(cat_acc > 2 * cat_base_rate and clu_prec > 2 * clu_base_rate))
