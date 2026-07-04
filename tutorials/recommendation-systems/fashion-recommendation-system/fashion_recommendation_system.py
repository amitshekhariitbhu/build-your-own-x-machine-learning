import numpy as np


class FashionRecommender:
    """Content-based fashion recommender. Each item is a style-attribute vector
    (one-hot color + pattern + category + formality). Similar items are found by
    cosine similarity; complete-the-look pairs the query with a complementary
    category that shares its color/pattern/formality feel."""

    def __init__(self, top_k=5):
        self.top_k = top_k

    def fit(self, features, categories):
        X = np.asarray(features, dtype=float)
        # L2-normalize rows so a dot product equals cosine similarity.
        self.X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        self.categories = np.asarray(categories)
        self.sim = self.X @ self.X.T          # item x item cosine similarity
        return self

    def recommend(self, item, top_k=None):
        # Most stylistically similar items, excluding the query itself.
        k = top_k or self.top_k
        sims = self.sim[item].copy()
        sims[item] = -np.inf                  # an item is not its own recommendation
        return np.argsort(sims)[::-1][:k]

    def complete_the_look(self, item, target_category, top_k=None):
        # Best match from a DIFFERENT (complementary) category to build an outfit:
        # rank items of target_category by style similarity to the query.
        k = top_k or self.top_k
        sims = self.sim[item].copy()
        sims[self.categories != target_category] = -np.inf
        return np.argsort(sims)[::-1][:k]


if __name__ == "__main__":
    np.random.seed(0)

    # Attribute vocabularies.
    colors      = ["black", "white", "blue", "red", "beige"]
    patterns    = ["solid", "striped", "floral", "checked"]
    categories  = ["top", "bottom", "dress", "shoes", "jacket"]
    formalities = ["casual", "smart", "formal"]
    dims = [len(colors), len(patterns), len(categories), len(formalities)]
    D = sum(dims)                              # concatenated one-hot length

    # Planted styles: each style biases color/pattern/formality toward a signature
    # look (e.g. "formal monochrome", "floral casual"). Items sample attributes
    # mostly from their style's signature, so same-style items cluster in feature
    # space while spanning all product categories.
    n_styles = 6
    all_sigs = [(c, p, f) for c in range(len(colors))
                for p in range(len(patterns)) for f in range(len(formalities))]
    chosen = np.random.choice(len(all_sigs), size=n_styles, replace=False)  # distinct
    style_sig = {s: all_sigs[chosen[s]] for s in range(n_styles)}

    def onehot(idx, size):
        v = np.zeros(size); v[idx] = 1.0; return v

    def sample(vocab_size, signature, keep=0.85):
        # Keep the style's signature attribute with prob `keep`, else pick random.
        return signature if np.random.rand() < keep else np.random.randint(vocab_size)

    per_style = 40
    feats, cats, styles = [], [], []
    for s in range(n_styles):
        sc, sp, sf = style_sig[s]
        for _ in range(per_style):
            c   = sample(len(colors), sc)
            p   = sample(len(patterns), sp)
            cat = np.random.randint(len(categories))     # category spans all styles
            f   = sample(len(formalities), sf)
            vec = np.concatenate([onehot(c, len(colors)),
                                  onehot(p, len(patterns)),
                                  onehot(cat, len(categories)),
                                  onehot(f, len(formalities))])
            feats.append(vec); cats.append(cat); styles.append(s)
    feats = np.array(feats); cats = np.array(cats); styles = np.array(styles)

    model = FashionRecommender(top_k=5).fit(feats, cats)

    n = len(styles)
    K = 5

    # Metric: do top-K similar items share the query's planted STYLE?
    # Precision@K = fraction of recommendations with the same style label.
    hits = sum(np.sum(styles[model.recommend(i, K)] == styles[i]) for i in range(n))
    style_prec = hits / (n * K)
    rand_prec = (per_style - 1) / (n - 1)    # chance a random other item matches

    # Show one concrete query, its similar items, and a complete-the-look pairing.
    q = 0
    rec = model.recommend(q, top_k=K)
    look = model.complete_the_look(q, target_category=3, top_k=3)   # 3 = "shoes"
    print("Items: %d | styles: %d | categories: %d" % (n, n_styles, len(categories)))
    print("Query item %d -> style=%d category=%s" % (q, styles[q], categories[cats[q]]))
    print("Top-%d similar items:" % K, rec.tolist())
    print("  their styles:    ", styles[rec].tolist(), "(query style=%d)" % styles[q])
    print("  their categories:", [categories[c] for c in cats[rec]])
    print("Complete-the-look (shoes):", look.tolist(),
          "styles", styles[look].tolist(), "(query style=%d)" % styles[q])
    print("-" * 60)
    print("Style Precision@%d:      %.3f" % (K, style_prec))
    print("Random baseline:        %.3f" % rand_prec)
    print("Improvement:            %.1fx" % (style_prec / rand_prec))
    print("Beats random baseline:", bool(style_prec > 2 * rand_prec))
