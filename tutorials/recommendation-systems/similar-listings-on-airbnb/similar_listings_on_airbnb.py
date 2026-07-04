import numpy as np

class ListingEmbeddings:
    """Airbnb-style listing embeddings via item2vec skip-gram with negative sampling.

    Listings that co-occur in browsing sessions are pushed together in vector
    space; similar listings are found as nearest cosine neighbors.
    """

    def __init__(self, n_items, dim=16, neg_k=5, lr=0.05, epochs=5, batch=1024):
        self.n_items = n_items
        self.dim = dim
        self.neg_k = neg_k
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        # Two embedding tables (input/center and output/context), as in word2vec
        self.W_in = (np.random.rand(n_items, dim) - 0.5) / dim
        self.W_out = np.zeros((n_items, dim))

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _pairs(self, sessions, window):
        # Build (center, context) index pairs from fixed-length sessions via shifts
        centers, contexts = [], []
        for off in range(1, window + 1):
            a, b = sessions[:, :-off].ravel(), sessions[:, off:].ravel()
            centers += [a, b]          # both directions -> symmetric context
            contexts += [b, a]
        return np.concatenate(centers), np.concatenate(contexts)

    def fit(self, sessions, window=5):
        centers, contexts = self._pairs(sessions, window)
        # Negative-sampling distribution = unigram frequency ^ 0.75
        counts = np.bincount(contexts, minlength=self.n_items).astype(float)
        neg_p = counts ** 0.75
        neg_p /= neg_p.sum()

        n = len(centers)
        for _ in range(self.epochs):
            order = np.random.permutation(n)
            for s in range(0, n, self.batch):
                idx = order[s:s + self.batch]
                c, o = centers[idx], contexts[idx]
                negs = np.random.choice(self.n_items, size=(len(idx), self.neg_k), p=neg_p)

                v = self.W_in[c]                              # (B, d)
                u_pos = self.W_out[o]                         # (B, d)
                u_neg = self.W_out[negs]                      # (B, K, d)

                # Gradient coefficients from binary logistic loss (label 1 pos, 0 neg)
                g_pos = self._sigmoid(np.sum(v * u_pos, axis=1)) - 1.0          # (B,)
                g_neg = self._sigmoid(np.sum(v[:, None, :] * u_neg, axis=2))    # (B, K)

                grad_v = g_pos[:, None] * u_pos + np.sum(g_neg[:, :, None] * u_neg, axis=1)
                grad_u_pos = g_pos[:, None] * v
                grad_u_neg = g_neg[:, :, None] * v[:, None, :]

                # Scatter-add gradients (indices repeat) then step
                np.add.at(self.W_in, c, -self.lr * grad_v)
                np.add.at(self.W_out, o, -self.lr * grad_u_pos)
                np.add.at(self.W_out, negs.ravel(), -self.lr * grad_u_neg.reshape(-1, self.dim))
        return self

    def _unit(self):
        E = self.W_in
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def similar(self, item, k=5):
        # Top-k nearest listings by cosine similarity (excluding the query itself)
        E = self._unit()
        sims = E @ E[item]
        sims[item] = -np.inf
        return np.argsort(sims)[::-1][:k]


def make_sessions(n_types=10, per_type=20, n_sessions=2500, length=10, purity=0.9):
    # Each listing has a planted latent "type"; sessions co-view same-type listings
    n_items = n_types * per_type
    types = np.repeat(np.arange(n_types), per_type)
    by_type = [np.where(types == t)[0] for t in range(n_types)]

    sessions = np.empty((n_sessions, length), dtype=int)
    for s in range(n_sessions):
        t = np.random.randint(n_types)
        same = np.random.random(length) < purity           # mostly same-type, some noise
        for j in range(length):
            pool = by_type[t] if same[j] else np.arange(n_items)
            sessions[s, j] = np.random.choice(pool)
    return sessions, types


if __name__ == "__main__":
    np.random.seed(0)

    sessions, types = make_sessions()
    n_items, n_types = len(types), types.max() + 1

    model = ListingEmbeddings(n_items).fit(sessions)

    # Correctness: do a listing's top-K neighbors share its planted type?
    K = 5
    hits = sum(np.mean(types[model.similar(i, K)] == types[i]) for i in range(n_items))
    accuracy = hits / n_items
    baseline = (n_items // n_types - 1) / (n_items - 1)   # random same-type rate

    print("Listings:", n_items, "| Types:", n_types, "| Embedding dim:", model.dim)
    q = 0
    print("Query listing", q, "(type", int(types[q]), ") -> neighbors types:",
          types[model.similar(q, K)].tolist())
    print(f"Top-{K} same-type accuracy: {accuracy:.3f}")
    print(f"Random baseline:           {baseline:.3f}")
    print("Beats baseline:", accuracy > baseline)
