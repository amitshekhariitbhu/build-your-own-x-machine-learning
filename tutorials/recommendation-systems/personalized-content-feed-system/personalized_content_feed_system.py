import numpy as np

def sigmoid(z):
    # Numerically stable sigmoid mapping logits to (0, 1)
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

class ContentFeedRanker:
    """Learned feed ranker: a logistic-regression click model trained by
    gradient descent on engineered impression features (interest match,
    recency, popularity). The feed is ranked by predicted click probability."""

    def __init__(self, learning_rate=0.5, n_epochs=400, l2=1e-3):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.l2 = l2
        self.weights = None
        self.bias = 0.0
        self.mean = None
        self.std = None

    def _standardize(self, X):
        # Z-score features with training statistics for stable gradient descent
        return (X - self.mean) / self.std

    def fit(self, X, y):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        Xs = self._standardize(X)
        n_samples, n_features = Xs.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Batch gradient descent on binary cross-entropy with L2 regularization
        for _ in range(self.n_epochs):
            p = sigmoid(Xs @ self.weights + self.bias)
            error = p - y
            dw = Xs.T @ error / n_samples + self.l2 * self.weights
            db = error.mean()
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        return self

    def predict_proba(self, X):
        # Predicted click probability for each (user, item) impression
        return sigmoid(self._standardize(X) @ self.weights + self.bias)

    def recommend(self, X_feed, k=10):
        # Rank a user's candidate feed and return indices of the top-k items
        scores = self.predict_proba(X_feed)
        return np.argsort(-scores)[:k]


def make_feed_data(n_users=300, n_items=250, n_topics=6, seed=0):
    """Synthetic impressions with a PLANTED click model. Clicks depend on
    interest match, recency and popularity, so a ranker that learns those
    weights should beat popularity-only and chronological-only baselines."""
    rng = np.random.RandomState(seed)
    # Each user is a distribution over topics; each item has one primary topic
    user_interest = rng.rand(n_users, n_topics)
    user_interest /= user_interest.sum(axis=1, keepdims=True)
    item_topic = rng.randint(0, n_topics, size=n_items)
    item_pop = rng.rand(n_items)                      # global item attractiveness

    # Every (user, item) pair is an impression in the user's candidate feed
    users = np.repeat(np.arange(n_users), n_items)
    items = np.tile(np.arange(n_items), n_users)
    interest = user_interest[users, item_topic[items]]  # user's taste for topic
    recency = rng.rand(users.size)                       # per-impression freshness
    popularity = item_pop[items]

    X = np.column_stack([interest, recency, popularity])
    # Planted ground-truth click model (interest is the dominant signal)
    logit = 6.0 * interest + 2.0 * recency + 2.5 * popularity - 4.3
    clicks = (rng.rand(users.size) < sigmoid(logit)).astype(float)
    return X, clicks, users


def auc_score(scores, labels):
    # AUC via the rank-sum (Mann-Whitney U) identity, with tie-averaged ranks
    order = np.argsort(scores)
    ranks = np.empty(scores.size)
    ranks[order] = np.arange(1, scores.size + 1)
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def precision_at_k(scores, clicks, users, k=10):
    # Mean over users of the click rate within each user's top-k ranked feed
    totals = []
    for u in np.unique(users):
        idx = np.where(users == u)[0]
        if idx.size < k:
            continue
        top = idx[np.argsort(-scores[idx])[:k]]
        totals.append(clicks[top].mean())
    return np.mean(totals)


if __name__ == "__main__":
    np.random.seed(0)
    X, clicks, users = make_feed_data()

    # Split impressions into train / held-out test sets
    n = X.shape[0]
    perm = np.random.RandomState(1).permutation(n)
    cut = int(0.7 * n)
    tr, te = perm[:cut], perm[cut:]

    ranker = ContentFeedRanker(learning_rate=0.5, n_epochs=400)
    ranker.fit(X[tr], clicks[tr])

    scores = ranker.predict_proba(X[te])
    auc = auc_score(scores, clicks[te])

    # Baselines rank the same held-out feeds by a single raw feature
    Xte, cte, ute = X[te], clicks[te], users[te]
    k = 10
    p_learned = precision_at_k(scores, cte, ute, k)
    p_pop = precision_at_k(Xte[:, 2], cte, ute, k)   # popularity-only ranking
    p_chrono = precision_at_k(Xte[:, 1], cte, ute, k)  # recency-only (chronological)
    p_random = cte.mean()                             # base click rate = random feed

    print("Learned click-model weights [interest, recency, popularity]:",
          np.round(ranker.weights, 3))
    print(f"Held-out ranking AUC : {auc:.3f}  (random = 0.500)")
    print(f"Precision@{k} learned    : {p_learned:.3f}")
    print(f"Precision@{k} popularity : {p_pop:.3f}  (baseline)")
    print(f"Precision@{k} chrono     : {p_chrono:.3f}  (baseline)")
    print(f"Precision@{k} random     : {p_random:.3f}  (base click rate)")
    lift = p_learned / max(p_pop, p_chrono, p_random)
    print(f"Learned ranker beats best baseline by {lift:.2f}x -> "
          f"{'PASS' if auc > 0.7 and p_learned > max(p_pop, p_chrono) else 'FAIL'}")
