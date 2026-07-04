import numpy as np


class ArticleRecommender:
    """Content-based recommender: build a user profile from read articles'
    TF-IDF vectors, then rank unread articles by cosine to that profile."""

    @staticmethod
    def _normalize(X):
        norm = np.linalg.norm(X, axis=-1, keepdims=True)
        return X / np.where(norm == 0, 1.0, norm)

    def fit(self, docs):
        # docs: list of articles, each a list of word tokens.
        vocab = sorted({t for toks in docs for t in toks})
        self.vocab = {w: i for i, w in enumerate(vocab)}
        V, N = len(vocab), len(docs)

        # Raw term counts (articles x vocab).
        tf = np.zeros((N, V))
        for i, toks in enumerate(docs):
            for t in toks:
                tf[i, self.vocab[t]] += 1.0
        # Sublinear TF: dampen repeated terms.
        tf = np.where(tf > 0, 1.0 + np.log(np.maximum(tf, 1.0)), 0.0)

        # Smoothed inverse document frequency: rare words weigh more.
        df = (tf > 0).sum(axis=0)
        self.idf = np.log((1.0 + N) / (1.0 + df)) + 1.0

        # TF-IDF matrix, L2-normalized rows so a dot product equals cosine.
        self.M = self._normalize(tf * self.idf)
        return self

    def profile(self, read_idx):
        # A user's taste vector: mean TF-IDF of the articles they have read.
        return self._normalize(self.M[read_idx].mean(axis=0))

    def recommend(self, read_idx, k=5):
        # Cosine of the user profile to every article; hide already-read ones.
        scores = self.M @ self.profile(read_idx)
        scores[read_idx] = -np.inf
        return np.argsort(-scores)[:k]


if __name__ == "__main__":
    np.random.seed(0)

    # Planted topics: each has its own characteristic vocabulary.
    topics = {
        "tech":    "algorithm robot data software chip network code compiler cloud".split(),
        "health":  "brain cancer cell disease patient therapy gene immune vaccine".split(),
        "sports":  "match team player score league goal coach season stadium".split(),
        "finance": "market stock trade profit invest bond currency tax revenue".split(),
        "travel":  "flight hotel beach city passport journey resort culture map".split(),
    }
    generic = "the a new world people today story report".split()  # shared filler
    names = list(topics)
    per_topic = 60

    # Synthetic articles: each belongs to one planted topic; its text is that
    # topic's words plus generic filler noise shared across all topics.
    docs, labels = [], []
    for ti, name in enumerate(names):
        for _ in range(per_topic):
            n = np.random.randint(18, 26)
            body = np.random.choice(topics[name], size=n)
            noise = np.random.choice(generic, size=6)
            docs.append(list(body) + list(noise))
            labels.append(ti)
    labels = np.array(labels)
    by_topic = [np.where(labels == t)[0] for t in range(len(names))]

    rec = ArticleRecommender().fit(docs)

    # Each user prefers one topic. Their read history is drawn from that topic;
    # we hold those out and check whether recommendations stay on-topic.
    K, n_users = 5, 200
    hits = np.zeros(n_users)
    for u in range(n_users):
        pref = np.random.randint(len(names))
        read_idx = np.random.choice(by_topic[pref], size=6, replace=False)
        top = rec.recommend(read_idx, k=K)
        hits[u] = np.mean(labels[top] == pref)   # fraction of top-K on-topic

    precision = hits.mean()
    hit_rate = np.mean(hits > 0)                  # >=1 on-topic in top-K
    baseline = 1.0 / len(names)                   # random pick of a topic

    print("Articles: %d  Topics: %d  Vocab: %d  Users: %d"
          % (len(docs), len(names), len(rec.vocab), n_users))
    print("Precision@%d (profile): %.3f" % (K, precision))
    print("Precision@%d (random):  %.3f" % (K, baseline))
    print("Hit-Rate@%d:            %.3f" % (K, hit_rate))
    print("Improvement over random: %.1fx" % (precision / baseline))
    print("SUCCESS" if precision > 0.8 else "FAIL")
