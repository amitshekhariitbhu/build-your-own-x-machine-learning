import numpy as np


class TEDRecommender:
    """Content-based item-to-item recommender: rank TED talks by TF-IDF cosine."""

    @staticmethod
    def _normalize(X):
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norm == 0, 1.0, norm)

    def fit(self, docs):
        # docs: list of talks, each a list of tokens (topic tags + transcript words).
        self.docs = docs

        # Vocabulary: every distinct token mapped to a column index.
        vocab = sorted({t for toks in docs for t in toks})
        self.vocab = {w: i for i, w in enumerate(vocab)}
        V, N = len(vocab), len(docs)

        # Raw term counts (talks x vocab).
        tf = np.zeros((N, V))
        for i, toks in enumerate(docs):
            for t in toks:
                tf[i, self.vocab[t]] += 1.0
        # Sublinear TF: dampen repeated terms.
        tf = np.where(tf > 0, 1.0 + np.log(np.maximum(tf, 1.0)), 0.0)

        # Smoothed inverse document frequency: rare words weigh more.
        df = (tf > 0).sum(axis=0)
        self.idf = np.log((1.0 + N) / (1.0 + df)) + 1.0

        # TF-IDF matrix with L2-normalized rows (so dot product = cosine).
        self.M = self._normalize(tf * self.idf)
        return self

    def similar(self, i):
        # Cosine similarity of talk i to every other talk (self excluded).
        scores = self.M @ self.M[i]
        scores[i] = -np.inf
        return scores

    def recommend(self, i, k=5):
        # Top-k talks most similar to talk i (item-to-item, no query text).
        return np.argsort(-self.similar(i))[:k]


if __name__ == "__main__":
    np.random.seed(0)

    # Planted topics: each has a signature tag + a characteristic vocabulary.
    topics = {
        "technology": "algorithm robot data software internet chip network code ai".split(),
        "health":     "brain cancer cell disease patient therapy gene immune sleep".split(),
        "climate":    "carbon ocean warming energy planet emission forest species solar".split(),
        "psychology": "emotion behavior mind memory bias empathy habit fear trust".split(),
        "education":  "student teacher learning school curriculum skill literacy classroom".split(),
        "economics":  "market growth trade poverty wealth policy labor investment currency".split(),
    }
    generic = "world people idea change story future today talk".split()  # shared filler
    names = list(topics)

    # Synthetic talks: each belongs to one planted topic. A talk is its topic
    # tag plus a bag of topic words drawn with noise from generic filler.
    docs, labels = [], []
    for ti, name in enumerate(names):
        for _ in range(50):
            n = np.random.randint(18, 26)
            body = np.random.choice(topics[name], size=n)     # transcript tokens
            noise = np.random.choice(generic, size=6)         # off-topic filler
            tag = "tag_" + name                               # topic tag feature
            docs.append([tag] + list(body) + list(noise))
            labels.append(ti)
    labels = np.array(labels)

    rec = TEDRecommender().fit(docs)

    # Item-to-item quality: for every talk, do its top-K recommendations share
    # the same planted topic? Compare against the random level (1 / n_topics).
    K = 5
    hits = np.array([np.mean(labels[rec.recommend(i, k=K)] == labels[i])
                     for i in range(len(docs))])
    precision = hits.mean()
    hit_rate = np.mean(hits > 0)                # at least one same-topic in top-K
    baseline = 1.0 / len(names)

    seed = 0
    print("Talks:", len(docs), " Topics:", len(names), " Vocab:", len(rec.vocab))
    print("Seed talk topic:", names[labels[seed]])
    print("Top-%d recommended topics: %s" % (K, [names[labels[j]] for j in rec.recommend(seed, k=K)]))
    print("Precision@%d (TF-IDF): %.3f" % (K, precision))
    print("Precision@%d (random): %.3f" % (K, baseline))
    print("Hit-Rate@%d:           %.3f" % (K, hit_rate))
    print("Improvement over random: %.1fx" % (precision / baseline))
    print("SUCCESS" if precision > 0.8 else "FAIL")
