import numpy as np


class VideoSearch:
    """TF-IDF text retrieval: rank videos for a query by cosine similarity."""

    @staticmethod
    def _tokenize(text):
        return text.lower().split()

    @staticmethod
    def _normalize(X):
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norm == 0, 1.0, norm)

    def fit(self, docs):
        self.docs = docs
        tokenized = [self._tokenize(d) for d in docs]

        # Vocabulary: every distinct token, mapped to a column index.
        vocab = sorted({t for toks in tokenized for t in toks})
        self.vocab = {w: i for i, w in enumerate(vocab)}
        V, N = len(vocab), len(docs)

        # Raw term counts (docs x vocab).
        tf = np.zeros((N, V))
        for i, toks in enumerate(tokenized):
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

    def _vectorize(self, text):
        # Map a query string into the same normalized TF-IDF space.
        v = np.zeros(len(self.vocab))
        for t in self._tokenize(text):
            j = self.vocab.get(t)
            if j is not None:
                v[j] += 1.0
        v = np.where(v > 0, 1.0 + np.log(np.maximum(v, 1.0)), 0.0) * self.idf
        return v / (np.linalg.norm(v) or 1.0)

    def search(self, query, k=5):
        # Cosine similarity to every video, return the top-k indices.
        scores = self.M @ self._vectorize(query)
        return np.argsort(-scores)[:k]


if __name__ == "__main__":
    np.random.seed(0)

    # Planted topics: each has a characteristic vocabulary.
    topics = {
        "cooking": "recipe kitchen bake oven chef pasta sauce grill dish tasty".split(),
        "gaming":  "gameplay level boss console controller loot quest speedrun raid".split(),
        "fitness": "workout gym muscle cardio squat reps trainer protein stretch".split(),
        "music":   "guitar chord melody album vocals drums studio remix concert".split(),
        "coding":  "python function loop array debug compiler syntax variable api".split(),
        "travel":  "flight hotel beach passport city tour backpack island map".split(),
    }
    generic = "video watch new best top guide tips channel".split()  # shared filler
    names = list(topics)

    # Synthetic videos: each belongs to one planted topic.
    docs, labels = [], []
    for ti, name in enumerate(names):
        words = topics[name]
        for _ in range(60):
            n = np.random.randint(8, 13)
            title = np.random.choice(words, size=n)      # topic tokens
            noise = np.random.choice(generic, size=2)    # generic tokens
            docs.append(" ".join(np.r_[title, noise]))
            labels.append(ti)
    labels = np.array(labels)

    vs = VideoSearch().fit(docs)

    # Queries drawn from a topic; measure Precision@K vs a random baseline.
    K, Q = 5, 200
    hits = 0.0
    for _ in range(Q):
        ti = np.random.randint(len(names))
        query = " ".join(np.random.choice(topics[names[ti]], size=3))
        idx = vs.search(query, k=K)
        hits += np.mean(labels[idx] == ti)              # fraction in query topic
    p_at_k = hits / Q
    baseline = 1.0 / len(names)                          # random precision

    print("Videos:", len(docs), " Topics:", len(names), " Vocab:", len(vs.vocab))
    print("Example query:", repr(query), "-> topic:", names[ti])
    print("Precision@{} (TF-IDF): {:.3f}".format(K, p_at_k))
    print("Precision@{} (random): {:.3f}".format(K, baseline))
    print("Improvement over random: {:.1f}x".format(p_at_k / baseline))
    print("SUCCESS" if p_at_k > 0.8 else "FAIL")
