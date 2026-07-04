import numpy as np


def tokenize(text):
    return text.lower().split()


class TfidfVectorizer:
    """Bag-of-words -> non-negative TF-IDF matrix, fit on the training corpus."""

    def fit(self, docs):
        self.vocab = {}
        for d in docs:
            for w in tokenize(d):
                self.vocab.setdefault(w, len(self.vocab))
        df = np.zeros(len(self.vocab))
        for d in docs:
            for w in set(tokenize(d)):
                df[self.vocab[w]] += 1.0
        n = len(docs)
        self.idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)))
        for i, d in enumerate(docs):
            for w in tokenize(d):
                j = self.vocab.get(w)
                if j is not None:
                    X[i, j] += 1.0                 # term counts (TF)
        X *= self.idf[None, :]                     # up-weight rare words
        return X                                   # kept non-negative for NMF


class NMFTopicModel:
    """Topic modeling by Non-negative Matrix Factorization, from scratch.

    Factor the D x V document-word matrix X ~= W @ H with W, H >= 0:
        W is D x K  -- how much each document uses each latent topic
        H is K x V  -- how much each topic uses each vocabulary word
    Non-negativity forces an additive, parts-based decomposition, so each row
    of H reads as an interpretable topic (a small set of co-occurring words).
    We minimize ||X - WH||_F^2 with Lee & Seung multiplicative updates, which
    are guaranteed non-increasing and keep every entry non-negative:
        H <- H * (Wt X)   / (Wt W H)
        W <- W * (X Ht)   / (W H Ht)"""

    def __init__(self, n_topics=4, iters=200, seed=0):
        self.K = n_topics
        self.iters = iters
        self.seed = seed
        self.eps = 1e-9

    def fit(self, X):
        rng = np.random.RandomState(self.seed)
        D, V = X.shape
        # Non-negative random init scaled to the data magnitude.
        scale = np.sqrt(X.mean() / self.K)
        self.W = np.abs(rng.rand(D, self.K)) * scale
        self.H = np.abs(rng.rand(self.K, V)) * scale
        self.errors = []
        for _ in range(self.iters):
            # Update H with W fixed, then W with H fixed.
            Wt = self.W.T
            self.H *= (Wt @ X) / (Wt @ self.W @ self.H + self.eps)
            Ht = self.H.T
            self.W *= (X @ Ht) / (self.W @ (self.H @ Ht) + self.eps)
            self.errors.append(np.linalg.norm(X - self.W @ self.H))
        return self

    def transform(self, X):
        # Project (possibly new) docs onto learned topics H via W updates only.
        rng = np.random.RandomState(self.seed + 1)
        W = np.abs(rng.rand(X.shape[0], self.K)) * np.sqrt(X.mean() / self.K)
        Ht = self.H.T
        HHt = self.H @ Ht
        for _ in range(60):
            W *= (X @ Ht) / (W @ HHt + self.eps)
        return W

    def dominant_topic(self, X):
        return np.argmax(self.transform(X), axis=1)

    def top_words(self, inv_vocab, topic, n=6):
        return [inv_vocab[j] for j in np.argsort(self.H[topic])[::-1][:n]]


def make_corpus(n=300, seed=0):
    # Synthetic corpus: each document is generated from ONE latent topic using
    # mostly that topic's words plus shared filler, so a real topic structure
    # exists in the word co-occurrences for NMF to recover (never seen labels).
    rng = np.random.RandomState(seed)
    topics = {
        0: ["planet", "galaxy", "orbit", "telescope", "cosmic", "star", "nebula", "asteroid"],
        1: ["recipe", "flavor", "roast", "kitchen", "cuisine", "spice", "baking", "dessert"],
        2: ["portfolio", "dividend", "market", "trading", "investor", "equity", "yield", "bond"],
        3: ["muscle", "workout", "cardio", "fitness", "training", "endurance", "athlete", "recovery"],
    }
    filler = ["the", "a", "and", "of", "is", "to", "in", "with", "this", "for"]
    names = list(topics.keys())

    docs, labels = [], []
    for _ in range(n):
        c = rng.choice(names)
        length = rng.randint(14, 22)
        n_topic = int(length * 0.62)                 # 62% topical, 38% filler
        words = list(rng.choice(topics[c], n_topic)) + \
            list(rng.choice(filler, length - n_topic))
        rng.shuffle(words)
        docs.append(" ".join(words))
        labels.append(c)
    return docs, np.array(labels)


def best_match_accuracy(y_true, z_pred, K):
    # Topics are unlabeled: score the best topic->class permutation mapping.
    from itertools import permutations
    best = 0.0
    for perm in permutations(range(K)):
        mapped = np.array([perm[z] for z in z_pred])
        best = max(best, np.mean(mapped == y_true))
    return best


if __name__ == "__main__":
    np.random.seed(0)

    docs, y = make_corpus(n=300, seed=0)
    topic_names = ["Astronomy", "Cooking", "Finance", "Fitness"]
    K = 4

    # Held-out split: learn topics on train, project test docs onto them.
    idx = np.random.permutation(len(docs))
    split = int(0.7 * len(docs))
    tr, te = idx[:split], idx[split:]
    tr_docs = [docs[i] for i in tr]
    te_docs = [docs[i] for i in te]

    vec = TfidfVectorizer().fit(tr_docs)
    inv_vocab = {j: w for w, j in vec.vocab.items()}
    Xtr, Xte = vec.transform(tr_docs), vec.transform(te_docs)

    nmf = NMFTopicModel(n_topics=K, iters=200, seed=0).fit(Xtr)
    z_te = nmf.dominant_topic(Xte)

    acc = best_match_accuracy(y[te], z_te, K)
    random_level = 1.0 / K                           # chance topic assignment

    print("Documents: %d   Train: %d   Test: %d   Vocab: %d   Topics: %d"
          % (len(docs), len(tr), len(te), len(vec.vocab), K))
    print("Reconstruction error: %.3f -> %.3f (Frobenius, decreasing)"
          % (nmf.errors[0], nmf.errors[-1]))
    print("-" * 60)
    print("Recovered topics (top words):")
    for k in range(K):
        print("  topic %d: %s" % (k, ", ".join(nmf.top_words(inv_vocab, k))))
    print("-" * 60)
    print("NMF held-out topic-recovery accuracy: %.4f" % acc)
    print("Random-assignment baseline          : %.4f" % random_level)
    print("-" * 60)
    for text in ["galaxy orbit telescope cosmic star nebula",
                 "recipe flavor roast kitchen spice baking",
                 "portfolio dividend market investor equity yield",
                 "muscle workout cardio fitness training endurance"]:
        z = nmf.dominant_topic(vec.transform([text]))[0]
        print("  '%s...' -> topic %d" % (text[:36], z))
    print("-" * 60)
    print("NMF beats random topic assignment: %s"
          % (acc > random_level + 0.30))
